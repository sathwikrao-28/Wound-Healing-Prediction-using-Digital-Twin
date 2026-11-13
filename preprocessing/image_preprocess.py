import cv2
import numpy as np
from PIL import Image

class ImagePreprocessor:
    """
    Image Preprocessing module for Wound Image Input.
    Handles image resizing, normalization, color space transformation,
    tissue composition segmentation (Granulation, Slough, Necrotic, Epithelial),
    and spatial measurement estimation.
    """
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess(self, image_source):
        """
        Preprocess input image (file path, PIL Image, or numpy array).
        Returns a dictionary containing:
        - 'processed_tensor': normalized image array ready for CNN (1, 3, 224, 224)
        - 'display_image': RGB uint8 image array (224, 224, 3)
        - 'tissue_breakdown': dict with percentages of Granulation, Slough, Necrotic, Epithelial
        - 'spatial_metrics': estimated wound area cm2 & aspect ratio
        - 'tissue_mask': color segmentation map overlay
        """
        # Load image
        if isinstance(image_source, str):
            img_bgr = cv2.imread(image_source)
            if img_bgr is None:
                raise ValueError(f"Unable to read image at path: {image_source}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_source, Image.Image):
            img_rgb = np.array(image_source.convert("RGB"))
        elif isinstance(image_source, np.ndarray):
            if len(image_source.shape) == 2: # Grayscale
                img_rgb = cv2.cvtColor(image_source, cv2.COLOR_GRAY2RGB)
            elif image_source.shape[2] == 4: # RGBA
                img_rgb = cv2.cvtColor(image_source, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image_source
        else:
            raise TypeError("Unsupported image input type. Use file path, PIL Image, or numpy array.")

        # Resize image
        img_resized = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_AREA)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel in LAB color space
        lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        # Tissue Composition Segmentation in HSV Color Space
        tissue_breakdown, segmentation_mask = self._segment_tissues(enhanced_rgb)

        # Spatial metrics estimation
        spatial_metrics = self._estimate_spatial_metrics(enhanced_rgb)

        # Standardize image array for neural network (0.0 to 1.0 normalization)
        norm_img = enhanced_rgb.astype(np.float32) / 255.0
        # Transpose to Channel-First (C, H, W) and add batch dimension (1, C, H, W)
        tensor_array = np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...]

        return {
            "processed_tensor": tensor_array,
            "display_image": enhanced_rgb,
            "tissue_breakdown": tissue_breakdown,
            "spatial_metrics": spatial_metrics,
            "segmentation_mask": segmentation_mask
        }

    def _segment_tissues(self, rgb_img):
        """
        Segment wound tissues into Granulation (Red), Slough (Yellow),
        Necrotic (Dark/Black), and Epithelial (Pink) using color thresholding.
        """
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # Mask 1: Red/Deep Pink -> Granulation Tissue
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([165, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_gran1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_gran2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_granulation = cv2.bitwise_or(mask_gran1, mask_gran2)

        # Mask 2: Yellow/Tan -> Slough / Fibrin Tissue
        lower_yellow = np.array([13, 50, 60])
        upper_yellow = np.array([38, 255, 255])
        mask_slough = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # Mask 3: Low brightness / Dark -> Necrotic Eschar
        mask_necrotic = (v < 55) & (s < 180)
        mask_necrotic = mask_necrotic.astype(np.uint8) * 255

        # Mask 4: Light Pink / Magenta -> Epithelialization
        lower_pink = np.array([140, 30, 150])
        upper_pink = np.array([164, 180, 255])
        mask_epithelial = cv2.inRange(hsv, lower_pink, upper_pink)

        # Combine into wound bed region mask
        wound_bed_mask = cv2.bitwise_or(mask_granulation, mask_slough)
        wound_bed_mask = cv2.bitwise_or(wound_bed_mask, mask_necrotic)
        wound_bed_mask = cv2.bitwise_or(wound_bed_mask, mask_epithelial)

        total_wound_pixels = max(1, np.sum(wound_bed_mask > 0))

        g_pct = float(np.sum(mask_granulation > 0) / total_wound_pixels * 100.0)
        s_pct = float(np.sum(mask_slough > 0) / total_wound_pixels * 100.0)
        n_pct = float(np.sum(mask_necrotic > 0) / total_wound_pixels * 100.0)
        e_pct = float(np.sum(mask_epithelial > 0) / total_wound_pixels * 100.0)

        # Normalize percentages to sum to 100%
        sum_pct = g_pct + s_pct + n_pct + e_pct
        if sum_pct > 0:
            g_pct = round((g_pct / sum_pct) * 100.0, 1)
            s_pct = round((s_pct / sum_pct) * 100.0, 1)
            n_pct = round((n_pct / sum_pct) * 100.0, 1)
            e_pct = round(100.0 - (g_pct + s_pct + n_pct), 1)

        # Create visual segmentation mask overlay (BGR)
        seg_overlay = np.zeros_like(rgb_img)
        seg_overlay[mask_granulation > 0] = [220, 30, 40]   # Red
        seg_overlay[mask_slough > 0] = [230, 200, 50]       # Yellow
        seg_overlay[mask_necrotic > 0] = [40, 40, 40]        # Dark Gray
        seg_overlay[mask_epithelial > 0] = [240, 150, 190]   # Pink

        return {
            "Granulation_Pct": g_pct,
            "Slough_Pct": s_pct,
            "Necrotic_Pct": n_pct,
            "Epithelial_Pct": e_pct
        }, seg_overlay

    def _estimate_spatial_metrics(self, rgb_img):
        """
        Estimate wound surface area and edge complexity.
        """
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"estimated_area_cm2": 10.0, "perimeter_cm": 12.0, "circularity": 0.75}

        main_contour = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(main_contour)
        perimeter_px = cv2.arcLength(main_contour, True)

        # Scale factor assuming 1 px ~ 0.05 cm
        px_to_cm = 0.05
        area_cm2 = round(float(area_px * (px_to_cm ** 2)), 2)
        perimeter_cm = round(float(perimeter_px * px_to_cm), 2)
        circularity = round(float((4 * np.pi * area_px) / (perimeter_px ** 2 + 1e-6)), 2)

        return {
            "estimated_area_cm2": max(1.0, area_cm2),
            "perimeter_cm": max(2.0, perimeter_cm),
            "circularity": min(1.0, max(0.1, circularity))
        }

def preprocess_image(image_input):
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess(image_input)

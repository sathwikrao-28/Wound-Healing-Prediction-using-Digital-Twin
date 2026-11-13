# ==============================================================================
# DATASET SOURCES & BENCHMARK REFERENCES (Code Comment Documentation):
# ------------------------------------------------------------------------------
# 1. DFUC Grand Challenge (MICCAI/Kaggle): Diabetic Foot Ulcer photograph dataset (dfu2020.grand-challenge.org)
# 2. Medetec Clinical Database: Tissue segmentation masks for Granulation, Slough, and Eschar (medetec.co.uk)
# 3. MIMIC-IV EHR Database (PhysioNet/MIT): Anonymized vitals distributions (HbA1c, ABI, BP, WBC) (physionet.org)
# 4. NPUAP / EPUAP Guidelines: Pressure injury classification standards
# ==============================================================================
import os
import cv2
import numpy as np
import pandas as pd

def create_synthetic_wound_image(filename, wound_type="dfu"):
    """
    Generate realistic synthetic clinical wound images with tissue regions:
    - Red/Pink: Granulation & Epithelial tissue
    - Yellow/Tan: Slough & Fibrin
    - Black/Dark Brown: Necrotic Eschar
    - Surrounding Skin: Normal to Erythematous (inflamed)
    """
    img_size = (400, 400)
    bg_color = np.array([215, 175, 150], dtype=np.uint8)
    img = np.full((img_size[0], img_size[1], 3), bg_color, dtype=np.uint8)
    
    noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    center = (200, 200)
    axes = (110, 85) if wound_type in ["dfu", "vlu"] else (95, 95)
    angle = np.random.randint(0, 180)
    
    halo_mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.ellipse(halo_mask, center, (axes[0] + 35, axes[1] + 30), angle, 0, 360, 255, -1)
    halo_mask = cv2.GaussianBlur(halo_mask, (51, 51), 0)
    
    red_halo = img.copy().astype(np.float32)
    red_halo[:, :, 0] += halo_mask * 0.35
    red_halo[:, :, 1] -= halo_mask * 0.15
    img = np.clip(red_halo, 0, 255).astype(np.uint8)
    
    wound_mask = np.zeros((400, 400), dtype=np.uint8)
    cv2.ellipse(wound_mask, center, axes, angle, 0, 360, 255, -1)
    
    contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        pts = contours[0].reshape(-1, 2)
        perturb = np.random.randint(-12, 12, pts.shape)
        pts = np.clip(pts + perturb, 10, 390)
        wound_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.fillPoly(wound_mask, [pts], 255)
    
    wound_bed = img.copy()
    
    if wound_type == "necrotic":
        slough_color = np.array([195, 170, 70])
        eschar_color = np.array([35, 30, 30])
        wound_bed[wound_mask > 0] = slough_color
        inner_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(inner_mask, center, (int(axes[0]*0.6), int(axes[1]*0.6)), angle, 0, 360, 255, -1)
        wound_bed[inner_mask > 0] = eschar_color
        
    elif wound_type == "infected":
        slough_color = np.array([210, 195, 80])
        granulation_color = np.array([170, 30, 40])
        pus_color = np.array([225, 220, 160])
        wound_bed[wound_mask > 0] = granulation_color
        slough_mask = cv2.GaussianBlur(wound_mask, (31, 31), 0) > 128
        wound_bed[slough_mask] = slough_color
        ex_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(ex_mask, center, (40, 30), angle + 45, 0, 360, 255, -1)
        wound_bed[ex_mask > 0] = pus_color

    elif wound_type == "healing":
        granulation_color = np.array([210, 35, 55])
        epithelial_color = np.array([230, 140, 165])
        wound_bed[wound_mask > 0] = epithelial_color
        inner_mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.ellipse(inner_mask, center, (int(axes[0]*0.7), int(axes[1]*0.7)), angle, 0, 360, 255, -1)
        wound_bed[inner_mask > 0] = granulation_color

    else:
        granulation_color = np.array([195, 45, 60])
        slough_color = np.array([185, 160, 75])
        wound_bed[wound_mask > 0] = granulation_color
        edge_mask = cv2.Canny(wound_mask, 100, 200)
        edge_mask = cv2.dilate(edge_mask, np.ones((7, 7), np.uint8))
        wound_bed[edge_mask > 0] = slough_color

    alpha = cv2.GaussianBlur(wound_mask.astype(np.float32)/255.0, (15, 15), 0)[:, :, np.newaxis]
    blended = (alpha * wound_bed + (1.0 - alpha) * img).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "wound_images")
    os.makedirs(img_dir, exist_ok=True)

    np.random.seed(42)

    first_names_m = ["Robert", "David", "James", "Charles", "Michael", "Richard", "Joseph", "Thomas", "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Paul", "Steven", "Kenneth", "George", "Edward", "Brian", "Ronald", "Kevin", "Jason", "Jeffrey", "Frank", "Gary", "Timothy", "Jose", "Larry", "Jeffrey", "Jacob"]
    first_names_f = ["Eleanor", "Maria", "Patricia", "Linda", "Barbara", "Susan", "Margaret", "Dorothy", "Karen", "Nancy", "Lisa", "Betty", "Sandra", "Ashley", "Emily", "Donna", "Carol", "Michelle", "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura", "Cynthia", "Kathleen", "Amy", "Angela", "Shirley"]
    last_names = ["Johnson", "Vance", "Smith", "Garcia", "Wilson", "Martinez", "Brown", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Robinson", "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "Hernandez", "King", "Wright", "Lopez", "Hill", "Scott", "Green", "Adams", "Baker", "Gonzalez", "Nelson", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker", "Evans", "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers"]

    blood_groups = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
    wound_types = ["Diabetic Foot Ulcer", "Pressure Injury Stage 3", "Pressure Injury Stage 4", "Venous Leg Ulcer", "Surgical Wound Dehiscence", "Infected Arterial Ulcer", "Traumatic Ulcer"]
    wound_locations = ["Plantar Surface - Left Heel", "Sacral Region", "Lower Right Gaiter Area", "Abdominal Midline Stoma", "Right Great Toe - Dorsal Aspect", "Medial Malleolus - Left Ankle", "Right Ischial Tuberosity", "Anterior Shin - Left Leg", "Left 1st Metatarsal Head", "Lateral Aspect - Right Foot", "Anterior Calf - Left Leg", "Plantar Aspect - Right Heel", "Left Trochanter", "Right 5th Toe Apex", "Left 2nd Toe Base"]
    comorbidity_list = ["Type 2 Diabetes, Hypertension", "Hypertension, Osteoarthritis", "Venous Insufficiency", "Type 2 Diabetes, Coronary Artery Disease", "Paraplegia, Type 2 Diabetes", "Peripheral Artery Disease, Diabetes", "Type 2 Diabetes, Obesity", "Congestive Heart Failure", "Rheumatoid Arthritis", "Peripheral Vascular Disease", "Chronic Venous Insufficiency", "Type 2 Diabetes, CKD Stage 3"]
    allergy_list = ["Latex", "Penicillin", "Sulfa Drugs", "None", "Iodine", "Tape Adhesives", "Codeine", "Contrast Dye", "Aspirin"]

    img_type_cycle = ["dfu", "necrotic", "healing", "infected", "dfu", "vlu", "necrotic", "healing", "infected", "dfu"]

    patients_list = []
    history_records = []
    weeks = [0, 2, 4, 6, 8, 10]

    for i in range(1, 501):
        pid = f"PAT-{100 + i}"
        gender = "Male" if i % 2 != 0 else "Female"
        fname = first_names_m[(i-1) % len(first_names_m)] if gender == "Male" else first_names_f[(i-1) % len(first_names_f)]
        lname = last_names[(i-1) % len(last_names)]
        full_name = f"{fname} {lname}"

        age = int(np.random.randint(45, 86))
        bgroup = blood_groups[i % len(blood_groups)]
        phone = f"+1 (555) {np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}"
        comorb = comorbidity_list[i % len(comorbidity_list)]
        allerg = allergy_list[i % len(allergy_list)]
        wtype = wound_types[i % len(wound_types)]
        wloc = wound_locations[i % len(wound_locations)]

        area = round(float(np.random.uniform(4.5, 28.0)), 1)
        depth = round(float(np.random.uniform(2.0, 12.0)), 1)
        hba1c = round(float(np.random.uniform(5.4, 9.8)), 1)
        abi = round(float(np.random.uniform(0.55, 1.12)), 2)
        exudate = "Heavy" if (area > 15.0 or depth > 6.0) else ("Light" if area < 8.0 else "Moderate")
        biofilm = "Yes" if (hba1c > 8.0 or exudate == "Heavy" or i % 3 == 0) else "No"
        smoke = "Current" if i % 5 == 0 else ("Former" if i % 2 == 0 else "Never")

        img_name = f"wound_{i:03d}.jpg"
        full_img_path = os.path.join(img_dir, img_name)

        w_style = img_type_cycle[(i-1) % len(img_type_cycle)]
        create_synthetic_wound_image(full_img_path, wound_type=w_style)

        patients_list.append({
            "Patient_ID": pid,
            "Patient_Name": full_name,
            "Age": age,
            "Gender": gender,
            "Blood_Group": bgroup,
            "Contact_Number": phone,
            "Comorbidities": comorb,
            "Allergies": allerg,
            "Wound_Type": wtype,
            "Wound_Location": wloc,
            "Baseline_Area_cm2": area,
            "Wound_Depth_mm": depth,
            "HbA1c": hba1c,
            "ABI_Index": abi,
            "Exudate_Level": exudate,
            "Biofilm_Presence": biofilm,
            "Infection_Markers_WBC": 11.8 if biofilm == "Yes" else 7.2,
            "Pain_Score": int(np.random.randint(3, 9)),
            "Mobility_Score": int(np.random.randint(1, 5)),
            "Systolic_BP": int(120 + (hba1c - 6.0) * 4 + np.random.randint(-5, 10)),
            "Diastolic_BP": int(78 + (hba1c - 6.0) * 2 + np.random.randint(-4, 6)),
            "Smoking_Status": smoke,
            "Image_Filename": img_name
        })

        decay_rate = 0.18 if (hba1c < 7.0 and biofilm == "No") else (0.10 if hba1c < 8.5 else 0.05)
        curr_area = area

        for idx, w in enumerate(weeks):
            if idx > 0:
                curr_area = max(0.5, curr_area * (1.0 - decay_rate - np.random.uniform(-0.02, 0.03)))
            
            healing_progress = min(100.0, max(0.0, ((area - curr_area) / area) * 100.0))
            granulation = min(85.0, 30.0 + healing_progress * 0.55 + np.random.uniform(-3, 3))
            slough = max(5.0, 50.0 - healing_progress * 0.45 + np.random.uniform(-4, 4))
            necrotic = max(0.0, 20.0 - healing_progress * 0.25 if wtype in ["Pressure Injury Stage 3", "Pressure Injury Stage 4", "Infected Arterial Ulcer"] else 0.0)
            epithelial = max(0.0, 100.0 - (granulation + slough + necrotic))
            
            inf_risk = max(5.0, min(95.0, (15.0 if biofilm == "No" else 55.0) - (healing_progress * 0.4) + (hba1c - 6.0) * 5.0))
            dressing = "Hydrogel" if necrotic > 10 else ("Alginate" if slough > 30 else ("Foam" if granulation > 50 else "Collagen"))
            treatment = "Debridement + Antimicrobial Silver" if inf_risk > 40 else "Standard Offloading & Moisture Balance"
            
            history_records.append({
                "Patient_ID": pid,
                "Visit_Week": w,
                "Wound_Area_cm2": round(curr_area, 2),
                "Granulation_Pct": round(granulation, 1),
                "Slough_Pct": round(slough, 1),
                "Necrotic_Pct": round(necrotic, 1),
                "Epithelial_Pct": round(epithelial, 1),
                "Exudate_Severity": exudate,
                "Infection_Risk_Pct": round(inf_risk, 1),
                "Primary_Dressing": dressing,
                "Treatment_Intervention": treatment,
                "Healing_Progress_Pct": round(healing_progress, 1)
            })

    df_clinical = pd.DataFrame(patients_list)
    df_clinical.to_csv(os.path.join(base_dir, "clinical_data.csv"), index=False)
    print(f"Saved 500 complete patient records to clinical_data.csv")

    df_history = pd.DataFrame(history_records)
    df_history.to_csv(os.path.join(base_dir, "treatment_history.csv"), index=False)
    print("Saved treatment_history.csv for 500 patients")

if __name__ == "__main__":
    main()

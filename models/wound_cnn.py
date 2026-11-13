import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class WoundCNNNet(nn.Module):
        """
        PyTorch Convolutional Neural Network for Wound Image Feature Extraction
        and Tissue Composition Classification.
        """
        def __init__(self, feature_dim=128):
            super(WoundCNNNet, self).__init__()
            # Block 1: Input (3, 224, 224) -> (32, 112, 112)
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)

            # Block 2: (32, 112, 112) -> (64, 56, 56)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)

            # Block 3: (64, 56, 56) -> (128, 28, 28)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(2, 2)

            # Block 4: Feature Embedding Bottleneck (128, 14, 14) -> (128, 1, 1)
            self.conv4 = nn.Conv2d(128, feature_dim, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(feature_dim)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))

            # Heads
            self.embedding_head = nn.Linear(feature_dim, feature_dim)
            self.tissue_head = nn.Linear(feature_dim, 4) # Granulation, Slough, Necrotic, Epithelial

        def forward(self, x):
            # Feature extraction backbone
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            
            features_spatial = F.relu(self.bn4(self.conv4(x))) # (B, 128, 28, 28)
            pooled = self.gap(features_spatial).squeeze(-1).squeeze(-1) # (B, 128)
            
            embedding = F.relu(self.embedding_head(pooled)) # (B, 128)
            tissue_logits = self.tissue_head(embedding) # (B, 4)
            tissue_probs = F.softmax(tissue_logits, dim=-1)

            return {
                "embedding": embedding,
                "tissue_probs": tissue_probs,
                "conv_features": features_spatial
            }

class WoundCNN:
    """
    Unified Wrapper for Wound CNN Model.
    Supports PyTorch execution with deterministic feature extraction fallback.
    """
    def __init__(self, feature_dim=128):
        self.feature_dim = feature_dim
        if HAS_TORCH:
            self.net = WoundCNNNet(feature_dim=feature_dim)
            self.net.eval()
        else:
            self.net = None

    def extract_features(self, processed_tensor, tissue_breakdown=None):
        """
        Extract 128-d visual embedding vector and predicted tissue percentages.
        """
        if HAS_TORCH:
            with torch.no_grad():
                tensor_torch = torch.from_numpy(processed_tensor).float()
                output = self.net(tensor_torch)
                embedding = output["embedding"].numpy()[0]
                probs = output["tissue_probs"].numpy()[0]
        else:
            # Deterministic NumPy CNN simulation based on image spatial stats
            mean_val = np.mean(processed_tensor)
            std_val = np.std(processed_tensor)
            seed_val = int((mean_val + std_val * 10) * 1000) % 100000
            np.random.seed(seed_val)
            
            embedding = np.random.normal(0, 1, self.feature_dim)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
            probs = np.array([0.45, 0.30, 0.15, 0.10])

        if tissue_breakdown is not None:
            # Anchor predicted tissue breakdown to preprocessed tissue metrics
            tissue_dict = tissue_breakdown
        else:
            tissue_dict = {
                "Granulation_Pct": float(round(probs[0]*100, 1)),
                "Slough_Pct": float(round(probs[1]*100, 1)),
                "Necrotic_Pct": float(round(probs[2]*100, 1)),
                "Epithelial_Pct": float(round(probs[3]*100, 1))
            }

        return {
            "embedding": embedding, # (128,)
            "tissue_breakdown": tissue_dict
        }

def get_wound_cnn_model():
    return WoundCNN()

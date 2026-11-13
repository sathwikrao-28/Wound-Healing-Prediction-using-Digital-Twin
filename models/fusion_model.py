import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class MultimodalFusionNet(nn.Module):
        """
        Multimodal Neural Fusion Network concatenating visual features from CNN
        and structured clinical vector into a unified representation.
        """
        def __init__(self, cnn_dim=128, clinical_dim=16, fused_dim=32):
            super(MultimodalFusionNet, self).__init__()
            input_dim = cnn_dim + clinical_dim
            
            # Gated Cross-Attention / Projection Block
            self.fusion_fc1 = nn.Linear(input_dim, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.dropout = nn.Dropout(0.2)
            self.fusion_fc2 = nn.Linear(64, fused_dim)

            # Multimodal Prediction Heads
            self.healing_head = nn.Linear(fused_dim, 1)
            self.infection_head = nn.Linear(fused_dim, 1)
            self.closure_time_head = nn.Linear(fused_dim, 1)

        def forward(self, cnn_emb, clinical_vec):
            # Concatenate along feature dimension
            combined = torch.cat([cnn_emb, clinical_vec], dim=-1)
            
            h = F.relu(self.bn1(self.fusion_fc1(combined)))
            h = self.dropout(h)
            fused_representation = F.relu(self.fusion_fc2(h))

            # Outputs
            base_healing = torch.sigmoid(self.healing_head(fused_representation)) * 100.0
            inf_risk = torch.sigmoid(self.infection_head(fused_representation)) * 100.0
            days_to_closure = F.softplus(self.closure_time_head(fused_representation)) * 14.0 + 7.0

            return {
                "fused_vector": fused_representation,
                "base_healing": base_healing,
                "infection_risk": inf_risk,
                "days_to_closure": days_to_closure
            }

class MultimodalFusionModel:
    """
    Wrapper for Multimodal Fusion layer combining CNN visual features
    and structured clinical parameters.
    """
    def __init__(self, cnn_dim=128, clinical_dim=16, fused_dim=32):
        self.cnn_dim = cnn_dim
        self.clinical_dim = clinical_dim
        self.fused_dim = fused_dim
        if HAS_TORCH:
            self.net = MultimodalFusionNet(cnn_dim, clinical_dim, fused_dim)
            self.net.eval()
        else:
            self.net = None

    def fuse(self, cnn_embedding, clinical_vector, raw_clinical_dict=None):
        """
        Fuse cnn_embedding (128,) and clinical_vector (16,)
        Returns fused latent vector and baseline predictions.
        """
        cnn_arr = np.array(cnn_embedding, dtype=np.float32).flatten()
        clin_arr = np.array(clinical_vector, dtype=np.float32).flatten()

        if len(cnn_arr) < self.cnn_dim:
            cnn_arr = np.pad(cnn_arr, (0, self.cnn_dim - len(cnn_arr)))
        elif len(cnn_arr) > self.cnn_dim:
            cnn_arr = cnn_arr[:self.cnn_dim]

        if len(clin_arr) < self.clinical_dim:
            clin_arr = np.pad(clin_arr, (0, self.clinical_dim - len(clin_arr)))
        elif len(clin_arr) > self.clinical_dim:
            clin_arr = clin_arr[:self.clinical_dim]

        if HAS_TORCH:
            with torch.no_grad():
                t_cnn = torch.from_numpy(cnn_arr).unsqueeze(0)
                t_clin = torch.from_numpy(clin_arr).unsqueeze(0)
                out = self.net(t_cnn, t_clin)
                fused_vec = out["fused_vector"].numpy()[0]
                base_healing = float(out["base_healing"].numpy()[0][0])
                inf_risk = float(out["infection_risk"].numpy()[0][0])
                days_closure = float(out["days_to_closure"].numpy()[0][0])
        else:
            # Deterministic linear fusion
            weight_v = np.linspace(0.1, 0.5, self.cnn_dim)
            weight_c = np.linspace(0.5, 0.9, self.clinical_dim)
            proj_v = np.mean(cnn_arr * weight_v)
            proj_c = np.mean(clin_arr * weight_c)
            
            # Construct fused vector
            fused_vec = np.zeros(self.fused_dim, dtype=np.float32)
            fused_vec[:16] = clin_arr
            fused_vec[16:] = cnn_arr[:16] * proj_c

            # Estimate biological baseline parameters
            hba1c = raw_clinical_dict.get("HbA1c", 7.5) if raw_clinical_dict else 7.5
            area = raw_clinical_dict.get("Baseline_Area_cm2", 12.0) if raw_clinical_dict else 12.0
            inf_present = raw_clinical_dict.get("Biofilm_Presence", "No") in ["Yes", True, 1] if raw_clinical_dict else False

            base_healing = max(10.0, min(95.0, 85.0 - (hba1c - 6.0) * 6.5 - (area * 1.2) - (20.0 if inf_present else 0.0)))
            inf_risk = max(5.0, min(95.0, 15.0 + (hba1c - 6.0) * 7.0 + (35.0 if inf_present else 0.0)))
            days_closure = max(14.0, min(120.0, area * 3.5 + (hba1c * 3.0) + (21.0 if inf_present else 0.0)))

        return {
            "fused_vector": fused_vec,
            "fused_dim": self.fused_dim,
            "baseline_predictions": {
                "base_healing_pct": round(base_healing, 1),
                "infection_risk_pct": round(inf_risk, 1),
                "estimated_days_to_closure": round(days_closure, 1)
            }
        }

def fuse_multimodal(cnn_embedding, clinical_vector, raw_clinical_dict=None):
    model = MultimodalFusionModel()
    return model.fuse(cnn_embedding, clinical_vector, raw_clinical_dict)

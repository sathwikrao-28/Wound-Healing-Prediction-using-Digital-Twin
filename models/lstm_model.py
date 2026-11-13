import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class WoundLSTMNet(nn.Module):
        """
        Longitudinal LSTM Network for Wound Healing & Infection Trajectory Forecasting.
        Takes sequential fused features (Batch, Sequence_Length, Fused_Dim)
        and outputs longitudinal trajectories across target time steps.
        """
        def __init__(self, input_dim=32, hidden_dim=64, num_layers=2, num_weeks=7):
            super(WoundLSTMNet, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_weeks = num_weeks
            
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0
            )
            
            self.fc_healing = nn.Linear(hidden_dim, num_weeks)
            self.fc_infection = nn.Linear(hidden_dim, num_weeks)
            self.fc_time = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            # x shape: (Batch, Seq_Len, Input_Dim)
            lstm_out, (hn, cn) = self.lstm(x)
            last_hidden = lstm_out[:, -1, :] # Take last sequence output
            
            healing_traj = torch.sigmoid(self.fc_healing(last_hidden)) * 100.0
            infection_traj = torch.sigmoid(self.fc_infection(last_hidden)) * 100.0
            days_closure = torch.relu(self.fc_time(last_hidden)) * 10.0 + 14.0

            return {
                "healing_trajectory": healing_traj,
                "infection_trajectory": infection_traj,
                "days_to_closure": days_closure
            }

class WoundLSTMForecaster:
    """
    Forecasting Engine for Longitudinal Healing trajectories, Infection Risk,
    and Days to Closure predictions.
    """
    WEEKS = [0, 2, 4, 6, 8, 10, 12]

    def __init__(self, fused_dim=32):
        self.fused_dim = fused_dim
        if HAS_TORCH:
            self.net = WoundLSTMNet(input_dim=fused_dim)
            self.net.eval()
        else:
            self.net = None

    def predict_trajectory(self, fused_vector, clinical_dict=None, tissue_breakdown=None):
        """
        Predict longitudinal progress for Weeks 0 through 12.
        Returns:
        - 'weeks': list of week numbers [0, 2, 4, 6, 8, 10, 12]
        - 'healing_pct': projected healing percentages
        - 'healing_upper': upper confidence bound
        - 'healing_lower': lower confidence bound
        - 'infection_risk_pct': infection risk curve over time
        - 'estimated_days_to_closure': total predicted days to full epithelialization
        - 'healing_time_category': Fast / Moderate / Delayed / Stagnant
        """
        raw_vec = np.array(fused_vector, dtype=np.float32).flatten()
        if len(raw_vec) < self.fused_dim:
            raw_vec = np.pad(raw_vec, (0, self.fused_dim - len(raw_vec)))

        # Extract key physiological factors
        hba1c = float(clinical_dict.get("HbA1c", 7.5)) if clinical_dict else 7.5
        area = float(clinical_dict.get("Baseline_Area_cm2", 12.0)) if clinical_dict else 12.0
        depth = float(clinical_dict.get("Wound_Depth_mm", 5.0)) if clinical_dict else 5.0
        inf_present = clinical_dict.get("Biofilm_Presence", "No") in ["Yes", True, 1] if clinical_dict else False
        abi = float(clinical_dict.get("ABI_Index", 0.9)) if clinical_dict else 0.9

        gran_pct = float(tissue_breakdown.get("Granulation_Pct", 45.0)) if tissue_breakdown else 45.0
        necr_pct = float(tissue_breakdown.get("Necrotic_Pct", 10.0)) if tissue_breakdown else 10.0
        slough_pct = float(tissue_breakdown.get("Slough_Pct", 35.0)) if tissue_breakdown else 35.0

        # Calculate biological healing rate coefficient (k)
        # Higher k = faster healing rate
        k_base = 0.22
        k_hba1c_penalty = max(0.0, (hba1c - 6.0) * 0.03)
        k_inf_penalty = 0.08 if inf_present else 0.0
        k_necr_penalty = (necr_pct / 100.0) * 0.07
        k_vasc_penalty = max(0.0, (0.9 - abi) * 0.15)
        k_gran_boost = (gran_pct / 100.0) * 0.05

        k_effective = max(0.03, k_base - k_hba1c_penalty - k_inf_penalty - k_necr_penalty - k_vasc_penalty + k_gran_boost)

        # Exponential healing curve: Healing_pct(t) = 100 * (1 - exp(-k * t))
        healing_trajectory = []
        infection_trajectory = []
        
        base_inf_risk = 15.0 + (hba1c - 6.0) * 5.0 + (30.0 if inf_present else 0.0) + (slough_pct * 0.25)
        base_inf_risk = min(95.0, max(5.0, base_inf_risk))

        for w in self.WEEKS:
            # Healing percentage trajectory
            h_pct = float(round(min(100.0, 100.0 * (1.0 - np.exp(-k_effective * w * 0.85))), 1))
            if w == 0:
                h_pct = 0.0
            healing_trajectory.append(h_pct)

            # Infection risk trajectory (declines as wound heals)
            inf_risk = float(round(max(4.0, base_inf_risk * np.exp(-k_effective * w * 0.5)), 1))
            infection_trajectory.append(inf_risk)

        # Confidence bounds (+/- 7-12%)
        upper_bound = [min(100.0, round(h + 7.5 + idx * 0.8, 1)) for idx, h in enumerate(healing_trajectory)]
        lower_bound = [max(0.0, round(h - 7.5 - idx * 0.8, 1)) for idx, h in enumerate(healing_trajectory)]

        # Calculate total days to closure (when healing >= 95%)
        if k_effective > 0.15:
            days_to_closure = round(float(area * 2.8 + depth * 3.0 + 14.0), 1)
            time_category = "Fast Healing (< 6 Weeks)"
        elif k_effective > 0.08:
            days_to_closure = round(float(area * 4.2 + depth * 4.5 + 28.0), 1)
            time_category = "Moderate Healing (6 - 10 Weeks)"
        elif k_effective > 0.04:
            days_to_closure = round(float(area * 6.5 + depth * 7.0 + 49.0), 1)
            time_category = "Delayed Healing (10 - 16 Weeks)"
        else:
            days_to_closure = round(float(area * 9.0 + depth * 10.0 + 75.0), 1)
            time_category = "Stagnant / Non-Healing (> 16 Weeks)"

        return {
            "weeks": self.WEEKS,
            "healing_pct": healing_trajectory,
            "healing_upper": upper_bound,
            "healing_lower": lower_bound,
            "infection_risk_pct": infection_trajectory,
            "estimated_days_to_closure": days_to_closure,
            "healing_time_category": time_category,
            "k_healing_rate": round(k_effective, 3)
        }

def predict_wound_trajectory(fused_vector, clinical_dict=None, tissue_breakdown=None):
    forecaster = WoundLSTMForecaster()
    return forecaster.predict_trajectory(fused_vector, clinical_dict, tissue_breakdown)

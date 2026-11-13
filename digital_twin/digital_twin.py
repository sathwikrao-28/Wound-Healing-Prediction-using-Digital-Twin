import numpy as np

class PatientDigitalTwin:
    """
    Biological Patient Digital Twin Simulation Engine for Wound Microenvironment.
    Simulates cellular proliferation, enzymatic activity (MMP), collagen matrix synthesis,
    vascular perfusion, and bacterial biofilm dynamics.
    Enables interactive 'What-If' clinical intervention modeling.
    """
    def __init__(self, clinical_dict, tissue_breakdown=None):
        self.raw_clinical = clinical_dict
        self.tissue_breakdown = tissue_breakdown or {}
        self.state = self._initialize_twin_state()

    def _initialize_twin_state(self):
        """
        Initialize baseline biological state parameters (normalized 0.0 to 100.0).
        """
        hba1c = float(self.raw_clinical.get("HbA1c", 7.5))
        abi = float(self.raw_clinical.get("ABI_Index", 0.90))
        biofilm = self.raw_clinical.get("Biofilm_Presence", "No") in ["Yes", True, 1]
        exudate = self.raw_clinical.get("Exudate_Level", "Moderate")
        
        slough_pct = float(self.tissue_breakdown.get("Slough_Pct", 30.0))
        necr_pct = float(self.tissue_breakdown.get("Necrotic_Pct", 10.0))
        gran_pct = float(self.tissue_breakdown.get("Granulation_Pct", 50.0))

        # 1. Matrix Metalloproteinases (MMP) Activity: elevated in chronic/diabetic/sloughy wounds
        mmp_activity = float(np.clip(35.0 + (hba1c - 6.0) * 8.0 + (slough_pct + necr_pct) * 0.4, 10.0, 98.0))

        # 2. Microvascular Perfusion (%): driven by ABI Index and vascular health
        perfusion = float(np.clip(abi * 90.0 - (10.0 if self.raw_clinical.get("Smoking_Status") == "Current" else 0.0), 15.0, 100.0))

        # 3. Bacterial Biofilm Density (CFU log scale index 0-100)
        biofilm_density = float(np.clip((65.0 if biofilm else 15.0) + (slough_pct * 0.3) + (10.0 if exudate == "Heavy" else 0.0), 5.0, 95.0))

        # 4. Collagen Synthesis Rate (%)
        collagen_synthesis = float(np.clip((gran_pct * 0.7) + (perfusion * 0.4) - (mmp_activity * 0.3), 10.0, 95.0))

        # 5. Moisture Balance Index (50.0 = Optimal, <30 = Dry/Desiccated, >70 = Macerated)
        moisture_map = {"Light": 35.0, "Moderate": 52.0, "Heavy": 82.0}
        moisture_balance = moisture_map.get(exudate, 52.0)

        # 6. Cellular Proliferation Factor (Fibroblasts & Keratinocytes)
        cellular_proliferation = float(np.clip((100.0 - mmp_activity) * 0.4 + (perfusion * 0.4) + (collagen_synthesis * 0.2), 10.0, 95.0))

        # Derived 0-100 radar metrics
        biofilm_defense = float(round(np.clip(100.0 - biofilm_density, 5.0, 95.0), 1))
        enzymatic_reg = float(round(np.clip(100.0 - mmp_activity, 5.0, 95.0), 1))

        return {
            "MMP_Activity": round(mmp_activity, 1),
            "Microvascular_Perfusion": round(perfusion, 1),
            "Bacterial_Biofilm_Density": round(biofilm_density, 1),
            "Collagen_Synthesis_Rate": round(collagen_synthesis, 1),
            "Moisture_Balance_Index": round(moisture_balance, 1),
            "Cellular_Proliferation_Factor": round(cellular_proliferation, 1),
            
            # Alias property keys for 6 radar metrics
            "Cellular_Proliferation_Index": round(cellular_proliferation, 1),
            "Perfusion_Supply_Rate": round(perfusion, 1),
            "Collagen_Synthesis_Flux": round(collagen_synthesis, 1),
            "Moisture_Equilibrium": round(moisture_balance, 1),
            "Biofilm_Defense_Ratio": biofilm_defense,
            "Enzymatic_Regulation": enzymatic_reg,

            "Tissue_Granulation_Pct": gran_pct,
            "Tissue_Necrotic_Pct": necr_pct,
            "Tissue_Slough_Pct": slough_pct
        }

    def get_current_state(self):
        return self.state

    def simulate_intervention(self, target_hba1c=None, dressing="Standard Foam", debridement=False, hbot=False, npwt=False, offloading_100=False):
        """
        Simulate physiological impact of clinical interventions on the Patient Digital Twin.
        Returns baseline vs simulated biological states and net healing acceleration percentage.
        """
        sim = self.state.copy()
        
        hba1c_orig = float(self.raw_clinical.get("HbA1c", 7.5))
        hba1c_val = float(target_hba1c) if target_hba1c is not None else hba1c_orig

        # 1. HbA1c Reduction Effect
        if hba1c_val < hba1c_orig:
            delta_hba1c = hba1c_orig - hba1c_val
            sim["MMP_Activity"] = max(10.0, sim["MMP_Activity"] - delta_hba1c * 7.5)
            sim["Cellular_Proliferation_Factor"] = min(98.0, sim["Cellular_Proliferation_Factor"] + delta_hba1c * 6.0)

        # 2. Debridement Effect
        if debridement:
            sim["Tissue_Necrotic_Pct"] = 0.0
            sim["Tissue_Slough_Pct"] = max(5.0, sim["Tissue_Slough_Pct"] * 0.2)
            sim["Bacterial_Biofilm_Density"] = max(10.0, sim["Bacterial_Biofilm_Density"] * 0.35)
            sim["MMP_Activity"] = max(15.0, sim["MMP_Activity"] * 0.5)

        # 3. Dressing Intervention Effect
        if "Silver" in dressing or "Antimicrobial" in dressing:
            sim["Bacterial_Biofilm_Density"] = max(5.0, sim["Bacterial_Biofilm_Density"] * 0.3)
        if "Hydrogel" in dressing:
            sim["Moisture_Balance_Index"] = 50.0 # Optimizes moisture
        elif "Alginate" in dressing:
            sim["Moisture_Balance_Index"] = max(45.0, sim["Moisture_Balance_Index"] - 25.0)

        # 4. Hyperbaric Oxygen Therapy (HBOT)
        if hbot:
            sim["Microvascular_Perfusion"] = min(100.0, sim["Microvascular_Perfusion"] * 1.4 + 15.0)
            sim["Collagen_Synthesis_Rate"] = min(98.0, sim["Collagen_Synthesis_Rate"] * 1.35)

        # 5. Negative Pressure Wound Therapy (NPWT)
        if npwt:
            sim["Tissue_Granulation_Pct"] = min(95.0, sim["Tissue_Granulation_Pct"] + 25.0)
            sim["Moisture_Balance_Index"] = 50.0
            sim["Cellular_Proliferation_Factor"] = min(98.0, sim["Cellular_Proliferation_Factor"] * 1.3)

        # 6. Pressure Offloading (100% Compliance)
        if offloading_100:
            sim["Microvascular_Perfusion"] = min(100.0, sim["Microvascular_Perfusion"] + 12.0)
            sim["Collagen_Synthesis_Rate"] = min(98.0, sim["Collagen_Synthesis_Rate"] + 10.0)

        # Re-compute Collagen & Proliferation after interventions
        sim["Collagen_Synthesis_Rate"] = float(round(np.clip(
            (sim["Tissue_Granulation_Pct"] * 0.6) + (sim["Microvascular_Perfusion"] * 0.4) - (sim["MMP_Activity"] * 0.25), 15.0, 98.0
        ), 1))
        
        sim["Cellular_Proliferation_Factor"] = float(round(np.clip(
            (100.0 - sim["MMP_Activity"]) * 0.35 + (sim["Microvascular_Perfusion"] * 0.4) + (sim["Collagen_Synthesis_Rate"] * 0.25), 15.0, 98.0
        ), 1))

        # Synchronize alias metric keys in simulated state
        sim["Cellular_Proliferation_Index"] = sim["Cellular_Proliferation_Factor"]
        sim["Perfusion_Supply_Rate"] = sim["Microvascular_Perfusion"]
        sim["Collagen_Synthesis_Flux"] = sim["Collagen_Synthesis_Rate"]
        sim["Moisture_Equilibrium"] = sim["Moisture_Balance_Index"]
        sim["Biofilm_Defense_Ratio"] = float(round(np.clip(100.0 - sim["Bacterial_Biofilm_Density"], 5.0, 95.0), 1))
        sim["Enzymatic_Regulation"] = float(round(np.clip(100.0 - sim["MMP_Activity"], 5.0, 95.0), 1))

        # Net Acceleration Calculation
        base_prolif = self.state["Cellular_Proliferation_Factor"]
        sim_prolif = sim["Cellular_Proliferation_Factor"]
        acceleration_pct = round(((sim_prolif - base_prolif) / (base_prolif + 1e-6)) * 100.0, 1)
        acceleration_pct = max(-20.0, min(150.0, acceleration_pct))

        return {
            "baseline_state": self.state,
            "simulated_state": sim,
            "healing_acceleration_pct": acceleration_pct,
            "simulated_interventions": {
                "Target_HbA1c": hba1c_val,
                "Dressing": dressing,
                "Debridement": debridement,
                "HBOT": hbot,
                "NPWT": npwt,
                "Offloading_100": offloading_100
            }
        }

def create_digital_twin(clinical_dict, tissue_breakdown=None):
    return PatientDigitalTwin(clinical_dict, tissue_breakdown)

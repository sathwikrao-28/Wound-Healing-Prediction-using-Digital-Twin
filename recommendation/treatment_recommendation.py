import numpy as np

class TreatmentRecommendationEngine:
    """
    Clinical Decision Support System (Block 8 of System Architecture).
    Generates evidence-based treatment suggestions for Clinicians AND a
    patient-friendly Care Plan.
    """
    def __init__(self):
        pass

    def generate_plan(self, clinical_dict, tissue_breakdown, model_predictions, digital_twin_state=None):
        """
        Generate complete evidence-based clinical treatment plan and patient care guide.
        """
        hba1c = float(clinical_dict.get("HbA1c", 7.5))
        exudate = clinical_dict.get("Exudate_Level", "Moderate")
        wound_type = clinical_dict.get("Wound_Type", "Diabetic Foot Ulcer")
        biofilm = clinical_dict.get("Biofilm_Presence", "No") in ["Yes", True, 1]
        abi = float(clinical_dict.get("ABI_Index", 0.90))
        depth = float(clinical_dict.get("Wound_Depth_mm", 5.0))
        area = float(clinical_dict.get("Baseline_Area_cm2", 12.0))

        g_pct = float(tissue_breakdown.get("Granulation_Pct", 50.0))
        s_pct = float(tissue_breakdown.get("Slough_Pct", 30.0))
        n_pct = float(tissue_breakdown.get("Necrotic_Pct", 10.0))

        inf_risk = float(model_predictions.get("baseline_predictions", {}).get("infection_risk_pct", 30.0))

        # Categorize Infection Risk: HIGH, MEDIUM, LOW
        if inf_risk > 50.0 or biofilm:
            infection_rating = "HIGH"
        elif inf_risk > 25.0:
            infection_rating = "MEDIUM"
        else:
            infection_rating = "LOW"

        # --- 1. CLINICIAN PROTOCOL (TIME Framework) ---
        if n_pct > 15.0 or "Stage 4" in wound_type or "Stage 3" in wound_type:
            debridement_plan = "Sharp Surgical Debridement indicated to remove non-viable necrotic eschar, followed by autolytic debridement."
            debridement_priority = "URGENT / HIGH"
        elif s_pct > 25.0:
            debridement_plan = "Enzymatic Debridement (Collagenase Ointment) applied daily to dissolve fibrinous slough."
            debridement_priority = "MODERATE"
        else:
            debridement_plan = "Light mechanical wound cleansing with sterile saline or hypochlorous acid during dressing changes."
            debridement_priority = "MAINTENANCE"

        if infection_rating == "HIGH":
            infection_protocol = "Topical Antimicrobial Silver / Cadexomer Iodine dressing for 14 days. Obtain deep tissue wound swab culture."
        else:
            infection_protocol = "Routine cleansing with non-cytotoxic wound cleanser. Prophylactic antimicrobial barrier dressing."

        if exudate == "Heavy":
            primary_dressing = "Superabsorbent Polymer / Calcium Alginate Dressing"
            secondary_dressing = "High-MVTR Hydropolysac foam secondary dressing to prevent maceration."
        elif exudate == "Light" or n_pct > 20.0:
            primary_dressing = "Amorphous Hydrogel Matrix with Transparent Film"
            secondary_dressing = "Semi-permeable occlusive film to donate moisture."
        else:
            primary_dressing = "Hydrocellular Polyurethane Foam Dressing with Silicone Contact Layer"
            secondary_dressing = "Conforming gauze wrap or tubular bandage."

        if depth > 8.0 or area > 20.0:
            adjuvant_therapy = "Negative Pressure Wound Therapy (NPWT) at -125 mmHg continuous suction."
        elif abi < 0.6:
            adjuvant_therapy = "Urgent Vascular Surgery Referral for revascularization evaluation prior to sharp debridement."
        else:
            adjuvant_therapy = "Collagen Matrix / Extracellular Matrix (ECM) graft application."

        if "Diabetic Foot" in wound_type:
            offloading = "Total Contact Casting (TCC) or Removable Cast Walker (RCW) with rigid rocker sole."
        elif "Pressure" in wound_type:
            offloading = "Dynamic Air Redistribution Mattress. Q2H patient turning schedule."
        else:
            offloading = "Multi-layer compression wrap (30-40 mmHg) after confirming ABI > 0.8."

        nutrition = [
            f"Target Glycemic Control: HbA1c < 7.0% (Current: {hba1c}%).",
            "High Protein Diet: 1.25 - 1.50 g/kg/day for collagen synthesis.",
            "Daily Micronutrients: Zinc 220 mg, Vitamin C 500 mg twice daily."
        ]

        # --- 2. PATIENT-FRIENDLY CARE PLAN ---
        patient_care_plan = [
            f"Keep your wound dressing dry and clean. Change dressing every {2 if exudate=='Heavy' else 3} days.",
            f"Wear your protective offloading boot/shoe ({offloading.split(' ')[0]}) at ALL times when standing or walking.",
            f"Monitor blood sugar daily to keep HbA1c below 7.0%. High sugar slows tissue healing.",
            "Eat protein-rich meals (chicken, fish, eggs, beans) to help your body rebuild healthy wound tissue.",
            "⚠️ WATCH FOR WARNING SIGNS: Call clinic immediately if you notice spreading red skin, foul odor, sudden heavy fluid leakage, or fever above 100.4°F."
        ]

        # Dressing Change Frequency & Next Clinical Review Date
        if exudate == "Heavy":
            dressing_change = "Every 24 to 48 Hours (Daily to Alternate Days based on saturation)"
        elif exudate == "Light":
            dressing_change = "Every 3 to 4 Days (Twice Weekly)"
        else:
            dressing_change = "Every 48 to 72 Hours (Every 2 to 3 Days)"

        if infection_rating == "HIGH":
            next_review = "3 to 4 Days (Urgent Progress & Bioburden Re-Evaluation)"
        elif infection_rating == "MEDIUM":
            next_review = "7 Days (Weekly Routine Clinic Assessment)"
        else:
            next_review = "7 to 10 Days (Bi-Weekly Progress Assessment)"

        return {
            "infection_rating": infection_rating,
            "primary_dressing": primary_dressing,
            "secondary_dressing": secondary_dressing,
            "dressing_change_frequency": dressing_change,
            "next_clinical_review": next_review,
            "debridement_regimen": debridement_plan,
            "debridement_priority": debridement_priority,
            "infection_protocol": infection_protocol,
            "adjuvant_therapy": adjuvant_therapy,
            "offloading_protocol": offloading,
            "nutritional_recommendations": nutrition,
            "patient_care_plan": patient_care_plan,
            "followup_schedule": next_review,
            "summary_recommendation": (
                f"Recommended regimen combines **{primary_dressing}** for moisture balance, "
                f"**{debridement_priority} debridement**, and **{offloading}**."
            )
        }

def generate_treatment_recommendation(clinical_dict, tissue_breakdown, model_predictions, digital_twin_state=None):
    engine = TreatmentRecommendationEngine()
    return engine.generate_plan(clinical_dict, tissue_breakdown, model_predictions, digital_twin_state)

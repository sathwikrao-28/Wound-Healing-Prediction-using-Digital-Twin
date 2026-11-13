#!/opt/anaconda3/bin/python3
import os
import sys
import json
import base64
import cv2
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from PIL import Image
import io

# Add workspace directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing.image_preprocess import preprocess_image
from preprocessing.clinical_preprocess import preprocess_clinical
from models.wound_cnn import get_wound_cnn_model
from models.fusion_model import fuse_multimodal
from models.lstm_model import predict_wound_trajectory
from digital_twin.digital_twin import create_digital_twin
from explainability.shap_analysis import explain_clinical_shap
from explainability.score_cam import generate_cam_explainability
from recommendation.treatment_recommendation import generate_treatment_recommendation
from evaluation.performance_eval import get_performance_evaluation, PerformanceEvaluator
from database.database import get_database

# ==============================================================================
# DATASET SOURCES & BENCHMARK REFERENCES (Code Comment Documentation):
# ------------------------------------------------------------------------------
# 1. DFUC Grand Challenge (MICCAI/Kaggle): Diabetic Foot Ulcer photograph dataset (dfu2020.grand-challenge.org)
# 2. Medetec Clinical Database: Tissue segmentation masks for Granulation, Slough, and Eschar (medetec.co.uk)
# 3. MIMIC-IV EHR Database (PhysioNet/MIT): Anonymized vitals distributions (HbA1c, ABI, BP, WBC) (physionet.org)
# 4. NPUAP / EPUAP Guidelines: Pressure injury classification standards
# ==============================================================================

app = Flask(__name__, template_folder="templates", static_folder="static")

db = get_database()
evaluator = PerformanceEvaluator()

def cv2_to_base64(img_np_rgb):
    """Helper to convert RGB numpy image to base64 jpeg string."""
    if img_np_rgb is None:
        return ""
    img_bgr = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/docs")
def api_docs():
    """Serve interactive Swagger UI API documentation."""
    swagger_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>WoundCare Clinical API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
    <style>
        html { box-sizing: border-box; overflow-y: scroll; }
        body { margin:0; background: #0f172a; color: #f8fafc; font-family: sans-serif; }
        .swagger-ui .topbar { display: none; }
        .swagger-ui { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .swagger-ui .info .title { color: #38bdf8; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js" charset="UTF-8"> </script>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-standalone-preset.js" charset="UTF-8"> </script>
    <script>
    window.onload = function() {
      window.ui = SwaggerUIBundle({
        url: "/api/openapi.json",
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
        layout: "StandaloneLayout"
      });
    };
    </script>
</body>
</html>"""
    return swagger_html

@app.route("/api/openapi.json")
def openapi_spec():
    """Return OpenAPI 3.0 specification JSON."""
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": "WoundCare Clinical Decision Support REST API",
            "description": "Multimodal Wound Healing Assessment, Digital Twin Simulation, XAI Explainability (SHAP & Score-CAM) & CDSS Treatment Recommendation API",
            "version": "1.0.0"
        },
        "servers": [{"url": "http://localhost:5000", "description": "Local Development Server"}],
        "paths": {
            "/api/patients": {
                "get": {
                    "summary": "Get Patient Directory",
                    "description": "Retrieve list of all 500 patient records from SQLite database.",
                    "responses": {
                        "200": {"description": "Array of patient profiles"}
                    }
                }
            },
            "/api/patient/{patient_id}": {
                "get": {
                    "summary": "Get Patient Profile & History",
                    "description": "Retrieve profile and visit history for a specific patient MRN ID.",
                    "parameters": [{
                        "name": "patient_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string", "example": "PAT-101"}
                    }],
                    "responses": {
                        "200": {"description": "Patient detail profile and historical visit records"},
                        "404": {"description": "Patient not found"}
                    }
                }
            },
            "/api/analyze": {
                "post": {
                    "summary": "Run Multimodal AI Analysis Pipeline",
                    "description": "Executes image segmentation, clinical feature encoding, multimodal fusion, LSTM trajectory forecasting, digital twin baseline, SHAP, Score-CAM, and CDSS treatment plan.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "patient_name": {"type": "string", "example": "Robert Johnson"},
                                        "age": {"type": "number", "example": 62},
                                        "gender": {"type": "string", "example": "Male"},
                                        "diabetes_status": {"type": "string", "example": "Type 2 Diabetes"},
                                        "wound_type": {"type": "string", "example": "Diabetic Foot Ulcer"},
                                        "baseline_area_cm2": {"type": "number", "example": 14.5},
                                        "wound_depth_mm": {"type": "number", "example": 5.2},
                                        "hba1c": {"type": "number", "example": 8.8},
                                        "systolic_bp": {"type": "number", "example": 135},
                                        "diastolic_bp": {"type": "number", "example": 85},
                                        "abi_index": {"type": "number", "example": 0.85},
                                        "exudate_level": {"type": "string", "example": "Moderate"},
                                        "biofilm_presence": {"type": "string", "example": "Yes"},
                                        "image_filename": {"type": "string", "example": "wound_001.jpg"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Complete multimodal assessment JSON output"}
                    }
                }
            },
            "/api/simulate": {
                "post": {
                    "summary": "Run Digital Twin Counterfactual Simulation",
                    "description": "Simulates physiological changes and days saved based on target HbA1c, dressing choice, debridement, HBOT, NPWT, and offloading.",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "target_hba1c": {"type": "number", "example": 6.8},
                                        "dressing": {"type": "string", "example": "Silver Antimicrobial Alginate"},
                                        "debridement": {"type": "boolean", "example": True},
                                        "hbot": {"type": "boolean", "example": False},
                                        "npwt": {"type": "boolean", "example": False},
                                        "offloading": {"type": "boolean", "example": True}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {"description": "Simulation trajectory and risk adjustments"}
                    }
                }
            },
            "/api/export-report": {
                "post": {
                    "summary": "Generate Clinical Report",
                    "description": "Formats assessment parameters into a downloadable clinical text report.",
                    "responses": {
                        "200": {"description": "Formatted report text"}
                    }
                }
            }
        }
    }
    return jsonify(spec)

@app.route("/data/wound_images/<path:filename>")
def serve_wound_image(filename):
    img_dir = os.path.join(BASE_DIR, "data", "wound_images")
    return send_from_directory(img_dir, filename)

@app.route("/api/patients", methods=["GET"])
def get_patients():
    """Return all 500 patients from SQLite."""
    patients_df = db.get_all_patients()
    if patients_df.empty:
        return jsonify([])
    return jsonify(patients_df.to_dict(orient="records"))

@app.route("/api/patient/<patient_id>", methods=["GET"])
def get_patient_detail(patient_id):
    """Return specific patient profile and historical visits."""
    patients_df = db.get_all_patients()
    matched = patients_df[patients_df["patient_id"] == patient_id]
    if matched.empty:
        return jsonify({"error": "Patient not found"}), 404
    patient_data = matched.iloc[0].to_dict()
    history_df = db.get_patient_history(patient_id)
    history_records = history_df.to_dict(orient="records") if not history_df.empty else []
    return jsonify({
        "profile": patient_data,
        "history": history_records
    })

@app.route("/api/analyze", methods=["POST"])
def analyze_wound():
    """
    Execute full multimodal pipeline:
    Image Preprocessing -> Clinical Preprocessing -> CNN Features -> Fusion ->
    LSTM Forecast -> Digital Twin -> SHAP -> Score-CAM/Grad-CAM -> CDSS Recommendations -> Eval.
    """
    data = request.json or {}
    
    # 1. Image loading
    image_filename = data.get("image_filename", "wound_001.jpg")
    custom_b64 = data.get("image_base64", None)
    
    if custom_b64 and len(custom_b64) > 100:
        try:
            if "," in custom_b64:
                custom_b64 = custom_b64.split(",")[1]
            img_bytes = base64.b64decode(custom_b64)
            raw_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            raw_image = Image.new("RGB", (300, 300), color=(200, 60, 70))
    else:
        sample_img_path = os.path.join(BASE_DIR, "data", "wound_images", image_filename)
        if os.path.exists(sample_img_path):
            raw_image = Image.open(sample_img_path).convert("RGB")
        else:
            raw_image = Image.new("RGB", (300, 300), color=(200, 60, 70))

    # 2. Clinical Vitals
    clinical_dict = {
        "Patient_Name": data.get("patient_name", "Robert Johnson"),
        "Age": float(data.get("age", 62)),
        "Gender": data.get("gender", "Male"),
        "Diabetes_Status": data.get("diabetes_status", "Type 2 Diabetes"),
        "Wound_Type": data.get("wound_type", "Diabetic Foot Ulcer"),
        "Baseline_Area_cm2": float(data.get("baseline_area_cm2", 14.5)),
        "Wound_Depth_mm": float(data.get("wound_depth_mm", 5.2)),
        "HbA1c": float(data.get("hba1c", 8.8)),
        "Systolic_BP": float(data.get("systolic_bp", 135)),
        "Diastolic_BP": float(data.get("diastolic_bp", 85)),
        "ABI_Index": float(data.get("abi_index", 0.85)),
        "Exudate_Level": data.get("exudate_level", "Moderate"),
        "Biofilm_Presence": data.get("biofilm_presence", "Yes"),
        "Smoking_Status": data.get("smoking_status", "Former"),
        "Infection_Markers_WBC": 11.5 if data.get("biofilm_presence") == "Yes" else 7.2,
        "Pain_Score": 6,
        "Mobility_Score": 3
    }

    # 3. Pipeline execution
    img_prep_res = preprocess_image(raw_image)
    display_img = img_prep_res["display_image"]
    tissue_breakdown = img_prep_res["tissue_breakdown"]
    segmentation_mask = img_prep_res["segmentation_mask"]

    clin_prep_res = preprocess_clinical(clinical_dict)
    clinical_vector = clin_prep_res["feature_vector"]
    derived_risks = clin_prep_res["derived_risks"]

    cnn_model = get_wound_cnn_model()
    cnn_features = cnn_model.extract_features(img_prep_res["processed_tensor"], tissue_breakdown)
    cnn_embedding = cnn_features["embedding"]

    fusion_res = fuse_multimodal(cnn_embedding, clinical_vector, clinical_dict)
    fused_vec = fusion_res["fused_vector"]

    lstm_forecast = predict_wound_trajectory(fused_vec, clinical_dict, tissue_breakdown)

    twin = create_digital_twin(clinical_dict, tissue_breakdown)
    baseline_twin_state = twin.get_current_state()

    shap_res = explain_clinical_shap(clinical_dict, tissue_breakdown)
    cam_res = generate_cam_explainability(display_img, tissue_breakdown)

    treatment_plan = generate_treatment_recommendation(clinical_dict, tissue_breakdown, fusion_res, baseline_twin_state)
    eval_metrics = get_performance_evaluation()

    # Encode images to base64 for frontend consumption
    display_b64 = cv2_to_base64(display_img)
    seg_b64 = cv2_to_base64(segmentation_mask)
    score_cam_b64 = cv2_to_base64(cam_res["score_cam"]["overlay_rgb"])
    grad_cam_b64 = cv2_to_base64(cam_res["grad_cam"]["overlay_rgb"])

    return jsonify({
        "success": True,
        "images": {
            "display_image": display_b64,
            "segmentation_mask": seg_b64,
            "score_cam": score_cam_b64,
            "grad_cam": grad_cam_b64,
            "score_cam_desc": cam_res["score_cam"]["description"],
            "grad_cam_desc": cam_res["grad_cam"]["description"]
        },
        "tissue_breakdown": tissue_breakdown,
        "derived_risks": derived_risks,
        "lstm_forecast": {
            "weeks": lstm_forecast["weeks"],
            "projected_healing_pct": lstm_forecast.get("healing_pct", []),
            "infection_risk_pct": lstm_forecast.get("infection_risk_pct", []),
            "ci_lower": lstm_forecast.get("healing_lower", []),
            "ci_upper": lstm_forecast.get("healing_upper", []),
            "estimated_days_to_closure": lstm_forecast.get("estimated_days_to_closure", 35),
            "healing_time_category": lstm_forecast.get("healing_time_category", "Standard Healing Trajectory")
        },
        "digital_twin_baseline": baseline_twin_state,
        "shap": {
            "features": shap_res["impact_df"]["Feature"].tolist(),
            "values": shap_res["impact_df"]["SHAP_Value"].tolist(),
            "narrative": shap_res["narrative_summary"]
        },
        "treatment_plan": treatment_plan,
        "evaluation": {
            "classification_metrics": eval_metrics["classification_metrics"],
            "regression_metrics": eval_metrics["regression_metrics"],
            "confusion_matrix": eval_metrics["confusion_matrix"].tolist(),
            "classification_report": eval_metrics["classification_report"]
        }
    })

@app.route("/api/simulate", methods=["POST"])
def simulate_twin():
    """Run interactive Digital Twin counterfactual intervention."""
    data = request.json or {}
    clinical_dict = data.get("clinical_dict", {})
    tissue_breakdown = data.get("tissue_breakdown", {})

    target_hba1c = float(data.get("target_hba1c", 6.8))
    dressing = data.get("dressing", "Silver Antimicrobial Alginate")
    debridement = bool(data.get("debridement", True))
    hbot = bool(data.get("hbot", False))
    npwt = bool(data.get("npwt", False))
    offloading = bool(data.get("offloading", True))

    twin = create_digital_twin(clinical_dict, tissue_breakdown)
    sim_res = twin.simulate_intervention(
        target_hba1c=target_hba1c,
        dressing=dressing,
        debridement=debridement,
        hbot=hbot,
        npwt=npwt,
        offloading_100=offloading
    )
    return jsonify(sim_res)

@app.route("/api/export-report", methods=["POST"])
def export_report():
    """Return text content of clinical assessment report."""
    data = request.json or {}
    p_name = data.get("patient_name", "Robert Johnson")
    pid = data.get("patient_id", "PAT-101")
    age = data.get("age", 62)
    gender = data.get("gender", "Male")
    bgroup = data.get("blood_group", "A+")
    phone = data.get("contact_number", "+1 (555) 234-5678")
    wtype = data.get("wound_type", "Diabetic Foot Ulcer")
    wloc = data.get("wound_location", "Plantar Surface - Left Heel")
    comorb = data.get("comorbidities", "Type 2 Diabetes, Hypertension")
    allerg = data.get("allergies", "Latex")
    clin = data.get("clinical_dict", {})
    t_bd = data.get("tissue_breakdown", {"Granulation_Pct": 65, "Slough_Pct": 25, "Necrotic_Pct": 5, "Epithelial_Pct": 5})
    fc = data.get("lstm_forecast", {"estimated_days_to_closure": 35, "healing_time_category": "Standard Healing"})
    tx = data.get("treatment_plan", {})
    rk = data.get("derived_risks", {"Overall_Risk_Score": 40})

    lines = [
        "================================================================================",
        "                    WOUNDCARE CLINICAL DECISION SUPPORT REPORT                  ",
        "================================================================================",
        "Clinical Center    : Vascular & Advanced Wound Care Department",
        "Attending Physician: Dr. Sathwik Rao, MD",
        f"Report Patient     : {p_name} (MRN: {pid})",
        "================================================================================",
        "",
        "1. PATIENT HEALTH PROFILE & DEMOGRAPHICS",
        "--------------------------------------------------------------------------------",
        f"• Patient Full Name    : {p_name}",
        f"• Medical Record ID    : {pid}",
        f"• Age & Biological Sex : {age} Years | {gender}",
        f"• Blood Group          : {bgroup}",
        f"• Emergency Contact    : {phone}",
        f"• Primary Comorbidities: {comorb}",
        f"• Known Allergies      : {allerg}",
        f"• Tobacco Smoking      : {clin.get('Smoking_Status', 'Never')}",
        "",
        "2. CLINICAL VITALS & MEASUREMENTS",
        "--------------------------------------------------------------------------------",
        f"• Wound Etiology       : {wtype}",
        f"• Anatomical Location  : {wloc}",
        f"• Wound Surface Area   : {clin.get('Baseline_Area_cm2', 14.5)} cm²",
        f"• Wound Depth          : {clin.get('Wound_Depth_mm', 5.2)} mm",
        f"• Glycated HbA1c       : {clin.get('HbA1c', 8.8)}% ({clin.get('Diabetes_Status', 'Type 2 Diabetes')})",
        f"• Blood Pressure       : {clin.get('Systolic_BP', 135)}/{clin.get('Diastolic_BP', 85)} mmHg",
        f"• Ankle-Brachial Index : {clin.get('ABI_Index', 0.85)} (Vascular Perfusion)",
        f"• Exudate Severity     : {clin.get('Exudate_Level', 'Moderate')}",
        f"• Biofilm Presence     : {clin.get('Biofilm_Presence', 'Yes')}",
        "",
        "3. COMPUTER VISION TISSUE ANALYSIS",
        "--------------------------------------------------------------------------------",
        f"• Healthy Granulation Tissue (Red) : {t_bd.get('Granulation_Pct')}%",
        f"• Fibrinous Slough Bed (Yellow)    : {t_bd.get('Slough_Pct')}%",
        f"• Necrotic Eschar Tissue (Black)   : {t_bd.get('Necrotic_Pct')}%",
        f"• New Skin Epithelial Margin (Pink): {t_bd.get('Epithelial_Pct')}%",
        "",
        "4. AI PREDICTIVE FORECAST & CLINICAL RISKS",
        "--------------------------------------------------------------------------------",
        f"• Projected Recovery Time  : {fc.get('estimated_days_to_closure')} Days",
        f"• Healing Trajectory       : {fc.get('healing_time_category')}",
        f"• Infection Risk Rating    : {tx.get('infection_rating')} RISK",
        f"• Overall Clinical Risk    : {rk.get('Overall_Risk_Score')}%",
        "",
        "5. RECOMMENDED CLINICAL TREATMENT PLAN",
        "--------------------------------------------------------------------------------",
        f"• Primary Dressing          : {tx.get('primary_dressing')}",
        f"• Secondary Dressing        : {tx.get('secondary_dressing')}",
        f"• Dressing Change Frequency : {tx.get('dressing_change_frequency', 'Every 48 to 72 Hours')}",
        f"• Next Clinical Review Date : {tx.get('next_clinical_review', '7 Days (Weekly Review)')}",
        f"• Debridement Regimen       : {tx.get('debridement_regimen')} (Priority: {tx.get('debridement_priority')})",
        f"• Infection Control         : {tx.get('infection_protocol')}",
        f"• Pressure Offloading       : {tx.get('offloading_protocol')}",
        f"• Strategy Summary          : {tx.get('summary_recommendation')}",
        "",
        "6. PATIENT DAILY HOME CARE INSTRUCTIONS",
        "--------------------------------------------------------------------------------",
    ]
    for idx, step in enumerate(tx.get("patient_care_plan", []), 1):
        lines.append(f"{idx}. {step}")

    lines.extend([
        "",
        "================================================================================",
        "PHYSICIAN SIGNATURE: _____________________________ DATE: ___________________",
        "================================================================================"
    ])
    report_text = "\n".join(lines)
    return jsonify({"report": report_text, "filename": f"WoundCare_Assessment_{pid}_{p_name.replace(' ', '_')}.txt"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"🩺 WoundCare Web Application Server Running on:")
    print(f"👉 Local URL: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

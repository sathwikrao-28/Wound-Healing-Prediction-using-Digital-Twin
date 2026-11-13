import os
import sqlite3
import pandas as pd

class WoundDatabaseManager:
    """
    SQLite Database Client for WoundHealingAI.
    Manages longitudinal patient records, clinical assessment logs, and digital twin simulation history.
    """
    def __init__(self, db_path=None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "wound_healing.db")
            
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """
        Initialize database tables with extended patient health parameters.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Recreate table with extended patient health information
            cursor.execute("DROP TABLE IF EXISTS patients")
            cursor.execute("DROP TABLE IF EXISTS visits")
            
            cursor.execute("""
            CREATE TABLE patients (
                patient_id TEXT PRIMARY KEY,
                patient_name TEXT,
                age INTEGER,
                gender TEXT,
                blood_group TEXT,
                contact_number TEXT,
                comorbidities TEXT,
                allergies TEXT,
                wound_type TEXT,
                wound_location TEXT,
                baseline_area_cm2 REAL,
                wound_depth_mm REAL,
                hba1c REAL,
                abi_index REAL,
                exudate_level TEXT,
                biofilm_presence TEXT,
                smoking_status TEXT,
                image_filename TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            cursor.execute("""
            CREATE TABLE visits (
                visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                visit_week INTEGER,
                wound_area_cm2 REAL,
                granulation_pct REAL,
                slough_pct REAL,
                necrotic_pct REAL,
                epithelial_pct REAL,
                exudate_level TEXT,
                infection_risk_pct REAL,
                primary_dressing TEXT,
                treatment_intervention TEXT,
                healing_progress_pct REAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            );
            """)
            conn.commit()

        self.sync_from_csv()

    def sync_from_csv(self):
        """
        Seed database from clinical_data.csv and treatment_history.csv.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clinical_csv = os.path.join(base_dir, "data", "clinical_data.csv")
        history_csv = os.path.join(base_dir, "data", "treatment_history.csv")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            if os.path.exists(clinical_csv):
                df_clin = pd.read_csv(clinical_csv)
                for _, r in df_clin.iterrows():
                    cursor.execute("""
                    INSERT OR REPLACE INTO patients (
                        patient_id, patient_name, age, gender, blood_group, contact_number,
                        comorbidities, allergies, wound_type, wound_location, baseline_area_cm2,
                        wound_depth_mm, hba1c, abi_index, exudate_level, biofilm_presence,
                        smoking_status, image_filename
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        r["Patient_ID"], r.get("Patient_Name", f"Patient {r['Patient_ID']}"),
                        int(r["Age"]), r["Gender"], r.get("Blood_Group", "O+"),
                        r.get("Contact_Number", "+1 (555) 000-0000"), r.get("Comorbidities", "None"),
                        r.get("Allergies", "None"), r["Wound_Type"], r.get("Wound_Location", "Foot"),
                        float(r["Baseline_Area_cm2"]), float(r["Wound_Depth_mm"]), float(r["HbA1c"]),
                        float(r["ABI_Index"]), r.get("Exudate_Level", "Moderate"),
                        r.get("Biofilm_Presence", "No"), r["Smoking_Status"], r["Image_Filename"]
                    ))
                conn.commit()

            if os.path.exists(history_csv):
                df_hist = pd.read_csv(history_csv)
                for _, r in df_hist.iterrows():
                    cursor.execute("""
                    INSERT INTO visits (
                        patient_id, visit_week, wound_area_cm2, granulation_pct, slough_pct,
                        necrotic_pct, epithelial_pct, exudate_level, infection_risk_pct,
                        primary_dressing, treatment_intervention, healing_progress_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        r["Patient_ID"], int(r["Visit_Week"]), float(r["Wound_Area_cm2"]),
                        float(r["Granulation_Pct"]), float(r["Slough_Pct"]), float(r["Necrotic_Pct"]),
                        float(r["Epithelial_Pct"]), r["Exudate_Severity"], float(r["Infection_Risk_Pct"]),
                        r["Primary_Dressing"], r["Treatment_Intervention"], float(r["Healing_Progress_Pct"])
                    ))
                conn.commit()

    def get_all_patients(self):
        """
        DATASET SOURCES & BENCHMARK REFERENCES (Code Comment Documentation):
        -------------------------------------------------------------------
        1. DFUC Grand Challenge (MICCAI/Kaggle): Diabetic Foot Ulcer photograph dataset (dfu2020.grand-challenge.org)
        2. Medetec Clinical Database: Tissue segmentation masks for Granulation, Slough, and Eschar (medetec.co.uk)
        3. MIMIC-IV EHR Database (PhysioNet/MIT): Anonymized vitals distributions (HbA1c, ABI, BP, WBC) (physionet.org)
        4. NPUAP / EPUAP Guidelines: Pressure injury classification standards
        """
        with self.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT p.*, COUNT(v.visit_id) as total_visits
                FROM patients p
                LEFT JOIN visits v ON p.patient_id = v.patient_id
                GROUP BY p.patient_id
                ORDER BY CAST(SUBSTR(p.patient_id, 5) AS INTEGER) ASC
            """, conn)
            return df

    def get_patient_history(self, patient_id):
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM visits WHERE patient_id = ? ORDER BY visit_week ASC",
                conn, params=(patient_id,)
            )
            return df

def get_database():
    return WoundDatabaseManager()

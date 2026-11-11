SEED = 2026
TARGET = "readmitted"

# -- Columns dropped at the start. --
COLUMNS_TO_DROP = [
    "weight",  # ~97% of missing values
]

# -- Leakage --
EXPIRED_IDS = [11, 19, 20, 21]
NEWBORN_DISCHARGED_IDS = [10]
NEWBORN_ADMISSION_TYPE_IDS = [4]
BIRTH_ADMISSION_SOURCE_IDS = [11, 12, 13, 14, 23, 24]

# -- Drugs --
# Drugs that are prescribed in <1% of encounters
RARE_DRUGS = [
    "acarbose", "acetohexamide", "chlorpropamide", "citoglipton", "examide",
    "glimepiride-pioglitazone", "glipizide-metformin", "glyburide-metformin",
    "metformin-pioglitazone", "metformin-rosiglitazone", "miglitol", "nateglinide",
    "tolazamide", "tolbutamide", "troglitazone"
]   

DRUG_COLS = [
    "glimepiride", "glipizide", "glyburide", "insulin", 
    "metformin", "pioglitazone", "repaglinide", "rosiglitazone",
]

# -- Columns classification --
ID_COLS = ["encounter_id", "patient_nbr"]
DEMOGRAPHIC_COLS = ["race", "gender", "age"]

LAB_COLS = ["A1Cresult", "max_glu_serum"]
ADMIN_COLS = ["payer_code", "medical_specialty"]
MED_SUMMARY_COLS = ["change", "diabetesMed"]

DIAG_COLS = ["diag_1", "diag_2", "diag_3"]

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
CODED_ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


# -- train constants --
RANDOM_STATE = 42
N_SPLITS = 5
MAX_GAP = 0.03

# Logistic Regression
C = 0.5

# XGBoost
XGB_CONFIG = {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 500, "min_child_weight": 20, "reg_lambda": 10, "gamma": 1}

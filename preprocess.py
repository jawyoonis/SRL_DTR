import os
import numpy as np
import pandas as pd
from google.colab import auth
from google.cloud import bigquery
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# ── Auth & BigQuery client ────────────────────────────────────
auth.authenticate_user()
PROJECT_ID = "cse6250-final-project-488221"
client     = bigquery.Client(project=PROJECT_ID)
DS         = "physionet-data.mimiciii_clinical"

# ── Output path ───────────────────────────────────────────────
OUT_DIR = "/content/drive/MyDrive/CSE6250_final_project/processed"
os.makedirs(OUT_DIR, exist_ok=True)

LAB_COLS  = ["dbp", "fio2", "GCS", "blood_glucose", "sbp",
             "hr", "PH", "rr", "bos", "temp", "urine_output"]
AC_COLS   = ["l" + str(i) for i in range(180)]
DI_COLS   = [str(i) for i in range(39)]
DEMO_COLS = ["sofa", "GENDER", "RELIGION", "MARITAL_STATUS",
             "age", "weight", "height", "language", "ethnicity"]

print("=" * 55)
print("  MIMIC-III Preprocessing via BigQuery")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# 1. Admissions + patients
# ─────────────────────────────────────────────────────────────
print("\n[1/7] Loading admissions and patients...")

admissions = client.query(f"""
    SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime,
           a.hospital_expire_flag, a.marital_status,
           a.religion, a.language, a.ethnicity,
           p.gender,
           DATE_DIFF(DATE(a.admittime), DATE(p.dob), YEAR) as age
    FROM `{DS}.admissions` a
    JOIN `{DS}.patients`   p USING (subject_id)
    WHERE DATE_DIFF(DATE(a.admittime), DATE(p.dob), YEAR) >= 18
""").to_dataframe()

admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
admissions = admissions.sort_values("admittime")
admissions = admissions.groupby("subject_id").first().reset_index()
admissions["weight"] = 70.0
admissions["height"] = 170.0
admissions["sofa"]   = 0

print(f"  ✓ Cohort: {len(admissions):,} admissions")

hadm_ids = admissions["hadm_id"].tolist()
ids_str  = ",".join(str(x) for x in hadm_ids)

# ─────────────────────────────────────────────────────────────
# 2. Vitals from CHARTEVENTS
# ─────────────────────────────────────────────────────────────
print("\n[2/7] Loading vitals from CHARTEVENTS...")

VITAL_ITEMS = {
    "dbp"          : [8368, 8440, 8441, 8555, 220180, 220051],
    "fio2"         : [3420, 190, 223835, 3422],
    "GCS"          : [198, 226755, 227013],
    "blood_glucose": [225664, 220621, 226537, 807, 811],
    "sbp"          : [51, 442, 455, 6701, 220179, 220050],
    "hr"           : [211, 220045],
    "PH"           : [780, 1126, 223830, 220274],
    "rr"           : [615, 618, 220210, 224690],
    "bos"          : [646, 220277],
    "temp"         : [223761, 678, 223762, 676],
}

all_item_ids = [i for ids in VITAL_ITEMS.values() for i in ids]
items_str    = ",".join(str(x) for x in all_item_ids)

chart = client.query(f"""
    SELECT hadm_id, itemid, charttime, valuenum
    FROM `{DS}.chartevents`
    WHERE hadm_id IN ({ids_str})
      AND itemid  IN ({items_str})
      AND valuenum IS NOT NULL
      AND error IS DISTINCT FROM 1
""").to_dataframe()

chart["charttime"] = pd.to_datetime(chart["charttime"])
item_to_feat       = {i: f for f, ids in VITAL_ITEMS.items() for i in ids}
chart["feature"]   = chart["itemid"].map(item_to_feat)
print(f"  ✓ Chart rows: {len(chart):,}")

# ─────────────────────────────────────────────────────────────
# 3. Urine output
# ─────────────────────────────────────────────────────────────
print("\n[3/7] Loading urine output...")

URINE_ITEMS = [40055,43175,40069,40094,40715,40473,40085,40057,
               40056,40405,40428,40086,40096,40651,226559,226560,
               226561,226584,226563,226564,226565,226567,226557,226558]
urine_str = ",".join(str(x) for x in URINE_ITEMS)

outputs = client.query(f"""
    SELECT hadm_id, charttime, value
    FROM `{DS}.outputevents`
    WHERE hadm_id IN ({ids_str})
      AND itemid  IN ({urine_str})
      AND value   IS NOT NULL
""").to_dataframe()

outputs["charttime"] = pd.to_datetime(outputs["charttime"])
print(f"  ✓ Output rows: {len(outputs):,}")

# ─────────────────────────────────────────────────────────────
# 4. Build 24-hour windows
# ─────────────────────────────────────────────────────────────
print("\n[4/7] Building 24-hour windows...")

records = []
total   = len(admissions)
for idx, row in admissions.iterrows():
    hadm_id  = row["hadm_id"]
    admit    = row["admittime"]
    disch    = row["dischtime"]
    expire   = row["hospital_expire_flag"]
    los_days = max(1, int((disch - admit).total_seconds() / 86400))

    ch  = chart[chart["hadm_id"]     == hadm_id]
    out = outputs[outputs["hadm_id"] == hadm_id]

    for day in range(los_days):
        t0 = admit + pd.Timedelta(days=day)
        t1 = admit + pd.Timedelta(days=day+1)

        ch_day  = ch[(ch["charttime"]   >= t0) & (ch["charttime"]   < t1)]
        out_day = out[(out["charttime"] >= t0) & (out["charttime"]   < t1)]

        rec = {"hadm_id": hadm_id, "day": day}
        for feat in LAB_COLS[:-1]:
            vals      = ch_day[ch_day["feature"] == feat]["valuenum"]
            rec[feat] = vals.mean() if len(vals) > 0 else np.nan

        rec["urine_output"] = out_day["value"].sum() \
                              if len(out_day) > 0 else np.nan
        rec["flag"] = -15 if (day == los_days-1 and expire == 1) \
                      else (15 if day == los_days-1 else 0)
        records.append(rec)

    if idx % 1000 == 0:
        print(f"  ... {idx:,} / {total:,} admissions processed", flush=True)

vitals_df = pd.DataFrame(records)
print(f"  ✓ Total rows: {len(vitals_df):,}")

# ─────────────────────────────────────────────────────────────
# 5. Imputation + scaling
# ─────────────────────────────────────────────────────────────
print("\n[5/7] Imputing and scaling...")

missing   = vitals_df[LAB_COLS].isnull().sum(axis=1)
vitals_df = vitals_df[missing <= 10].copy()

imputer             = KNNImputer(n_neighbors=5)
vitals_df[LAB_COLS] = imputer.fit_transform(vitals_df[LAB_COLS])

scaler              = StandardScaler()
vitals_df[LAB_COLS] = scaler.fit_transform(vitals_df[LAB_COLS])
print(f"  ✓ Rows after imputation: {len(vitals_df):,}")

# ─────────────────────────────────────────────────────────────
# 6. Medications → top 180 → l0..l179
# ─────────────────────────────────────────────────────────────
print("\n[6/7] Processing medications...")

presc = client.query(f"""
    SELECT hadm_id, drug
    FROM `{DS}.prescriptions`
    WHERE hadm_id IN ({ids_str})
      AND drug    IS NOT NULL
""").to_dataframe()

presc["drug_clean"] = presc["drug"].str.strip().str.upper()
presc["drug_clean"] = presc["drug_clean"].str.replace(
    r"\s+\d+.*$", "", regex=True)

top_meds = presc["drug_clean"].value_counts().head(180).index.tolist()

med_records = []
for hadm_id, grp in presc.groupby("hadm_id"):
    drugs = set(grp["drug_clean"].values)
    row   = {"hadm_id": hadm_id}
    for i, med in enumerate(top_meds):
        row[f"l{i}"] = 1 if med in drugs else 0
    med_records.append(row)

med_df    = pd.DataFrame(med_records)
vitals_df = vitals_df.merge(med_df, on="hadm_id", how="left")
for col in AC_COLS:
    if col not in vitals_df.columns:
        vitals_df[col] = 0
vitals_df[AC_COLS] = vitals_df[AC_COLS].fillna(0).astype(int)
print(f"  ✓ Medication columns: {len(AC_COLS)}")

# ─────────────────────────────────────────────────────────────
# 7. Diagnoses + demographics
# ─────────────────────────────────────────────────────────────
print("\n[7/7] Processing diagnoses and demographics...")

diag = client.query(f"""
    SELECT hadm_id, icd9_code
    FROM `{DS}.diagnoses_icd`
    WHERE hadm_id IN ({ids_str})
      AND icd9_code IS NOT NULL
""").to_dataframe()

top_icd = diag["icd9_code"].value_counts().head(2000).index.tolist()

di_records = []
for hadm_id, grp in diag.groupby("hadm_id"):
    codes = set(grp["icd9_code"].values)
    row   = {"hadm_id": hadm_id}
    for i in range(39):
        icd         = top_icd[i] if i < len(top_icd) else None
        row[str(i)] = 1 if (icd and icd in codes) else 0
    di_records.append(row)

di_df = pd.DataFrame(di_records)

# Demographics encoding
gender_map  = {"M": 1, "F": 0}
marital_map = {"MARRIED": 0, "SINGLE": 1, "WIDOWED": 2,
               "DIVORCED": 3, "SEPARATED": 4,
               "LIFE PARTNER": 5, "UNKNOWN (DEFAULT)": 6}
religion_map = {"CATHOLIC": 0, "NOT SPECIFIED": 1,
                "PROTESTANT QUAKER": 2, "UNOBTAINABLE": 3,
                "JEWISH": 4, "BUDDHIST": 5, "MUSLIM": 6,
                "ORTHODOX": 7, "OTHER": 8}
language_map = {"ENGL": 0}
ethnic_map   = {"WHITE": 0, "BLACK/AFRICAN AMERICAN": 1,
                "HISPANIC OR LATINO": 2, "ASIAN": 3,
                "OTHER": 4, "UNKNOWN/NOT SPECIFIED": 5}

sta = admissions[["hadm_id","age","gender","marital_status",
                  "religion","language","ethnicity",
                  "weight","height","sofa"]].copy()

sta["GENDER"]         = sta["gender"].map(gender_map).fillna(0).astype(int)
sta["MARITAL_STATUS"] = sta["marital_status"].map(marital_map).fillna(6).astype(int)
sta["RELIGION"]       = sta["religion"].map(religion_map).fillna(8).astype(int)
sta["language"]       = sta["language"].map(language_map).fillna(1).astype(int)
sta["ethnicity"]      = sta["ethnicity"].map(ethnic_map).fillna(5).astype(int)

sta_final = sta[["hadm_id","sofa","GENDER","RELIGION",
                 "MARITAL_STATUS","age","weight","height",
                 "language","ethnicity"]].copy()

# ─────────────────────────────────────────────────────────────
# Train / Val split 80/10
# ─────────────────────────────────────────────────────────────
print("\nSplitting 80/10...")

all_ids   = vitals_df["hadm_id"].unique()
np.random.seed(42)
np.random.shuffle(all_ids)
n         = len(all_ids)
train_ids = all_ids[:int(0.8*n)]
val_ids   = all_ids[int(0.8*n):int(0.9*n)]

train_df  = vitals_df[vitals_df["hadm_id"].isin(train_ids)]
val_df    = vitals_df[vitals_df["hadm_id"].isin(val_ids)]
train_di  = di_df[di_df["hadm_id"].isin(train_ids)]
val_di    = di_df[di_df["hadm_id"].isin(val_ids)]
train_sta = sta_final[sta_final["hadm_id"].isin(train_ids)]
val_sta   = sta_final[sta_final["hadm_id"].isin(val_ids)]

# ─────────────────────────────────────────────────────────────
# Save 6 CSV files — exact names from config_srl.py
# ─────────────────────────────────────────────────────────────
print("\nSaving 6 CSV files to Drive...")

train_df.to_csv( f"{OUT_DIR}/train_all_12_31_scale.csv", index=False)
val_df.to_csv(   f"{OUT_DIR}/val_all_12_31_scale.csv",   index=False)
train_di.to_csv( f"{OUT_DIR}/train_di_base.csv",          index=False)
val_di.to_csv(   f"{OUT_DIR}/val_di_base.csv",            index=False)
train_sta.to_csv(f"{OUT_DIR}/train_stastic_12_23.csv",    index=False)
val_sta.to_csv(  f"{OUT_DIR}/val_stastic_12_23.csv",      index=False)

print("\n" + "="*55)
print("  ✓ PREPROCESSING COMPLETE")
print(f"  train_all_12_31_scale.csv  → {len(train_df):,} rows")
print(f"  val_all_12_31_scale.csv    → {len(val_df):,} rows")
print(f"  train_di_base.csv          → {len(train_di):,} rows")
print(f"  val_di_base.csv            → {len(val_di):,} rows")
print(f"  train_stastic_12_23.csv    → {len(train_sta):,} rows")
print(f"  val_stastic_12_23.csv      → {len(val_sta):,} rows")
print(f"\n  Saved to: {OUT_DIR}")
print("="*55)

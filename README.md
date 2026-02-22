# SRL-RNN: Dynamic Treatment Recommendation

Modernized implementation of the KDD 2018 paper:
> "Supervised Reinforcement Learning with Recurrent Neural Network for Dynamic Treatment Recommendation"
> Lu Wang, Wei Zhang, Xiaofeng He, Hongyuan Zha
> DOI: https://doi.org/10.1145/3219819.3219961

---

## Prerequisites

Before running this project you must have:

1. PhysioNet account — https://physionet.org/register/
2. CITI training completed — https://physionet.org/about/citi-course/
3. PhysioNet credentialing approved (2-5 business days)
4. MIMIC-III Data Use Agreement signed — https://physionet.org/content/mimiciii/1.4/
5. Google account linked to PhysioNet — https://physionet.org/settings/cloud/
6. GCP project with BigQuery enabled
7. BigQuery access requested on MIMIC-III page

---

## Step 1 — Open Google Colab and install dependencies
```python
!pip install tensorflow==2.19.0 tf-keras==2.19.0 scikit-learn \
            google-cloud-bigquery pyarrow db-dtypes requests --quiet

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

---

## Step 2 — Authenticate and mount Drive
```python
from google.colab import auth, drive
auth.authenticate_user()
drive.mount("/content/drive")

import os
BASE    = "/content/drive/MyDrive/CSE6250_final_project"
OUT_DIR = f"{BASE}/processed"
CKPT    = f"{BASE}/checkpoints"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT,    exist_ok=True)
print("✓ Ready")
```

---

## Step 3 — Test BigQuery connection
```python
from google.cloud import bigquery

PROJECT_ID = "cse6250-final-project-488221"
client     = bigquery.Client(project=PROJECT_ID)

df_test = client.query("""
    SELECT COUNT(*) as n
    FROM `physionet-data.mimiciii_clinical.admissions`
""").to_dataframe()

print(f"✓ Connected — {df_test['n'][0]:,} admissions in MIMIC-III")
```

---

## Step 4 — Run preprocessing pipeline

Run the full preprocessing notebook to generate the 6 CSV files.
This only needs to be done once — files are saved to Google Drive.

The pipeline will:
- Pull vitals, labs, medications, diagnoses, demographics from MIMIC-III
- Filter patients >= 18 years old
- Aggregate into 24-hour time windows
- Apply KNN imputation (drop admissions with > 10 missing values)
- Map medications to ATC level-3 codes (top 180 categories)
- Build reward signal (+15 survival, -15 death, 0 otherwise)
- Split 80/10/10 train/val/test
- Save 6 CSV files to Google Drive

Expected files after preprocessing:
```
CSE6250_final_project/processed/
├── train_all_12_31_scale.csv   ← vitals + medications + reward (train)
├── val_all_12_31_scale.csv     ← vitals + medications + reward (val)
├── train_di_base.csv           ← disease ICD-9 codes (train)
├── val_di_base.csv             ← disease ICD-9 codes (val)
├── train_stastic_12_23.csv     ← demographics (train)
└── val_stastic_12_23.csv       ← demographics (val)
```

---

## Step 5 — Clone the repo
```python
!git clone https://github.com/Joywanglulu/SRL_DTR.git /content/SRL_DTR
%cd /content/SRL_DTR
```

---

## Step 6 — Verify imports
```python
import sys
sys.path.insert(0, "/content/SRL_DTR")

from config_srl    import config
from ActorNetwork  import ActorNetwork
from CriticNetwork import CriticNetwork

print("✓ All imports successful")
print(f"✓ med_size  : {config.med_size}")
print(f"✓ lab_size  : {config.lab_size}")
print(f"✓ tiem_stamp: {config.tiem_stamp}")
```

---

## Step 7 — Start training
```python
from srl_rnn import SRL_RNN

model = SRL_RNN(config)
model.DTR()
```

Training prints every 10 episodes:
```
Episode      0 | Q: 0.0123 | Jaccard: 0.3241
Episode     10 | Q: 0.0456 | Jaccard: 0.3512
  ✓ Best model saved (Jaccard=0.3512)
```

Model checkpoints are saved automatically to:
```
CSE6250_final_project/checkpoints/
├── actor_best.weights.h5     ← best actor weights
├── critic_best.weights.h5    ← best critic weights
├── actor_ep1000.weights.h5   ← periodic checkpoints
└── critic_ep1000.weights.h5
```

---

## Step 8 — Resume training after Colab restart

Training resumes automatically from the last saved checkpoint.
Just re-run Steps 1, 2, 5, 6, and 7.

---

## Hyperparameters (config_srl.py)

| Parameter     | Value   | Description                     |
|---------------|---------|---------------------------------|
| batch_size    | 30      | Training batch size             |
| gamma         | 0.99    | Discount factor                 |
| tau           | 0.001   | Soft target update rate         |
| lra           | 0.001   | Actor learning rate             |
| lrc           | 0.005   | Critic learning rate            |
| epsilon       | 0.5     | Balance RL vs supervised loss   |
| tiem_stamp    | 5       | Time steps per window           |
| episode_count | 100000  | Total training episodes         |
| med_size      | 180     | Medication categories (ATC-3)   |
| lab_size      | 12      | Vital sign features             |
| demo_size     | 8       | Demographic features            |

---

## Expected Results (from paper)

| Metric             | Value         |
|--------------------|---------------|
| Jaccard Score      | 0.409 - 0.563 |
| Mortality Reduction| 4.4%          |

---

## Troubleshooting

**ImportError: cannot import name 'merge'**
Make sure you set the env variable before any imports:
```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

**BigQuery permission denied**
Make sure you have completed all 7 prerequisites above
and your Gmail is linked on https://physionet.org/settings/cloud/

**Colab disconnects during training**
Checkpoints are saved to Google Drive every 1000 episodes.
Re-run Steps 1, 2, 5, 6, 7 and training resumes from last checkpoint.

**tensorflow version not found**
Use tensorflow==2.19.0 — earlier versions are not available
on Colab with Python 3.12.

## Reference

1 [https://github.com/yanpanlau/DDPG-Keras-Torcs]

2 [https://github.com/songrotek/DDPG]

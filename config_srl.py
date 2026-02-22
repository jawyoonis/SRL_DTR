
# %%writefile /content/SRL_DTR/config_srl.py
class config:
    df_pkl         = "/content/drive/MyDrive/CSE6250_final_project/processed/train_all_12_31_scale.csv"
    val_df_pkl     = "/content/drive/MyDrive/CSE6250_final_project/processed/val_all_12_31_scale.csv"
    df_disease_pkl = "/content/drive/MyDrive/CSE6250_final_project/processed/train_di_base.csv"
    val_df_di_pkl  = "/content/drive/MyDrive/CSE6250_final_project/processed/val_di_base.csv"
    df_sta_pkl     = "/content/drive/MyDrive/CSE6250_final_project/processed/train_stastic_12_23.csv"
    val_sta_pkl    = "/content/drive/MyDrive/CSE6250_final_project/processed/val_stastic_12_23.csv"
    batch_size    = 30
    gamma         = 0.99
    tau           = 0.001
    lra           = 0.001
    lrc           = 0.005
    epsilon       = 0.5
    tiem_stamp    = 5
    lab_size      = 12
    demo_size     = 8
    max_reward    = 30
    state_dim     = 12
    med_size      = 180
    episode_count = 100000
    seed          = 1337
    model_seed    = 42

def get_config():
    return config()
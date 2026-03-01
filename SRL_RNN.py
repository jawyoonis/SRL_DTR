import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import jaccard_score
from ActorNetwork1  import ActorNetwork
from CriticNetwork1 import CriticNetwork
from config_srl    import config

tf.random.set_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

CKPT_DIR = "/content/drive/MyDrive/CSE6250_final_project/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

df         = pd.read_csv(config.df_pkl)
val_df     = pd.read_csv(config.val_df_pkl)
df_disease = pd.read_csv(config.df_disease_pkl)
val_df_di  = pd.read_csv(config.val_df_di_pkl)
df_sta     = pd.read_csv(config.df_sta_pkl)
val_sta    = pd.read_csv(config.val_sta_pkl)

med_size   = config.med_size
di_size    = 39
tiem_stamp = config.tiem_stamp

di_cols   = [str(i) for i in range(di_size)]
ac_cols   = ["l" + str(i) for i in range(med_size)]
demo_cols = ["sofa", "GENDER", "RELIGION", "MARITAL_STATUS",
             "age", "weight", "height", "language", "ethnicity"]
lab_cols  = ["dbp", "fio2", "GCS", "blood_glucose", "sbp",
             "hr", "PH", "rr", "bos", "temp", "urine_output"]

val_df_di_arr = val_df_di.drop("hadm_id", axis=1).values
val_sta_arr   = val_sta.drop("hadm_id", axis=1).values if "hadm_id" in val_sta.columns else val_sta.values
unique_id     = df["hadm_id"].drop_duplicates().values

print(f"Data loaded — {len(unique_id):,} patients")


def make_sequences(hadm_ids, dataframe, disease_df, demo_df, T=5):
    states_list, disease_list, demo_list = [], [], []
    actions_list, rewards_list, dones_list = [], [], []
    for hid in hadm_ids:
        traj     = dataframe[dataframe["hadm_id"] == hid]
        di_vals  = disease_df[di_cols][disease_df["hadm_id"] == hid].values
        dem_vals = demo_df[demo_cols][demo_df["hadm_id"] == hid].values
        if len(di_vals) == 0 or len(dem_vals) == 0:
            continue
        di_vals  = di_vals[0]
        dem_vals = dem_vals[0]
        labs  = traj[lab_cols].values
        acts  = traj[ac_cols].values
        flags = traj["flag"].values
        days  = len(labs)
        if days >= T:
            labs  = labs[-T:]
            acts  = acts[-T:]
            flags = flags[-T:]
        else:
            pad   = T - days
            labs  = np.vstack([np.zeros((pad, config.state_dim)), labs])
            acts  = np.vstack([np.zeros((pad, med_size)),         acts])
            flags = np.concatenate([np.zeros(pad), flags])
        reward = flags[-1]
        done   = 1 if reward != 0 else 0
        states_list.append(labs)
        disease_list.append(di_vals)
        demo_list.append(dem_vals)
        actions_list.append(acts)
        rewards_list.append(reward)
        dones_list.append(done)
    return (np.array(states_list),
            np.array(disease_list),
            np.array(demo_list),
            np.array(actions_list),
            np.array(rewards_list),
            np.array(dones_list))


print("Building validation sequences...")
val_ids = val_df["hadm_id"].unique()
val_states, val_diseases, val_demos, val_actions, val_rewards, val_dones = make_sequences(
    val_ids, val_df, val_df_di, val_sta, T=tiem_stamp)
print(f"  val_states : {val_states.shape}")
print(f"  val_demos  : {val_demos.shape}")


class SRL_RNN:
    def __init__(self, config):
        self.config = config
        np.random.seed(config.model_seed)

    def batch(self, batch_size):
        ids = np.random.choice(unique_id, size=batch_size, replace=True)
        return make_sequences(ids, df, df_disease, df_sta, T=tiem_stamp)

    def DTR(self):
        cfg           = self.config
        BATCH_SIZE    = cfg.batch_size
        GAMMA         = cfg.gamma
        TAU           = cfg.tau
        LRA           = cfg.lra
        LRC           = cfg.lrc
        epsilon       = cfg.epsilon
        lab_size      = cfg.lab_size
        demo_size     = cfg.demo_size
        max_reward    = cfg.max_reward
        action_dim    = cfg.med_size
        state_dim     = cfg.state_dim
        episode_count = cfg.episode_count

        actor  = ActorNetwork(None, state_dim, action_dim, BATCH_SIZE, TAU,
                              LRA, epsilon, tiem_stamp, med_size,
                              lab_size, demo_size, di_size)
        critic = CriticNetwork(None, state_dim, action_dim, BATCH_SIZE, TAU,
                               LRC, epsilon, tiem_stamp, med_size,
                               lab_size, demo_size, di_size, action_dim)

        try:
            actor.load(f"{CKPT_DIR}/actor_best.weights.h5")
            critic.load(f"{CKPT_DIR}/critic_best.weights.h5")
            print("Weights loaded successfully")
        except:
            print("No existing weights found, starting fresh")

        jac, qv      = [], []
        best_jaccard = 0.0

        for i in range(episode_count):
            states, diseases, demos, actions, rewards, dones = self.batch(BATCH_SIZE)

            ac_target = actor.target_model.predict([states, diseases, demos], verbose=0)
            target_q  = critic.target_model.predict([states, ac_target, diseases, demos], verbose=0)
            target_q  = np.clip(target_q, -max_reward, max_reward)

            N   = states.shape[0]
            y_t = np.array([
                rewards[k] if dones[k] == 1
                else rewards[k] + GAMMA * float(np.mean(target_q[k]))
                for k in range(N)
            ])

            a_for_grad = actor.model.predict([states, diseases, demos], verbose=0)
            critic.train_on_batch(states, diseases, demos, a_for_grad, y_t)
            grads      = critic.gradients(states, diseases, demos, a_for_grad)
            actor.train(states, diseases, demos, actions, grads)
            actor.target_train()
            critic.target_train()

            if i % 10 == 0:
                preds    = actor.model.predict([val_states, val_diseases, val_demos], verbose=0)
                target_q = critic.target_model.predict([val_states, preds, val_diseases, val_demos], verbose=0)
                q        = float(np.mean(target_q))
                pred_bin = (preds[:, -1, :] >= 0.5).astype(int)
                act_last = val_actions[:, -1, :]
                j = jaccard_score(act_last.astype(int), pred_bin,
                                  average="samples", zero_division=0)
                print(f"Episode {i:6d} | Q: {q:.4f} | Jaccard: {j:.4f}")

                if i % 100 == 0:
                    jac.append(j)
                    qv.append(q)

                if j > best_jaccard:
                    best_jaccard = j
                    actor.save(f"{CKPT_DIR}/actor_best.weights.h5")
                    critic.save(f"{CKPT_DIR}/critic_best.weights.h5")
                    print(f"  Best model saved (Jaccard={best_jaccard:.4f})")

                if i % 1000 == 0 and i > 0:
                    actor.save(f"{CKPT_DIR}/actor_ep{i}.weights.h5")
                    critic.save(f"{CKPT_DIR}/critic_ep{i}.weights.h5")
                    np.save(f"{CKPT_DIR}/jac_{i}.npy", np.array(jac))
                    np.save(f"{CKPT_DIR}/qv_{i}.npy",  np.array(qv))

        print(f"Training complete. Best Jaccard: {best_jaccard:.4f}")


if __name__ == "__main__":
    model = SRL_RNN(config)
    model.DTR()

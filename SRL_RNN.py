# %%writefile /content/SRL_DTR/srl_rnn.py
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

med_size = config.med_size
di_size  = 39

di_cols   = [str(i) for i in range(di_size)]
ac_cols   = ["l" + str(i) for i in range(med_size)]
demo_cols = ["sofa", "GENDER", "RELIGION", "MARITAL_STATUS",
             "age", "weight", "height", "language", "ethnicity"]
lab_cols  = ["dbp", "fio2", "GCS", "blood_glucose", "sbp",
             "hr", "PH", "rr", "bos", "temp", "urine_output"]

val_df_di_arr = val_df_di.drop("hadm_id", axis=1).values
val_sta_arr   = val_sta.drop("hadm_id", axis=1).values \
                if "hadm_id" in val_sta.columns else val_sta.values

unique_id = df["hadm_id"].drop_duplicates().values

class SRL_RNN:
    def __init__(self, config):
        self.config = config
        np.random.seed(config.model_seed)

    def batch(self, batch_size):
        states, meds, rewards, next_states, done_flags = None, None, None, None, None
        disease_list, demo_list = [], []

        for _ in range(batch_size):
            traj_id   = np.random.choice(unique_id)
            a         = df.loc[df["hadm_id"] == traj_id]
            di_vals   = df_disease[di_cols][df_disease["hadm_id"] == traj_id].values
            demo_vals = df_sta[demo_cols][df_sta["hadm_id"] == traj_id].values
            x         = 0

            for i in a.index:
                disease_list.append(di_vals)
                demo_list.append(demo_vals)
                x    += 1
                state  = np.reshape(a.loc[i, lab_cols].values,  [1, config.state_dim])
                med    = np.reshape(a.loc[i, ac_cols].values,   [1, med_size])
                reward = a.loc[i, "flag"]

                if x < len(a):
                    next_idx   = a.index[x]
                    next_state = np.reshape(df.loc[next_idx, lab_cols].values,
                                           [1, config.state_dim])
                    done = 0
                else:
                    next_state = np.reshape(np.zeros(config.state_dim),
                                           [1, config.state_dim])
                    done = 1

                states      = state      if states      is None else np.vstack((states,      state))
                meds        = med        if meds        is None else np.vstack((meds,        med))
                rewards     = [reward]   if rewards     is None else np.vstack((rewards,     reward))
                next_states = next_state if next_states is None else np.vstack((next_states, next_state))
                done_flags  = [done]     if done_flags  is None else np.vstack((done_flags,  done))

        return (states,
                np.squeeze(meds),
                np.squeeze(rewards),
                next_states,
                np.squeeze(done_flags),
                np.squeeze(np.array(disease_list)),
                np.squeeze(np.array(demo_list)))

    def DTR(self):
        cfg           = self.config
        BATCH_SIZE    = cfg.batch_size
        GAMMA         = cfg.gamma
        TAU           = cfg.tau
        LRA           = cfg.lra
        LRC           = cfg.lrc
        epsilon       = cfg.epsilon
        tiem_stamp    = cfg.tiem_stamp
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
            print("✓ Weights loaded successfully")
        except:
            print("No existing weights found, starting fresh")

        jac, qv      = [], []
        best_jaccard = 0.0

        for i in range(episode_count):
            states, actions, rewards, new_states, dones, diseases, demos = \
                self.batch(BATCH_SIZE)

            ac_target = actor.target_model.predict(
                [new_states, diseases, demos], verbose=0
            )
            target_q  = critic.target_model.predict(
                [new_states, ac_target, diseases, demos], verbose=0
            )
            target_q  = np.clip(target_q, -max_reward, max_reward)

            len1 = states.shape[0]
            y_t  = np.array([
                rewards[k] if dones[k] == 1
                else rewards[k] + GAMMA * float(np.mean(target_q[k]))
                for k in range(len1)
            ])

            a_for_grad = actor.model.predict([states, diseases, demos], verbose=0)
            critic.train_on_batch(states, diseases, demos, a_for_grad, y_t)
            grads      = critic.gradients(states, diseases, demos, a_for_grad)
            actor.train(states, diseases, demos, actions, grads)

            actor.target_train()
            critic.target_train()

            if i % 10 == 0:
                preds    = actor.model.predict(
                    [val_df[lab_cols].values, val_df_di_arr, val_sta_arr], verbose=0
                )
                target_q = critic.target_model.predict(
                    [val_df[lab_cols].values, preds, val_df_di_arr, val_sta_arr],
                    verbose=0
                )
                q        = float(np.mean(target_q))
                pred_bin = (preds >= 0.5).astype(int)
                j        = jaccard_score(
                    val_df[ac_cols].values.astype(int),
                    pred_bin,
                    average="samples",
                    zero_division=0
                )
                print(f"Episode {i:6d} | Q: {q:.4f} | Jaccard: {j:.4f}")

                if i % 100 == 0:
                    jac.append(j)
                    qv.append(q)

                if j > best_jaccard:
                    best_jaccard = j
                    actor.save(f"{CKPT_DIR}/actor_best.weights.h5")
                    critic.save(f"{CKPT_DIR}/critic_best.weights.h5")
                    print(f"  ✓ Best model saved (Jaccard={best_jaccard:.4f})")

                if i % 1000 == 0 and i > 0:
                    actor.save(f"{CKPT_DIR}/actor_ep{i}.weights.h5")
                    critic.save(f"{CKPT_DIR}/critic_ep{i}.weights.h5")
                    np.save(f"{CKPT_DIR}/jac_{i}.npy", np.array(jac))
                    np.save(f"{CKPT_DIR}/qv_{i}.npy",  np.array(qv))

        print(f"\nTraining complete. Best Jaccard: {best_jaccard:.4f}")


if __name__ == "__main__":
    model = SRL_RNN(config)
    model.DTR()
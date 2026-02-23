import os
import sys
import random
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"
sys.path.insert(0, '/content/SRL_DTR')

from config_srl     import config
from ActorNetwork1  import ActorNetwork
from CriticNetwork1 import CriticNetwork

# ── Dummy data matching real data shapes ──────────────────────
BATCH      = 4
TIME_STAMP = config.tiem_stamp   # 5
LAB_SIZE   = config.lab_size     # 12
DEMO_SIZE  = config.demo_size    # 8
MED_SIZE   = config.med_size     # 180
DI_SIZE    = 39

dummy_lab     = np.random.rand(BATCH, TIME_STAMP, LAB_SIZE).astype(np.float32)
dummy_di      = np.random.randint(0, DI_SIZE, (BATCH, DI_SIZE)).astype(np.int32)
dummy_demo    = np.random.rand(BATCH, DEMO_SIZE).astype(np.float32)
dummy_actions = np.random.rand(BATCH, TIME_STAMP, MED_SIZE).astype(np.float32)
dummy_labels  = np.random.randint(0, 2, (BATCH, TIME_STAMP, MED_SIZE)).astype(np.float32)
dummy_targets = np.random.rand(BATCH, TIME_STAMP, 1).astype(np.float32)

print("✓ Dummy data created")
print(f"  lab     : {dummy_lab.shape}")
print(f"  di      : {dummy_di.shape}")
print(f"  demo    : {dummy_demo.shape}")
print(f"  actions : {dummy_actions.shape}")

# ── Build networks ────────────────────────────────────────────
actor = ActorNetwork(
    None, config.state_dim, MED_SIZE,
    config.batch_size, config.tau, config.lra,
    config.epsilon, TIME_STAMP, MED_SIZE,
    LAB_SIZE, DEMO_SIZE, DI_SIZE
)
print("✓ Actor built")

critic = CriticNetwork(
    None, config.state_dim, MED_SIZE,
    config.batch_size, config.tau, config.lrc,
    config.epsilon, TIME_STAMP, MED_SIZE,
    LAB_SIZE, DEMO_SIZE, DI_SIZE, MED_SIZE
)
print("✓ Critic built")

# ── Forward passes ────────────────────────────────────────────
pred  = actor.model.predict([dummy_lab, dummy_di, dummy_demo], verbose=0)
print(f"✓ Actor forward  : {pred.shape}")

q     = critic.model.predict([dummy_lab, dummy_actions, dummy_di, dummy_demo], verbose=0)
print(f"✓ Critic forward : {q.shape}")

grads = critic.gradients(dummy_lab, dummy_di, dummy_demo, dummy_actions)
print(f"✓ Gradients      : {grads.shape}")

# ── Training steps ────────────────────────────────────────────
a_loss = actor.train(dummy_lab, dummy_di, dummy_demo, dummy_labels, grads)
print(f"✓ Actor loss     : {a_loss:.4f}")

c_loss = critic.train_on_batch(dummy_lab, dummy_di, dummy_demo, dummy_actions, dummy_targets)
print(f"✓ Critic loss    : {c_loss:.4f}")

# ── Soft target updates ───────────────────────────────────────
actor.target_train()
critic.target_train()
print("✓ Target networks updated")

print("\n✓ ALL TESTS PASSED — ready for real data")

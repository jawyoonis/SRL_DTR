# %%writefile /content/SRL_DTR/CriticNetwork1.py
import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras.models import Model
from tf_keras.layers import (
    Input, LSTM, Dense, Embedding, Dropout, Masking,
    TimeDistributed, concatenate, RepeatVector, Lambda, Add
)
from tf_keras.optimizers import Adam
import tf_keras.backend as K

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180

def avg(t):
    return K.mean(t, -2)

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU,
                 LEARNING_RATE, epsilon, tiem_stamp, med_size, lab_size,
                 demo_size, di_size, action_dim):
        self.BATCH_SIZE    = BATCH_SIZE
        self.TAU           = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.epsilon       = epsilon
        self.time_stamp    = tiem_stamp
        self.med_size      = med_size
        self.lab_size      = lab_size
        self.demo_size     = demo_size
        self.di_size       = di_size
        self.action_dim    = action_dim

        self.model        = self.create_critic_network(state_size, action_size)
        self.target_model = self.create_critic_network(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer    = Adam(learning_rate=LEARNING_RATE)

    def create_critic_network(self, state_size, action_size):
        main_input_lab  = Input(shape=(self.time_stamp, self.lab_size),
                                dtype="float32", name="lab_input")
        main_input_demo = Input(shape=(self.demo_size,),
                                dtype="float32", name="demo_input")
        main_input_di   = Input(shape=(self.di_size,),
                                dtype="int32", name="di_input")
        action_input    = Input(shape=(self.time_stamp, self.action_dim),
                                dtype="float32", name="action_input")

        d1      = Dropout(0.1)(main_input_lab)
        demo    = Dense(HIDDEN1_UNITS, activation="relu")(main_input_demo)
        demo    = RepeatVector(self.time_stamp)(demo)

        d2      = Dropout(0.1)(main_input_di)
        e1      = Embedding(output_dim=HIDDEN1_UNITS, input_dim=2001,
                            input_length=self.di_size, mask_zero=True)(d2)
        emb_out = Lambda(avg)(e1)
        emb_out = RepeatVector(self.time_stamp)(emb_out)
        emb_out = TimeDistributed(Dense(HIDDEN1_UNITS, activation="relu"))(emb_out)

        m1     = Masking(mask_value=0)(d1)
        l1     = LSTM(units=HIDDEN2_UNITS, return_sequences=True)(m1)

        # merged shape: (batch, time, 180+40+40) = (batch, time, 260)
        merged = concatenate([l1, emb_out, demo])

        # ── FIX: match action dense output to merged size (260) ──
        a1     = TimeDistributed(Dense(260, activation="linear"))(action_input)
        h2     = Add()([merged, a1])
        output = TimeDistributed(Dense(1, activation="linear"))(h2)

        model  = Model(
            inputs=[main_input_lab, action_input, main_input_di, main_input_demo],
            outputs=output
        )
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.LEARNING_RATE))
        return model

    def gradients(self, states, disease, demos, actions):
        actions_var = tf.Variable(tf.cast(actions, tf.float32), trainable=True)
        with tf.GradientTape() as tape:
            tape.watch(actions_var)
            q = self.model([states, actions_var, disease, demos], training=False)
        return tape.gradient(q, actions_var).numpy()

    def train_on_batch(self, states, disease, demos, actions, targets):
        return self.model.train_on_batch(
            [states, actions, disease, demos], targets
        )

    def target_train(self):
        cw = self.model.get_weights()
        tw = self.target_model.get_weights()
        self.target_model.set_weights([
            self.TAU * c + (1 - self.TAU) * t for c, t in zip(cw, tw)
        ])

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
        self.target_model.set_weights(self.model.get_weights())
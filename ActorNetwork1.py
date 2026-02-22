# %%writefile /content/SRL_DTR/ActorNetwork.py
import numpy as np
import tensorflow as tf
import tf_keras
from tf_keras.models import Model
from tf_keras.layers import (
    Input, LSTM, Dense, Embedding, Dropout, Masking,
    TimeDistributed, concatenate, RepeatVector, Lambda
)
from tf_keras.optimizers import Adam
import tf_keras.backend as K

HIDDEN1_UNITS = 40
HIDDEN2_UNITS = 180

def avg(t):
    return K.mean(t, -2)

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU,
                 LEARNING_RATE, epsilon, tiem_stamp, med_size, lab_size,
                 demo_size, di_size):
        self.BATCH_SIZE    = BATCH_SIZE
        self.TAU           = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.epsilon       = epsilon
        self.time_stamp    = tiem_stamp
        self.med_size      = med_size
        self.lab_size      = lab_size
        self.demo_size     = demo_size
        self.di_size       = di_size

        self.model        = self.create_actor_network(state_size, action_size)
        self.target_model = self.create_actor_network(state_size, action_size)
        self.target_model.set_weights(self.model.get_weights())
        self.optimizer    = Adam(learning_rate=LEARNING_RATE)

    def create_actor_network(self, state_size, action_size):
        main_input_lab  = Input(shape=(self.time_stamp, self.lab_size),
                                dtype="float32", name="lab_input")
        main_input_demo = Input(shape=(self.demo_size,),
                                dtype="float32", name="demo_input")
        main_input_di   = Input(shape=(self.di_size,),
                                dtype="int32", name="di_input")

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

        merged = concatenate([l1, emb_out, demo])
        output = TimeDistributed(
            Dense(self.med_size, activation="sigmoid")
        )(merged)

        model  = Model(
            inputs=[main_input_lab, main_input_di, main_input_demo],
            outputs=output
        )
        return model

    def train(self, states, disease, demos, lable, action_grads):
        with tf.GradientTape() as tape:
            predicted  = self.model([states, disease, demos], training=True)
            sl_loss    = tf.reduce_mean(
                tf_keras.losses.binary_crossentropy(
                    tf.cast(lable, tf.float32), predicted
                )
            )
            rl_loss    = -tf.reduce_mean(
                tf.reduce_sum(
                    tf.cast(action_grads, tf.float32) * predicted, axis=-1
                )
            )
            total_loss = self.epsilon * rl_loss + (1 - self.epsilon) * sl_loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss.numpy()

    def target_train(self):
        aw = self.model.get_weights()
        tw = self.target_model.get_weights()
        self.target_model.set_weights([
            self.TAU * a + (1 - self.TAU) * t for a, t in zip(aw, tw)
        ])

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
        self.target_model.set_weights(self.model.get_weights())
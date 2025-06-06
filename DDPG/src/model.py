"""
The main model declaration
"""
import logging
import os

import numpy as np
import tensorflow as tf



from buffer import ReplayBuffer
from noise import OUActionNoise



GAMMA = 0.99 
RHO = 0.005 

STD_DEV = 0.2
BATCH_SIZE = 300
BUFFER_SIZE = 1e6
CRITIC_LR = 1e-3
ACTOR_LR = 1e-4

def ActorNetwork(num_states=24, num_actions=4,):
    last_init = tf.random_normal_initializer(stddev=0.0005)
    inputs = tf.keras.layers.Input(shape=((num_states,)), dtype=tf.float32)
    out = tf.keras.layers.Dense(900, activation=tf.nn.leaky_relu)(inputs)
    out= tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dense(450, activation=tf.nn.leaky_relu)(out)
    out= tf.keras.layers.BatchNormalization()(out)
    outputs = tf.keras.layers.Dense(units=num_actions, activation="tanh", kernel_initializer=last_init)(out)
    model = tf.keras.Model(inputs, outputs)
    model.summary()
    return model


def CriticNetwork(num_states=24, num_actions=4):
    last_init = tf.random_normal_initializer(stddev=0.00005)
    state_input = tf.keras.layers.Input(shape=(num_states,), dtype=tf.float32)
    state_out = tf.keras.layers.Dense(900, activation=tf.nn.leaky_relu)(state_input)
    state_out= tf.keras.layers.BatchNormalization()(state_out)
    state_out = tf.keras.layers.Dense(450, activation=tf.nn.leaky_relu)(state_out)
    state_out= tf.keras.layers.BatchNormalization()(state_out)
    action_input = tf.keras.layers.Input(shape=(num_actions,), dtype=tf.float32)
    action_out = tf.keras.layers.Dense(450, activation=tf.nn.leaky_relu)(action_input)
    action_out= tf.keras.layers.BatchNormalization()(action_out)
    added = tf.keras.layers.Add()([state_out, action_out])
    added= tf.keras.layers.BatchNormalization()(added)
    outs = tf.keras.layers.Dense(450, activation=tf.nn.leaky_relu)(added)
    out= tf.keras.layers.BatchNormalization()(outs)
    out = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu)(out)
    out= tf.keras.layers.BatchNormalization()(out)
    outputs = tf.keras.layers.Dense(1, kernel_initializer=last_init)(out)
    model = tf.keras.Model([state_input, action_input], outputs)
    model.summary()
    return model


class DDPG:  # pylint: disable=too-many-instance-attributes
    """
    The Brain that contains all the models
    """

    def __init__(
        self, num_states, num_actions, action_high, action_low, gamma=GAMMA, rho=RHO,
        std_dev=STD_DEV
    ):  
        self.actor_network = ActorNetwork(num_states, num_actions)
        self.critic_network = CriticNetwork(num_states, num_actions)
        self.actor_target = ActorNetwork(num_states, num_actions)
        self.critic_target = CriticNetwork(num_states, num_actions)

        self.actor_target.set_weights(self.actor_network.get_weights())
        self.critic_target.set_weights(self.critic_network.get_weights())

        self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.gamma = tf.constant(gamma)
        self.rho = rho
        self.action_high = action_high
        self.action_low = action_low
        self.num_states = num_states
        self.num_actions = num_actions


        self.noise = OUActionNoise(mean=np.zeros(4), std_deviation=float(std_dev) * np.ones(4))
        self.critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR, amsgrad=True,weight_decay=1e-4)
        self.actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR, amsgrad=True)


        self.cur_action = None

        # define update weights with tf.function for improved performance
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, num_states), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_actions), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, num_states), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
            ])
        def update_weights(s, a, r, sn, d):

            with tf.GradientTape() as tape:
                y = r + self.gamma * (1 - d) * self.critic_target([sn, self.actor_target(sn)])
                critic_loss = tf.math.reduce_mean(tf.math.abs(y - self.critic_network([s, a])))
            critic_grad = tape.gradient(critic_loss, self.critic_network.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic_network.trainable_variables))

            with tf.GradientTape() as tape:
                actor_loss = -tf.math.reduce_mean(self.critic_network([s, self.actor_network(s)]))
            actor_grad = tape.gradient(actor_loss, self.actor_network.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor_network.trainable_variables))
            return critic_loss, actor_loss

        self.update_weights = update_weights


    def act(self, state, noise=True):

        self.cur_action = self.actor_network(state)[0].numpy()+ (self.noise() if noise else 0)

        self.cur_action = np.clip(self.cur_action, self.action_low, self.action_high)
        return self.cur_action

    def remember(self, prev_state, reward, state, done):

        self.buffer.append(prev_state, self.cur_action, reward, state, done)


    @staticmethod
    def _update_target(model_target, model_ref, rho=0):
        model_target.set_weights(
            [
                rho * ref_weight + (1 - rho) * target_weight
                for (target_weight, ref_weight)
                in list(zip(model_target.get_weights(), model_ref.get_weights()))
            ]
        )

    def learn(self, entry):
        s, a, r, sn, d = zip(*entry)

        c_l, a_l = self.update_weights(tf.convert_to_tensor(s, dtype=tf.float32),
                                       tf.convert_to_tensor(a, dtype=tf.float32),
                                       tf.convert_to_tensor(r, dtype=tf.float32),
                                       tf.convert_to_tensor(sn, dtype=tf.float32),
                                       tf.convert_to_tensor(d, dtype=tf.float32))

        self._update_target(self.actor_target, self.actor_network, self.rho)
        self._update_target(self.critic_target, self.critic_network, self.rho)
        return c_l, a_l

    def save_models(self, path):

        parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        self.actor_network.save(path + "an.keras")
        self.critic_network.save(path + "cn.keras")
        self.critic_target.save(path + "ct.keras")
        self.actor_target.save(path + "at.keras")

    def load_models(self, path):
        try:
            self.actor_network.load_model(path + "an.keras")
            self.critic_network.load_model(path + "cn.keras")
            self.critic_target.load_model(path + "ct.keras")
            self.actor_target.load_model(path + "at.keras")
        except OSError as err:
            logging.warning("Weights files cannot be found, %s", err)

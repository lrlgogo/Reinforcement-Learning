import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Tuple
import numpy as np
import gym
import tqdm


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
del gpu

max_episodes = 10000
max_steps_per_episode = 200
rewards_threshold = 195
running_rewards = 0.

gamma = 0.99
num_hidden_layer_units = 128

env = gym.make('CartPole-v0')

Global_Seed = 42
np.random.seed(Global_Seed)
tf.random.set_seed(Global_Seed)
env.seed(Global_Seed)

eps = np.finfo(np.float32).eps
dim_action_space = env.action_space.n

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
Huber_Loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


class NetClass(tf.keras.Model):
    def __init__(self, unit_output, unit_hidden):
        super(NetClass, self).__init__()
        self.hidden_layer = layers.Dense(unit_hidden, 'relu')
        self.output_layer = layers.Dense(unit_output)

    def call(self, input, **kwargs):
        x = self.hidden_layer(input)
        return self.output_layer(x)


actor = NetClass(dim_action_space, num_hidden_layer_units)
critic = NetClass(1, num_hidden_layer_units)


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(action)
    return (
        state.astype(np.float32),
        np.array(reward, np.int32),
        np.array(done, np.bool)
    )


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
    return tf.numpy_function(
        env_step,
        [action],
        [tf.float32, tf.int32, tf.bool]
    )


@tf.function
def train_step(
        initial_state: tf.Tensor,
        gamma: float,
        max_episode_step: tf.int32,
        actor_model: tf.keras.Model,
        critic_model: tf.keras.Model,
        optimizer: tf.keras.optimizers
) -> tf.Tensor:
    state_curr = initial_state
    state_shape = initial_state.shape

    rewards = tf.TensorArray(tf.int32, 0, True)
    state_curr = tf.expand_dims(state_curr, 0)
    for curr_step in tf.range(max_episode_step):
        with tf.GradientTape(persistent=True) as tape:

            action_net_out = actor_model(state_curr)
            value_curr_out = critic_model(state_curr)

            # action = tf.random.categorical(action_net_out, 1)[0, 0]
            action_prob_softmax = tf.nn.softmax(action_net_out)  # [0, action]
            action = tf.random.categorical(
                tf.math.log(action_prob_softmax), 1
            )[0, 0]

            state_next, reward, curr_done = tf_env_step(action)
            state_next.set_shape(state_shape)
            state_next = tf.expand_dims(state_next, 0)

            value_next_out = critic_model(state_next)

            TD_err = tf.cast(reward, dtype=tf.float32) + \
                     gamma * value_next_out - value_curr_out
            loss_actor = - TD_err * tf.math.log(action_prob_softmax[0, action])
            loss_critic = tf.square(TD_err)
            """
            loss_critic = Huber_Loss(
                tf.cast(reward, tf.float32) + gamma * value_next_out,
                value_curr_out
            )
            """

        grads_actor = tape.gradient(loss_actor, actor_model.trainable_variables)
        grads_critic = tape.gradient(
            loss_critic, critic_model.trainable_variables
        )
        optimizer.apply_gradients(
            zip(grads_actor, actor_model.trainable_variables)
        )
        optimizer.apply_gradients(
            zip(grads_critic, critic_model.trainable_variables)
        )

        state_curr = state_next
        rewards = rewards.write(curr_step, reward)
        if curr_done:
            break

    return tf.reduce_sum(rewards.stack())


with tqdm.trange(max_episodes) as T:
    for t in T:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = int(
            train_step(
                initial_state,
                gamma,
                max_steps_per_episode,
                actor,
                critic,
                optimizer
            ).numpy()
        )
        running_rewards = 0.99 * running_rewards + 0.01 * episode_reward

        T.set_description(f'Episode {t}')
        T.set_postfix(
            episode_reward=episode_reward, running_rewards=running_rewards
        )

        if running_rewards > rewards_threshold:
            break
    print(f'solved at episode {t}, average reward: {running_rewards: .4f}')
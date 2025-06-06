
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

HARDCORE=False
TOTALEPISODES=100
EPISODELENGTH=1600
env = gym.make('BipedalWalker-v3',hardcore=HARDCORE,render_mode="human")
acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
ep_reward_list = []
actor_network = tf.keras.models.load_model('checkpoints/current/DDPG_at.keras')
def main():
    for episode in range(TOTALEPISODES):
            prev_state = env.reset()[0]
            episode_reward=0
            for step in range(EPISODELENGTH):

                cur_act = np.clip(actor_network(tf.expand_dims(prev_state, 0))[0].numpy(),-1,1)
                
                state, reward, done, _,_ = env.step(cur_act)
                episode_reward += reward
                prev_state = state

                if done:
                    ep_reward_list.append(episode_reward)
                    break
            print(episode_reward," ",episode)
    plt.plot(ep_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")     
    plt.savefig('avg_test_reward.png')
    env.close()
if __name__ == "__main__":
    main()

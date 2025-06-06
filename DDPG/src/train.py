
import random

import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

from model import DDPG

TRAINING=True
TOTALEPISODES=2000
EPISODELENGTH=1600


def main(): 


    env = gym.make('BipedalWalker-v3',render_mode="human")
    model = DDPG(24, 4, 1,-1)
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
    A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)
    ep_reward_list = []
    ep_q_loss = []
    ep_a_loss = []
    Q_loss_list = []
    A_loss_list=[]
    avg_reward_list = []
    
 
    with trange(TOTALEPISODES) as t:
        for ep in t:
            prev_state = env.reset()[0]
            acc_reward.reset_state()
            Q_loss.reset_state()
            A_loss.reset_state()
            model.noise.reset()
            for step in range(EPISODELENGTH):
                cur_act = model.act(
                    tf.expand_dims(prev_state, 0),
                    noise=(
                            random.random()
                            >
                            #from 0.01 to 0.81
                            0.01+(1-0.2)*(ep/TOTALEPISODES)
                        )
                )
                state, reward, done, _,_ = env.step(cur_act)
                acc_reward(reward)
                model.remember(prev_state, reward, state, int(done))
                if TRAINING:
                    c, a = model.learn(model.buffer.sample())
                    Q_loss(c)
                    A_loss(a)
                prev_state = state
                if done:
                    break
            ep_reward_list.append(acc_reward.result().numpy())
            ep_q_loss.append(Q_loss.result().numpy())
            ep_a_loss.append(A_loss.result().numpy())
            avg_reward = np.mean(ep_reward_list[-100:])
            avg_q_loss=np.mean(ep_q_loss[-100:])
            avg_a_loss=np.mean(ep_a_loss[-100:])
            avg_reward_list.append(avg_reward)
            Q_loss_list.append(avg_q_loss)
            A_loss_list.append(avg_a_loss)
            t.set_postfix(r=avg_reward)

            if TRAINING and ep % 2 == 0 and ep!=0:
                model.save_models("checkpoints/"+"current"+"/DDPG_")
            if ep%10==0 and ep!=0 and TRAINING:
                model.save_models("checkpoints/"+str(ep)+"/DDPG_")
                plt.plot(avg_reward_list)
                plt.xlabel("Episode")
                plt.ylabel("Avg. Epsiodic Reward")
                
                plt.savefig("figs/"+str(ep)+'_avg_reward.png')
                plt.close()
                plt.plot(Q_loss_list)
                plt.xlabel("Episode")
                plt.ylabel("Avg. Epsiodic Q loss")
                plt.savefig("figs/"+str(ep)+'_Q_loss.png')

                plt.close()
                plt.plot(A_loss_list)
                
                plt.xlabel("Episode")
                plt.ylabel("Avg. Epsiodic A loss")
                plt.savefig("figs/"+str(ep)+'_A_loss.png')

                plt.close()

    env.close()
    if TRAINING:
        model.save_models("checkpoints/"+"current"+"/DDPG_")
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()
if __name__ == "__main__":
    main()

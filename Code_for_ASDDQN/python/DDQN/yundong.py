#此代码用来直接使用训练好的DQN模型
import os
import gym
import numpy as np
import parl
from parl.utils import logger, ReplayMemory
from model import Model
from cartpole_agent import CartpoleAgent
from parl.algorithms import DQN
from env import ArmEnv

#arduino模块
import serial
import time
import numpy as np
ser = serial.Serial('COM12',9600,timeout=1)
#训练的频率
LEARN_FREQ = 5  # training frequency
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
GAMMA = 0.99

# evaluate 5 episodes
def run_evaluate_episodes(agent, env, eval_episodes=5, render=False):
    eval_reward = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            #向arduino发送电机驱动信号
            val = ser.write('str(action)'.encode('utf-8'))
            #给arduino预留一些响应时间
            time.sleep(0.4)
            val2 = ser.readline().decode('utf-8')
            print(val2)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def main():
    #env = gym.make('CartPole-v0')
    env = ArmEnv()
    #obs_dim = env.observation_space.shape[0]
    #状态量
    obs_dim = env.state_dim
    #动作维度
    act_dim = env.action_dim
    #act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # set action_shape = 0 while in discrete control environment
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)

    # build an agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = CartpoleAgent(
        alg, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-6)

    max_episode = 800

    episode = 0
    while episode < max_episode:
        # test part
        eval_reward = run_evaluate_episodes(agent, env, render=False)
        logger.info('episode:{}    e_greed:{}   Test reward:{}'.format(
            episode, agent.e_greed, eval_reward))

    # save the parameters to ./model.ckpt训练结束，保存模型
    save_path = './model.ckpt'
    agent.save(save_path)

    # save the model and parameters of policy network for inference
    #save_inference_path = './inference_model'
    #input_shapes = [[None, env.observation_space.shape[0]]]
    #input_dtypes = ['float32']
    #agent.save_inference_model(save_inference_path, input_shapes, input_dtypes)


if __name__ == '__main__':
    main()

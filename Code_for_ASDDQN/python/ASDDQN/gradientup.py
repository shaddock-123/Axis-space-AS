#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
#######################################################################
#参数定义
import numpy as np
k_arm = 10
true_reward = np.random.randn(k_arm)            #每个赌博机真实奖励值，服从N(0,1)
best_action = np.argmax(true_reward)            #最优动作
Q_estimation = np.zeros(k_arm)                  #对每个赌博机的奖励估计值Q
global_time = 1                                 #走过的步数（投币次数）
N_steps = 1000                                  #总步数（总金币数）
average_reward = 0.0                            #初始化平均奖励值
action_prob = np.zeros(k_arm)                   #选择动作的概率

#动作选择
def choose_action():
    #严格按照softmax来选择概率最大的动作
    #在知道目标位置的情况下，对action进行选择，同时对其他action削减概率
    exp_est = np.exp(Q_estimation)
    action_prob = exp_est / np.sum(exp_est)
    return np.random.choice(np.arange(k_arm), p=action_prob)

#获取奖励
def get_reward(action, gradient_baseline=True, Gradient_Bandit_alpha=0.1):
    reward = np.random.randn() + true_reward[action]
    global global_time,average_reward,Q_estimation
    global_time += 1
    average_reward = (global_time - 1.0) / global_time * average_reward + reward / global_time
    one_hot = np.zeros(k_arm)
    one_hot[action] = 1
    if gradient_baseline:
        baseline = average_reward
    else:
        baseline = 0
    #偏好更新公式！运用了one_hot数组，保证了选中与未选中的动作往不同方向更新
    Q_estimation = Q_estimation + Gradient_Bandit_alpha * (reward - baseline) * (one_hot - action_prob)

if __name__ == '__main__':
    #保存1000步的奖励
    gradient_bandit_rewards = []
    gradient_bandit_best_actions = 0

    for i in range(N_steps):
        gradient_bandit_action = choose_action()
        if gradient_bandit_action == best_action:
            gradient_bandit_best_actions += 1
        
        gradient_bandit_rewards.append(get_reward(gradient_bandit_action))
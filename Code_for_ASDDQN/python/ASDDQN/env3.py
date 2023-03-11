import numpy as np
import Seeker_SDK_ClientGetAFrame as S
import test_seeker
#从动捕系统中获取坐标位置
#Coor = S.b
import time
import math
import os
from seeker.seekersdk import *
import serial
import time
preFrmNo = 0
curFrmNo = 0
env_step = 1
import xlwt
#保存数据训练的步数和每一步带来的奖励

#print(Coor)
#从动捕系统测试中获取坐标位置
#Coor = test_seeker.test_seeker_func()
class ArmEnv3(object):
    #状态的量
    state_dim = 1
    #动作的量
    action_dim = 4 
    #每一步的步进量
    #action_bound = [-1,1]
    #refresh rate
    dt = 0.1 

    #目标位置
    goal = [153.,361.,1187.]
    #机械臂的末端位置
    #print(Coor[0][0])
    #print(Coor[0][1])
    #print(Coor[0][2])
    #obs = [Coor[0][0], Coor[0][1],Coor[0][2]]
    #obs=[123.,156.,156.]
    def __init__(self):
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        self.on_goal = 0
        #high = np.array(
        #    [
        #        1,
        #        1,
        #        1,
        #    ],
        #    dtype=np.float32,
        #)

        #self.action_space = spaces.Discrete(4)
        #self.observation_space = spaces.Box(-high, high, dtype=np.float32)
    
    def step(self,action):
        done = False
        action =action
        global env_step
        print("step func begin")
        #action = np.clip(action,*self.action_bound)
        #求reward
        #求出观察值state
        #Coor = S.b
        Coor=get_obs()
        print(np.size(Coor[1]))
        obs_ = [Coor[0], Coor[1],Coor[2]]
        print(obs_)
        print(self.goal)
        state = [(self.goal[0]-obs_[0]),(self.goal[1]-obs_[1]),(self.goal[2]-obs_[2])]
        print({"dist"},state)
        #求出空间距离
        dist = np.sqrt(state[0]**2+state[1]**2+state[2]**2)
        #sheet2.write(eval_step,0,eval_step)
        ########改为dist
        #sheet2.write(eval_step,1,reward)
        env_step +=1
        print({"dist"},dist)
        p = np.sqrt(state[0]**2)
        wb7=xlwt.Workbook()
        sheet7=wb7.add_sheet('tradddin_steps&train_rewards')
        text1 = ["total_taidddn_steps","reward_every_step"]
        for i in range(len(text1)):
            sheet7.write(0,i,text1[i])
        sheet7.write(env_step,0,env_step)
        sheet7.write(env_step,1,p)
        wb7.save('total_step&train_reward.xls')
        #done and reward 允许最终的末端位置误差为1mm,以内，并且停留10次则视为训练结束
        r=-np.sqrt(state[0]**2)*math.exp(p/10)
        print({"rewards"},r)
        if p<0.4:
            self.on_goal +=1
            if self.on_goal >=1:
                done =True
        else:
            self.on_goal = 0
        next_obs = [Coor[0]]
        #print(next_obs)
        return next_obs, r, done,{}
    
    def reset(self):
        self.goal[0] = np.random.uniform(165,165)
        self.goal[1] = np.random.uniform(115,135)
        self.goal[2] = np.random.uniform(1179,1188)
        #self.goal[0] =159.590
        self.goal[0] =391
        #self.goal[2] = 1182.856445
        self.on_goal = 0
        Coor = get_obs()
        obs = [Coor[0]]
        #dist= [(self.goal[0]-obs_[0]),(self.goal[1]-obs[1]),(self.goal[2]-obs[2])]
        obs = [(self.goal[0]-obs[0])]
        return obs

    def sample_action(self):
        #print(np.random.randint(1,8))
        return np.random.randint(0,7)
        
def get_obs():
    print("Started the Seeker_SDK_Client Demo")

    version = (c_ubyte * 4)()
    GetSdkVersion(version)
    print("VERSION:%d.%d.%d.%d" % (version[0], version[1], version[2], version[3]))

    SetVerbosityLevel(1)
    SetErrorMsgHandlerFunc(S.py_msg_func)

    # Change to your local ip in 10.1.1.0/24
    print("Begin to init the SDK Client")
    ret = Initialize(b"10.1.1.198", b"10.1.1.198")

    #if ret == 0:
    #    print("Connect to the Seeker Succeed")
    #else:
    #    print("Connect Failed: [%d]" % ret)
    #    Exit()
    #    exit(0)
  
    hostInfo = HostInfo()
    GetHostInfo(pointer(hostInfo))
    if hostInfo.bFoundHost == 1:
        print("Founded the Seeker Server")
    else:
        print("Can't find the Seeker Server")
        exit(0)
    frame=GetCurrentFrame()
    b=S.py_data_func(frame)
    print(type(b))
    obs = [b[0][0], b[0][1],b[0][2]]
    print(obs)
    return (obs)

'''
if __name__ =='__main__':
    env= ArmEnv()
    for i in range(0,10):
        version = (c_ubyte * 4)()
        GetSdkVersion(version)
        SetVerbosityLevel(1)
        SetErrorMsgHandlerFunc(S.py_msg_func)
        ret = Initialize(b"10.1.1.198", b"10.1.1.198")
        hostInfo = HostInfo()
        GetHostInfo(pointer(hostInfo))

        frame=GetCurrentFrame()
        b=S.py_data_func(frame)
        print(type(b))
        obs = [b[0][0], b[0][1],b[0][2]]
        print(obs)
        #py_data_func_user(frame,None)
        #print(b[0][1])
        #print(f) 
        time.sleep(0.1)
'''
'''
if __name__ =='__main__':
    env= ArmEnv()
    for i in range(0,1):
        get_obs()
'''


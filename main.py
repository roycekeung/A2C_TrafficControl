'''
based on
<<Traffic Light Control Using Deep Policy-Gradient and Value-Function Based Reinforcement Learning>>
but it only handle single intersection => one agent one controller
drawback no green split cycle time constraint with simple A2C method
''' 

import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import argparse
import numpy as np
from collections import deque
import random

import os
import time
import datetime

from modelAgent import Actor, Critic, AgentController
from env import Environment

from config import *


### add later when upto the use of linux vm as server training
# parser = argparse.ArgumentParser()
# parser.add_argument('--gamma', type=float, default=0.99)
# parser.add_argument('--update_interval', type=int, default=5)
# parser.add_argument('--actor_lr', type=float, default=0.0005)
# parser.add_argument('--critic_lr', type=float, default=0.001)
# args = parser.parse_args()


GPU = "NoGPU"
CPU = "Intel(R) Core(TM) i7-8750H"
MACHINE = "LAPTOP-HH4MF1MU"
### set up following config when u have better GPU with CUDA driver to do training
# os.environ["CUDA_VISIBLE_DEVICES"] = 1
config_dict = {
    "GPU_USED" : 0,
    "GPU" : GPU,
    "CPU" : CPU,
    "MACHINE" : MACHINE,
}



tf.keras.backend.set_floatx('float64')
projectname = 'A2C_Discrete_TrafficControl'
trialname = 'A2C_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

### upload training record to wandb server
wandb.init(
    name=trialname, 
    project=projectname,
    config = config_dict
    entity = "Royce")

### for local tensorboard backup
log_dir = "logs/{}".format(trialname)





if __name__ == "__main__":

    ## init env
    env = Environment()
    
    
    ## init Actor Critic model
    feature_shape = (128 , 128, 4)

    agent = AgentController(env, feature_shape=feature_shape, n_actions=2, output_graph= True , log_dir=log_dir )
    agent.learn(max_ep= MAX_EPISODE, max_timestep = MAX_EP_STEPS)
    actor_net = agent.get_actor_model()


    if SAVE_MODEL:
        ### get current file directory path
        folderPath = os.path.join(os.getcwd(), trialname+'.h5')
        ### save trained model
        actor_net.save_model(folderPath)
            
    # ### plot cost but if training in linux vm then ignore this
    if OUTPUT_GRAPH:
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(agent.get_cost_his())), agent.get_cost_his())
        plt.ylabel('Cost')
        plt.xlabel('accumulative steps')
        plt.show()    


import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPool2D, BatchNormalization
import pickle


import numpy as np
from collections import deque

from config import *





## seed
np.random.seed(np.random.randint(1000) if RANDOMSEED else 1)
tf.random.set_seed(np.random.randint(1000) if RANDOMSEED else 1)   ## tf2.0 



class Actor(object):
    ## policy-based
    def __init__(self, feature_shape:tuple , n_actions:int, build_net=True):
        '''
        feature_shape = (whatever x size , whatever y size, 1 or 2(pos and speed ) or x 4  coz frames of history)
        we stacked the last
        four frames of the history and provided them to the system
        as input. So, the input to the network was a 128 × 128 × 4 ;  feature_shape(128 , 128, 4)
        image
        '''
        self.feature_shape = feature_shape
        self.n_actions = n_actions
        if build_net:
            self.model = self.build_net()
            self.optimizer = tf.keras.optimizers.Adam(LR_A)
        else:
            self.model = None

    def build_net(self)->tf.keras.Model:

        model = tf.keras.models.Sequential()
        # inn = Input(shape=(feature_shape))
        '''
        The network consists a stack of two convolutional layers with filters 16 8 × 8 and 32 4 × 4 and strides 4
        and 2  (same to atari<<Playing Atari with Deep Reinforcement Learning>>)
        '''
        model.add(Conv2D(filters = 16, 
                         kernel_size=(8, 8), 
                         strides=(4,4), 
                         padding='valid', 
                         activation='relu', 
                         kernel_initializer=tf.random_normal_initializer(0., .1), 
                         bias_initializer=tf.constant_initializer(0.1), 
                         input_shape=(self.feature_shape) ,
                         name="layer1", 
                         )
                  )
        model.add(Conv2D(filters = 32, 
                         kernel_size=(4, 4), 
                         strides=(2,2), 
                         padding='valid', 
                         activation='relu',
                         kernel_initializer=tf.random_normal_initializer(0., .1), 
                         bias_initializer=tf.constant_initializer(0.1), 
                         name="layer2", 
                         )
                  )
        model.add(Flatten())
        model.add(Dense(256, activation='relu'), kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer3")
        model.add(BatchNormalization())
        model.add(Dense(int(256/4), activation='relu'), kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer4")
        model.add(BatchNormalization())
        model.add(Dense(int(256/4/4), activation='relu'), kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer5")
        model.add(BatchNormalization())
        model.add(Dense(self.n_actions, activation='softmax'))
        return model
    
    def choose_action(self, state):
        reshape_state = np.reshape(state, np.hstack((1, self.feature_shape)))
        probs = self.actor.model.predict(reshape_state )
        
        action = np.random.choice(self.n_actions, p=probs.ravel())  ## or flatten 1d

        return action
        
    def get_loss(self, actions, logits, advantages):
        ### to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
        #  # this is negative log of chosen action; built-in func
        # SPCEloss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # actions = tf.cast(actions, tf.int32)
        # policy_loss = SPCEloss(actions, logits, sample_weight=tf.stop_gradient(advantages))
        
        ### or cross entropy:
        prob = tf.math.reduce_sum(-tf.math.log(logits)*tf.one_hot(actions, self.n_actions), axis=1)  ## a prob of corresponding action taken   
        ## one_hot shape=[None, n_actions] ; prob shape=[None, 1]
        policy_loss = tf.reduce_mean(prob * advantages)  # reward guided loss

        return policy_loss
    
    def train(self, states:np.array, actions:np.array, advantages:np.array):
        with tf.GradientTape(persistent=False, watch_accessed_variables=True) as tape:
            ## watch (trainable tensor) automatically
            ### run the Network in training mode
            logits = self.model(states, training=True)
            policy_loss = self.get_loss(actions, logits, advantages)
        ## cal gradients
        # minimize(-policy_loss) = maximize(exp_v)
        gradients = tape.gradient(target = policy_loss, sources = self.model.trainable_variables)  # trainable_variables = trainable_weights ; sources=[w,b]  
        ## update w b
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))    # trainable_variables = trainable_weights
        return policy_loss
    
    def save_model(self, filedirname):
        # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
        self.model.save(filedirname)

    def load_model(self, filedirname):
        # It can be used to reconstruct the model identically.
        self.model = tf.keras.models.load_model(filedirname)
        return self.model



class Critic(object):
    ## value-based
    def __init__(self, feature_shape:tuple, build_net=True):
        '''
        feature_shape = (whatever x size , whatever y size, 1 or 2(pos and speed ) or x 4  coz frames of history)
        we stacked the last
        four frames of the history and provided them to the system
        as input. So, the input to the network was a 128 × 128 × 4 ;  feature_shape(128 , 128, 4)
        image
        '''
        self.feature_shape = feature_shape

        if build_net:
            self.model = self.build_net()
            self.optimizer = tf.keras.optimizers.Adam(LR_C)
        else:
            self.model = None

    def build_net(self)->tf.keras.Model:
        model = tf.keras.models.Sequential()
        model.add(Conv2D(filters = 16, 
                         kernel_size=(8, 8), 
                         strides=(4,4), 
                         padding='valid', 
                         activation='relu', 
                         kernel_initializer=tf.random_normal_initializer(0., .1), 
                         bias_initializer=tf.constant_initializer(0.1), 
                         input_shape=(self.feature_shape) ,
                         name="layer1"
                         )
                  )
        model.add(Conv2D(filters = 32, 
                         kernel_size=(4, 4), 
                         strides=(2,2), 
                         padding='valid', 
                         activation='relu',
                         kernel_initializer=tf.random_normal_initializer(0., .1), 
                         bias_initializer=tf.constant_initializer(0.1), 
                         name="layer2"
                         )
                  )
        model.add(Flatten())
        model.add(Dense(256, activation='relu'), kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer3")
        model.add(BatchNormalization())
        model.add(Dense(int(256/4), activation='relu'), kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer4")
        model.add(BatchNormalization())
        model.add(Dense(int(256/4/4), activation='relu'), kernel_initializer=tf.random_normal_initializer(0., .1), bias_initializer=tf.constant_initializer(0.1), name="layer5")
        model.add(BatchNormalization())
        model.add(Dense(1, activation='linear'))  ## score of this state
        return model
    
    def get_td_target(self, reward, next_state, is_done , _gamma = GAMMA):
        if is_done :
            return reward
        V_next = self.model.predict( np.reshape(next_state, [1, self.state_dim]) )
        td_target = np.reshape(reward + _gamma * np.max(V_next) , [1, 1])
        return td_target
    
    def get_advantage(self, td_targets, baselines):
        ## advantage function may not always be the same as the TD Error 
        advantage = td_targets - baselines
        return advantage

    def get_loss(self, V_eval, td_targets):
        ## loss = square(y_true - y_pred)  # y_true = td_target = reward + GAMMA * V_next ; y_pred = V_eval
        mse = tf.keras.losses.MeanSquaredError()  
        V_loss = mse(td_targets, V_eval)
        return V_loss   ##  # TD_error = (r+gamma*V_next) - V_eval

    def train(self, state:np.array, td_target:np.array):
        with tf.GradientTape(persistent=False,watch_accessed_variables=True) as tape:
            ### run the Network in training mode
            V_eval = self.model(state, training=True)
            
            V_loss = self.get_loss(V_eval, tf.stop_gradient(td_target))  # td_target = reward + GAMMA * V_next ; v_pred = V_eval
        ## cal gradients
        # minimize(V_loss) 
        grads = tape.gradient(V_loss, self.model.trainable_variables)    ### self.model.trainable_variables = weights and bias
        ## update w b
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return V_eval, V_loss

    def learn(self, state, reward, next_state, is_done ):
        ### single value td_target
        td_target = self.get_td_target( reward, next_state, is_done )
        
        V_eval, V_loss = self.train( state, td_target)
        
        ### single value advantage
        advantage = self.get_advantage( td_target, V_eval.numpy() )

        return advantage
        
    def save_model(self, filedirname):
        # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
        self.model.save(filedirname)

    def load_model(self, filedirname):
        # It can be used to reconstruct the model identically.
        self.model = tf.keras.models.load_model(filedirname)
        return self.model
    
    
class AgentController(object):
    ## a controller, control single intersection
    def __init__(self, env, feature_shape, n_actions, output_graph = OUTPUT_GRAPH, log_dir='logs/', build_net=True ):
        self.env = env
        
        ## init Actor Critic model
        self.actor = Actor(feature_shape, n_actions, build_net = build_net)
        self.critic = Critic(feature_shape, build_net = build_net)
        
        ### cost amonst the whole training 
        self.cost_his = []
        
        if output_graph:
            # create the file writer object
            self.writer = tf.summary.create_file_writer(log_dir = log_dir)
        
    def learn(self, max_ep= MAX_EPISODE, max_timestep = MAX_EP_STEPS ):
        for ep in range(max_ep):
            state = self.env.reset()
            t = 0
            reward_tracker = []
            while True:
                action = self.actor.choose_action(state)

                next_state, reward, is_done  = self.env.step(action)
                reward_tracker.append(reward)
                
                ### learn in each steps; could be multiple steps in each episode and the max step in each episode is MAX_EP_STEPS
                td_error = self.critic.learn(state, reward, next_state, is_done )
                policy_loss = self.actor.train(state, action, td_error)

                
                if OUTPUT_GRAPH:
                    ### write the loss value
                    with self.writer.as_default():
                        ### $ tensorboard --logdir=logs
                        tf.summary.scalar('policy_loss', policy_loss, step=t)
                        self.cost_his.append(policy_loss)

                if is_done  or t >= max_timestep:
                    ep_reward_sum = sum(reward_tracker)

                    if 'episode_reward' not in globals():
                        episode_reward = ep_reward_sum
                    else:
                        ### consider previous episode reward in; ratio self defined
                        episode_reward = episode_reward * 0.95 + ep_reward_sum * 0.05
                    
                    print("episode:{episode}  reward:{reward}".format(episode = ep, reward= episode_reward) )
                    wandb.log({'Reward': episode_reward})
                    break
                
                ### swap observation
                state = next_state
                t+=1
            
    def get_actor_model(self):
        return self.actor 
    
    def get_critic_model(self):
        return self.critic 
    
    def get_cost_his(self):
        return self.cost_his 
        
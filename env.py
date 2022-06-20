    
from config import *

class Environment(object):
    def __init__(self):
        '''
        either vissim, SUMO or CTM
        '''
    
    def reset(self):
        '''
        feature_shape = (whatever x size , whatever y size, 1 or 2(pos and speed ) or x 4  coz frames of history)
        we stacked the last
        four frames of the history and provided them to the system
        as input. So, the input to the network was a 128 × 128 × 4 ;  feature_shape(128 , 128, 4)
        image
        '''
        return state
    
    def step(self, action):

        ## action
        ## in <<Traffic Light Control Using Deep Policy-Gradient and Value-Function Based Reinforcement Learning>>
         '''
         eg.
         discrete actions
         action_set= 
         To control traffic signal phases, we define a set of possible
         actions A = { North/South Green (NSG), East/West Green
         (EWG)}. NSG allows vehicles to pass from North to South
         and vice versa, and also indicates the vehicles on East/West
         route should stop and not proceed through the intersection.
         EWG allows vehicles to pass from East to West and vice
         versa, and implies the vehicles on North/South route should
         stop and not proceed through the intersection
         so n_actions = 2; 
         could either be 
         [0]=>North/South Green (NSG) 
         or 
         [1]=>East/West Green(EWG)

         '''
        '''
        possible def
        1. possible actions, action space is defined by how to update the duration of every phase in the next cycle  
           could be minus or plus certain sec on a phase (adaptive); phse sequence kept the same 
           <<A Deep Reinforcement Learning Network for Traffic Light Cycle Control>>
        2. number of phase, account for yellow, red and min green; but sequence of actions vary (phase)
        selected action means that of phase will be extend by certaian time step
        <<An Agent-Based Learning Towards Decentralized and Coordinated Traffic Signal Control>>
        3. num actions = 2; 1= change phase ; 0= no change
            <<Deep Reinforcement Learning based Traffic Signal Optimization for Multiple Intersections in ITS>>

        '''

        
        ## state
        '''
        possible def of state
        1. matrix vehicles position ( + speed)  <<A Deep Reinforcement Learning Network for Traffic Light Cycle Control>>
        2. num of waiting veh or waiting que length  <<Design of reinforcement learning parameters for seamless application of adaptive traffic signal contro>>
        3. Arrival of vehicles to the current green direction and Queue Length at red directions 
        <<An Agent-Based Learning Towards Decentralized and Coordinated Traffic Signal Contro>>
        4. the maximum queue length associated with each phase
        <<An Agent-Based Learning Towards Decentralized and Coordinated Traffic Signal Contro>>
        5. cumulative delay for phase i is the summation of the cumulative delay of all the vehicles that are travelling on the L(i).
        <<An Agent-Based Learning Towards Decentralized and Coordinated Traffic Signal Contro>>
        6. distance between curent position of vehicle to intersection
        <<Deep Reinforcement Learning based Traffic Signal Optimization for Multiple Intersections in ITS>>
        7. last traffic signal in a lane (like 1 green, 0 red(no amber red coz action space cant represent it as well)
        <<Deep Reinforcement Learning based Traffic Signal Optimization for Multiple Intersections in ITS>>
        8. matrix for traffic states in a single intersection, dont care about actual geometry
          lanes r placed horizontally and stack up together into a state input matrix
          <<Cooperative Control for Multi-Intersection Traffic Signal Based on Deep Reinforcement Learning and Imitation Learning>>
        '''
        
        ## reward 
        ## in <<Traffic Light Control Using Deep Policy-Gradient and Value-Function Based Reinforcement Learning>>
        ## reward = difference between the total cumulative delays of two consecutive actions (or t+1 , t) 
        
        '''
        possible def of reward
        1. cumulative waiting time differece between two cycle (or t+1 , t)    
           <<A Deep Reinforcement Learning Network for Traffic Light Cycle Control>>
        2. difference between the total cumulative delays of two consecutive actions (or t+1 , t)   
           <<Traffic Light Control Using Deep Policy-Gradient and Value-Function Based Reinforcement Learning>>
        3. queue length differece between two cycle (or t+1 , t)
        4. sum num of passing veh in outgoing lanes / sum num of stopped veh in incoming lanes   
           <<Traffic Signal Optimization for Multiple Intersections Based on Reinforcement Learning>
        '''
        
        ## is_done True = 1 ; False =0
        '''
        possible def of is_done
        1. sim ends
        '''
        
        return next_state, reward, is_done
    
    
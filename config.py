# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 1050    ## i presume as epoch coz no batch
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 3600   # maximum time step in one episode
# RENDER = False  # rendering wastes time
GAMMA = 0.99     # reward discount in TD error
LR_A = 0.00001    # learning rate for actor
# we need a good teacher, so the teacher should learn faster than the actor
## so critic will learn faster and more aggresively jump when update theta
LR_C = 0.0001     # learning rate for critic   
EPOCH = 1050
# EPISODES = 10
RANDOMSEED = True
SAVE_MODEL = True


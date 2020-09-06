MODE = 'separate'
# cooperative combines actions for training
# partial uses independent actors/shared critic
# independent each arm learns completely independent with only shared reward/memory

EPISODE = 1
TIMESTEP = 1

# lists for dynamic chart generation, also stored in csv files.
STEP_LIST = [["episode", "step", "reward", "step_distance_moved", "step_distance_target", "solved", "time_elapsed"]]
ACTOR_LIST = [["episode", "step", "actor_1_loss", "actor_2_loss"]]
CRITIC_LIST = [["episode", "step,critic_1_loss", "critic_2_loss"]]
ERROR_LIST = [["episode", "step", "error"]]

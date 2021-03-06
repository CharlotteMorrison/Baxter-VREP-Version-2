from reports import Reports
import torch

# testing constants
MAX_EPISODE = 5000  # stop the training early and test the results

# don't move this- it creates circular dependencies.
report = Reports()

# flags
set_seed = True

# Program run constants
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: {}'.format(DEVICE))
num_frames = 300000
VIDEO_INTERVAL = 100  # change to 1 to record all videos
NUM_FRAMES_STACKED = 4
XYZ_GOAL = [.2, .1, 1.2]  # More specific goal numbers [0.231, 0.105, 1.261]
SOLVED_DISTANCE = 1.0  #
WRITE_TO_FILE = True
REPORT_INTERVAL = 1  # write all the reports
SIZE = (512, 512)
LR = 1e-2            # Other LR values, was 1e-3 or 3e-4

# prioritized replay params
PRIORITY = False    # True runs with priority replay
ALPHA = .6          # alpha param for priority replay buffer
BETA = .4           # initial value of beta
BETA_ITERS = None   # number of iterations over which beta will be annealed from initial value
EPS = 1e-6          # epsilon to add to the TD errors when updating priorities
BETA_SCHED = None   # do you want manually scheduled beta

# TD3 hyperparameters from addressing function approx. err paper
EXPLORATION = 5000000
OBSERVATION = 10000
EXPLORE_NOISE = 0.1
BATCH_SIZE = 1024        # paper used 1024, may be too large.
BUFFER_SIZE = 1000000   # shrunk from 1,000,000 to 200,000
GAMMA = 0.95            # discount was .99
TAU = 0.005
POLICY_NOISE = 0.01  # adjusted from .2, due to scale of movement
NOISE_CLIP = 0.05    # adjusted from .5, due to scale of movement
POLICY_FREQ = 2

# environment parameters
STATE_DIM = torch.empty(1, 14)
MAX_ACTION = 0.2
MIN_ACTION = -0.2
ACTION_DIM = 14






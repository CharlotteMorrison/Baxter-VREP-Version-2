import torch
from datetime import date

# testing constants
MAX_EPISODE = 50  # stop the training early and test the results

# flags
set_seed = True
MODE = 'independent'  # cooperative combines actions for training, independent uses two sep. actions

DEFAULT_NAME = "Baxter_TD3"
model_file = "td3/results/models/" + DEFAULT_NAME + "_model" + date.today().strftime("%b-%d-%Y") + ".pth"
FILE_NAME = "td3/results/" + DEFAULT_NAME + "_results"
EPISODE_PLOT_NAME = "td3/results/plots/" + DEFAULT_NAME + "_reward_episode.png"
ALL_PLOT_NAME = "td3/results/plots/" + DEFAULT_NAME + "_reward_all.png"
END_PLOT_NAME = "td3/results/plots/" + DEFAULT_NAME + "_reward_final.png"
AVG_10_PLOT_NAME = "td3/results/plots/" + DEFAULT_NAME + "_last_10_average.png"

# Program run constants
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_frames = 300000
VIDEO_INTERVAL = 1  # change to 1 to record all videos
NUM_FRAMES_STACKED = 4
XYZ_GOAL = [0.3, 0.0, 1.33]
SOLVED_DISTANCE = 1.0  #
WRITE_TO_FILE = True
REPORT_INTERVAL = 1  # write all the reports
SIZE = (512, 512)

# prioritized replay params
PRIORITY = True     # True runs with priority replay
ALPHA = .6          # alpha param for priority replay buffer
BETA = .4           # initial value of beta
BETA_ITERS = None   # number of iterations over which beta will be annealed from initial value
EPS = 1e-6          # epsilon to add to the TD errors when updating priorities
BETA_SCHED = None

# TD3 hyperparameters from addressing function approx. err paper
EXPLORATION = 5000000
OBSERVATION = 10000
EXPLORE_NOISE = 0.1
REWARD_THRESH = 1.10
BATCH_SIZE = 100
BUFFER_SIZE = 2200  # shrunk from 1,000,000
GAMMA = 0.99  # discount
TAU = 0.005
POLICY_NOISE = 0.01  # adjusted from .2, due to scale of movement
NOISE_CLIP = 0.05    # adjusted from .5, due to scale of movement
POLICY_FREQ = 2

# environment parameters
STATE_DIM = torch.empty(1, 14)
MAX_ACTION = 0.2
MIN_ACTION = -0.2
ACTION_DIM = 14






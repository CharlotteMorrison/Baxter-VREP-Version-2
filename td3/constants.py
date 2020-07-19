import torch
from datetime import datetime
from td3.reports import report

# testing constants
MAX_EPISODE = 1000  # stop the training early and test the results

# flags
set_seed = True
MODE = 'separate'  # cooperative combines actions for training, independent uses independent actors/shared critic
# separate each arm learns completely independent with only shared reward/memory

timestr = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
DEFAULT_NAME = "Baxter_TD3_" + MODE
model_file = "td3/results/models/" + DEFAULT_NAME + "_model_" + timestr + ".pth"
EPISODE_PLOT_NAME = "td3/results/plots/" + DEFAULT_NAME + "_reward_episode_" + timestr + ".png"
ALL_PLOT_NAME = "td3/results/plots/" + DEFAULT_NAME + "_reward_all_" + timestr + ".png"
END_PLOT_NAME = "td3/results/plots/" + DEFAULT_NAME + "_reward_final_" + timestr + ".png"
AVG_10_PLOT_NAME = "td3/results/plots/" + DEFAULT_NAME + "_last_10_average_" + timestr + ".png"
EPISODE_LENGTH_NAME = "td3/results/plots/" + DEFAULT_NAME + "_episode_length_" + timestr + ".png"

TD3_REPORT = report()

# Program run constants
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device: {}'.format(DEVICE))
num_frames = 300000
VIDEO_INTERVAL = 1  # change to 1 to record all videos
NUM_FRAMES_STACKED = 4
XYZ_GOAL = [.2, .1, 1.2]  # More specific goal numbers [0.231, 0.105, 1.261]
SOLVED_DISTANCE = 1.0  #
WRITE_TO_FILE = True
REPORT_INTERVAL = 1  # write all the reports
SIZE = (512, 512)

# prioritized replay params
PRIORITY = True    # True runs with priority replay
ALPHA = .6          # alpha param for priority replay buffer
BETA = .4           # initial value of beta
BETA_ITERS = None   # number of iterations over which beta will be annealed from initial value
EPS = 1e-6          # epsilon to add to the TD errors when updating priorities
BETA_SCHED = None   # do you want manually scheduled beta

# TD3 hyperparameters from addressing function approx. err paper
EXPLORATION = 5000000
OBSERVATION = 10000
EXPLORE_NOISE = 0.1
REWARD_THRESH = 1.10
BATCH_SIZE = 100
BUFFER_SIZE = 20000  # shrunk from 1,000,000
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






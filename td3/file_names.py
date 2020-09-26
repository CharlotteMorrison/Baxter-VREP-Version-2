import set_mode
from datetime import datetime

# create the file names fo the run
timestr = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")

# file for storing all file names
all_file_names = "td3/results/reports/{}_report_names_{}.csv".format(set_mode.MODE, timestr)

# files for storing the run data
STEP_REPORT_NAME = "td3/results/reports/{}_step_report_{}.csv".format(set_mode.MODE, timestr)
ACTOR_REPORT_NAME = "td3/results/reports/{}_actor_loss_report_{}.csv".format(set_mode.MODE, timestr)
CRITIC_REPORT_NAME = "td3/results/reports/{}_critic_loss_report_{}.csv".format(set_mode.MODE, timestr)
ERROR_REPORT_NAME = "td3/results/reports/{}_error_report_{}.csv".format(set_mode.MODE, timestr)

# open the files for program use ("a+" also works, just an fyi if you need it later)
STEP_REPORT = open(STEP_REPORT_NAME, "w+")
ACTOR_REPORT = open(ACTOR_REPORT_NAME, "w+")
CRITIC_REPORT = open(CRITIC_REPORT_NAME, "w+")
ERROR_REPORT = open(ERROR_REPORT_NAME, "w+")

# model save file
ACTOR_1 = 'td3/models/{}/{}_{}.pth'.format(set_mode.MODE, 'actor_1', timestr)
ACTOR_2 = 'td3/models/{}/{}_{}.pth'.format(set_mode.MODE, 'actor_2', timestr)
CRITIC_1 = 'td3/models/{}/{}_{}.pth'.format(set_mode.MODE, 'critic_1', timestr)
CRITIC_2 = 'td3/models/{}/{}_{}.pth'.format(set_mode.MODE, 'critic_2', timestr)
MODEL_DIRECTORY = "td3/models/{}".format(set_mode.MODE)

# graph save files based on step report
AVG_REWARD_EPISODE = "td3/results/plots/{}_avg_reward_episode_{}.png".format(set_mode.MODE, timestr)
AVG_ROLLING_REWARD = "td3/results/plots/{}_avg_rolling_reward_{}.png".format(set_mode.MODE, timestr)
EPISODE_LENGTH = "td3/results/plots/{}_episode_length_{}.png".format(set_mode.MODE, timestr)
TOTAL_EPISODE_DISTANCE = "td3/results/plots/{}_total_episode_distance_{}.png".format(set_mode.MODE, timestr)
MIN_DISTANCE_TO_TARGET = "td3/results/plots/{}_min_distance_to_target_{}.png".format(set_mode.MODE, timestr)

# graph save files for other reports
ACTOR_LOSS_PLOT = "td3/results/plots/{}_actor_loss_plot_{}.png".format(set_mode.MODE, timestr)
CRITIC_LOSS_PLOT = "td3/results/plots/{}_critic_loss_plot_{}.png".format(set_mode.MODE, timestr)

# save files for videos
EPISODE_VIDEO = "td3/videos/{}_video_{}.png".format(set_mode.MODE, timestr)
EPISODE_VIDEO_SOLVED = "td3videos/{}_video_{}_solved.png".format(set_mode.MODE, timestr)

# create a list of file names
names = [STEP_REPORT_NAME, ACTOR_REPORT_NAME, CRITIC_REPORT_NAME, ERROR_REPORT_NAME,
         ACTOR_1, ACTOR_2, CRITIC_1, CRITIC_2, EPISODE_VIDEO, EPISODE_VIDEO_SOLVED,
         AVG_REWARD_EPISODE, AVG_ROLLING_REWARD, ACTOR_LOSS_PLOT, CRITIC_LOSS_PLOT,
         EPISODE_LENGTH, TOTAL_EPISODE_DISTANCE, MIN_DISTANCE_TO_TARGET]

# save the file names for each run
report_names = open(all_file_names, "w+")
for name in names:
    report_names.write(name + "\n")
report_names.close()


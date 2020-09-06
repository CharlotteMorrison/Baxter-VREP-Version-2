import set_mode
from datetime import datetime

# create the file names fo the run
timestr = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")

# file for storing all file names
all_file_names = "results/reports/{}_report_names_{}.csv".format(set_mode.MODE, timestr)

# files for storing the run data
STEP_REPORT_NAME = "results/reports/{}_step_report_{}.csv".format(set_mode.MODE, timestr)
ACTOR_REPORT_NAME = "results/reports/{}_actor_loss_report_{}.csv".format(set_mode.MODE, timestr)
CRITIC_REPORT_NAME = "results/reports/{}_critic_loss_report_{}.csv".format(set_mode.MODE, timestr)
ERROR_REPORT_NAME = "results/reports/{}_error_report_{}.csv".format(set_mode.MODE, timestr)

# open the files for program use ("a+" also works, just an fyi if you need it later)
STEP_REPORT = open(STEP_REPORT_NAME, "w+")
ACTOR_REPORT = open(ACTOR_REPORT_NAME, "w+")
CRITIC_REPORT = open(CRITIC_REPORT_NAME, "w+")
ERROR_REPORT = open(ERROR_REPORT_NAME, "w+")

# model save file
model_file_name = "models/{}/{}_model_{}.pth".format(set_mode.MODE, set_mode.MODE, timestr)

# graph save files based on step report
ALL_TIMESTEP_REWARD = "results/plots/{}_all_timestep_reward_{}.png".format(set_mode.MODE, timestr)
AVG_REWARD_EPISODE = "results/plots/{}_avg_reward_episode_{}.png".format(set_mode.MODE, timestr)
AVG_ROLLING_REWARD_10 = "results/plots/{}_avg_rolling_reward_10_{}.png".format(set_mode.MODE, timestr)
AVG_ROLLING_REWARD_100 = "results/plots/{}_avg_rolling_reward_100_{}.png".format(set_mode.MODE, timestr)
EPISODE_LENGTH = "results/plots/{}_episode_length_{}.png".format(set_mode.MODE, timestr)
TOTAL_EPISODE_DISTANCE = "results/plots/{}_total_episode_distance_{}.png".format(set_mode.MODE, timestr)
MIN_DISTANCE_TO_TARGET = "results/plots/{}_min_distance_to_target_{}.png".format(set_mode.MODE, timestr)

# graph save files for other reports
# TODO implement other report files (actor, critic, and error)


# create a list of file names
names = [STEP_REPORT_NAME, ACTOR_REPORT_NAME, CRITIC_REPORT_NAME, ERROR_REPORT_NAME, model_file_name,
         ALL_TIMESTEP_REWARD, AVG_REWARD_EPISODE, AVG_ROLLING_REWARD_10, AVG_ROLLING_REWARD_100,
         EPISODE_LENGTH, TOTAL_EPISODE_DISTANCE, MIN_DISTANCE_TO_TARGET]

# save the file names for each run
report_names = open(all_file_names, "w+")
for name in names:
    report_names.write(name + "\n")
report_names.close()


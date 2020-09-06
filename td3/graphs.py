import file_names as names
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class Graphs:
    def __init__(self):
        # setup styling for graphs
        sns.set()
        sns.set_style('whitegrid')

    def all_timestep_reward(self):
        # reward at each timestep
        # load the file-->
        # names.ALL_TIMESTEP_REWARD
        step_file = pd.read_csv(names.STEP_REPORT_NAME)
        print(step_file.columns)
        # TODO make this work whenever called, not just at end.
        pass

    def avg_reward_episode(self):
        # average reward for each episode
        # names.AVG_REWARD_EPISODE
        pass

    def rolling_average_reward(self):
        # uses a 10 and 100 rolling window for average reward
        # names.AVG_ROLLING_REWARD_10
        # names.AVG_ROLLING_REWARD_100
        pass

    def episode_length(self):
        # counts the number of steps in each episode
        # names.EPISODE_LENGTH
        pass

    def total_episode_distance(self):
        # sums the distance traveled by the object in each episode
        # names.TOTAL_EPISODE_DISTANCE
        pass

    def min_distance_to_target(self):
        # finds the minimum distance the object is from the target in the episode
        # names.MIN_DISTANCE_TO_TARGET
        pass

    # TODO implement other reports (actor, critic, and error)

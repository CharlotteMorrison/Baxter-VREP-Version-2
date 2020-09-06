import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import globals
import file_names as names


class Graphs:
    def __init__(self):
        # setup styling for graphs
        sns.set()
        sns.set_style('whitegrid')
        self.datafile = pd.DataFrame()
        self.episode_df = pd.DataFrame()

    def update_step_list_graphs(self):
        # load the datafile
        headers = ["episode", "step", "reward", "step_distance_moved", "step_distance_target", "solved", "time_elapsed"]
        self.datafile = pd.DataFrame(globals.STEP_LIST, columns=headers)
        # datafile group by episode
        self.episode_df = self.datafile.groupby(['episode']).mean()
        # run the graphs
        self.all_timestep_reward()
        self.avg_reward_episode()
        # include a rolling average for 10, 100, and 100 episodes

        # episode length should be a scatterplot maybe

    def all_timestep_reward(self):
        # reward at each timestep
        plt.plot(self.datafile['reward'], label='Timestep Reward')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.savefig(names.ALL_TIMESTEP_REWARD)
        plt.show()

    def avg_reward_episode(self):
        # average reward for each episode
        plt.plot(self.episode_df['reward'], label='Timestep Reward')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.savefig(names.AVG_REWARD_EPISODE)
        plt.show()

    def rolling_average_reward(self, window):
        # rolling average of the reward
        pass

    def episode_length(self):
        # counts the number of steps in each episode
        pass

    def total_episode_distance(self):
        # sums the distance traveled by the object in each episode
        pass

    def min_distance_to_target(self):
        # finds the minimum distance the object is from the target in the episode
        pass

    # TODO implement other reports (actor, critic, and error)

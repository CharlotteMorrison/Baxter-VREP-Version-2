import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import globals
import file_names as names
import set_mode


class Graphs:
    def __init__(self):
        # setup styling for graphs
        sns.set()
        sns.set_style('whitegrid')
        plt.style.use('seaborn')
        self.datafile = pd.DataFrame()
        self.episode_df = pd.DataFrame()
        self.actor_df = pd.DataFrame()
        self.critic_df = pd.DataFrame()
        self.evaluate_df = pd.DataFrame()
        self.show_graphs = False

    def update_step_list_graphs(self):
        # load the datafile
        plt.close("all")
        headers = ["episode", "step", "reward", "step_distance_moved", "step_distance_target", "solved", "time_elapsed"]
        self.datafile = pd.DataFrame(globals.STEP_LIST, columns=headers)
        # datafile group by episode
        self.episode_df = self.datafile.groupby(['episode']).mean()

        # actor/critic datafiles

        self.actor_df = pd.DataFrame(globals.ACTOR_LIST, columns=['episode', 'step', 'actor_1_loss', 'actor_2_loss'])
        self.critic_df = pd.DataFrame(globals.CRITIC_LIST, columns=['episode', 'step', 'critic_1_loss', 'critic_2_loss'])

        # graph actor/critic
        self.avg_actor_loss()
        self.avg_critic_loss()

        # run the graphs
        self.avg_reward_episode()

        # include a rolling average for 10, 100, and 1000 episodes
        self.rolling_average_reward()

        self.episode_length()
        self.total_episode_distance()
        self.min_distance_to_target()

    def avg_reward_episode(self):
        # average reward for each episode
        plt.plot(self.episode_df['reward'], label='Average Episode Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(names.AVG_REWARD_EPISODE)
        if self.show_graphs:
            plt.show()
        plt.close()

    def rolling_average_reward(self):
        # rolling average of the reward
        rolling_avg_10 = self.episode_df['reward'].rolling(10).mean()
        rolling_avg_100 = self.episode_df['reward'].rolling(100).mean()
        rolling_avg_1000 = self.episode_df['reward'].rolling(1000).mean()
        fig, ax = plt.subplots(4, figsize=(12, 12), sharey=True)

        ax[0].plot(rolling_avg_10, label='window=10')
        next(ax[1]._get_lines.prop_cycler)
        ax[1].plot(rolling_avg_100, label='window=100')
        next(ax[2]._get_lines.prop_cycler)
        next(ax[2]._get_lines.prop_cycler)
        ax[2].plot(rolling_avg_1000, label='window=1000')
        ax[3].plot(rolling_avg_10, label='window=10')
        ax[3].plot(rolling_avg_100, label='window=100')
        ax[3].plot(rolling_avg_1000, label='window=1000')

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
        ax[3].legend()

        plt.savefig(names.AVG_ROLLING_REWARD)
        if self.show_graphs:
            plt.show()
        plt.close()

    def episode_length(self):
        # counts the number of steps in each episode
        plt.plot(self.datafile.groupby(['episode']).count()['step'], label='Length of Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps In Episode')
        plt.legend()
        plt.savefig(names.EPISODE_LENGTH)
        if self.show_graphs:
            plt.show()
        plt.close()

    def total_episode_distance(self):
        # sums the distance traveled by the object in each episode
        dist_datafile = self.datafile.groupby(['episode']).sum()
        plt.plot(dist_datafile['step_distance_moved'], label='Episode Distance Moved')
        plt.xlabel('Episode')
        plt.ylabel('Total Distance Moved')
        plt.legend()
        plt.savefig(names.TOTAL_EPISODE_DISTANCE)
        if self.show_graphs:
            plt.show()
        plt.close()

    def min_distance_to_target(self):
        # finds the minimum distance the object is from the target in the episode
        min_datafile = self.datafile.groupby(['episode']).min()
        plt.plot(min_datafile['step_distance_target'], label='Min Episode Distance to Target')
        plt.xlabel('Episode')
        plt.ylabel('Distance to Target')
        plt.legend()
        plt.savefig(names.MIN_DISTANCE_TO_TARGET)
        if self.show_graphs:
            plt.show()
        plt.close()

    def avg_actor_loss(self):
        plt.plot(self.actor_df.groupby(['episode']).mean()['actor_1_loss'], label="Actor 1 Loss")
        if set_mode.MODE != 'cooperative':
            plt.plot(self.actor_df.groupby(['episode']).mean()['actor_2_loss'], label="Actor 2 Loss")
        plt.xlabel('Episode')
        plt.ylabel('Actor Loss')
        plt.legend()
        plt.savefig(names.ACTOR_LOSS_PLOT)
        if self.show_graphs:
            plt.show()
        plt.close()

    def avg_critic_loss(self):
        plt.plot(self.critic_df.groupby(['episode']).mean()['critic_1_loss'], label="Critic 1 Loss")
        if set_mode.MODE == 'independent':
            plt.plot(self.critic_df.groupby(['episode']).mean()['critic_2_loss'], label="Critic 2 Loss")
        plt.xlabel('Episode')
        plt.ylabel('Critic Loss')
        plt.legend()
        plt.savefig(names.CRITIC_LOSS_PLOT)
        if self.show_graphs:
            plt.show()
        plt.close()

    # TODO implement error plot
    def error_plot(self):
        pass

    def avg_evaluation_episode_reward(self):
        # this is only called from the evaluate
        self.evaluate_df = pd.DataFrame(globals.EVALUATE_LIST, columns=['episode', 'step', 'reward'])
        plt.plot(self.evaluate_df.groupby(['episode']).mean()['reward'], label='Reward')
        plt.xlabel('Episode')
        plt.ylabel('Evaluate Reward')
        plt.legend()
        plt.savefig(names.EVALUATION_GRAPH)
        if self.show_graphs:
            plt.show()
        plt.close()

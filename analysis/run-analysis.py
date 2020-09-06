import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# TODO generalize this so it can take n- inputs for multiple runs.
def ra_plot(run_1, run_2, run_3, run_4, window=10):
    run_avg_1 = run_1['average'].rolling(window).mean()
    run_avg_2 = run_2['average'].rolling(window).mean()
    run_avg_3 = run_3['average'].rolling(window).mean()
    run_avg_4 = run_4['average'].rolling(window).mean()
    run_avg_combo = pd.concat([run_avg_1,
                               run_avg_2.reindex(run_avg_1.index),
                               run_avg_3.reindex(run_avg_1.index),
                               run_avg_4 .reindex(run_avg_1.index)], axis=1)
    run_avg_combo.set_axis(['average_1', 'average_2', 'average_3', 'average_4'], axis=1, inplace=True)
    run_avg_combo['min'] = run_avg_combo.min(axis=1)
    run_avg_combo['max'] = run_avg_combo.max(axis=1)
    run_avg_combo['average'] = run_avg_combo.mean(axis=1)

    plt.plot(run_avg_combo['average_1'], label='Run 1')
    plt.plot(run_avg_combo['average_2'], label='Run 2')
    plt.plot(run_avg_combo['average_3'], label='Run 3')
    plt.plot(run_avg_combo['average_4'], label='Run 4')
    plt.plot(run_avg_combo['average'], color='black', label='Average', linestyle='dashed')
    plt.fill_between(run_avg_combo.index, run_avg_combo['min'], run_avg_combo['max'], alpha=0.5, label='Range')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.title('Total Average Episode Reward')
    plt.legend()
    plt.savefig('graphs/total-average-episode-reward-window{}-new.png'.format(window))
    plt.show()


def sd_plot(run_1, run_2, run_3, run_4):
    run_1 = run_1.drop([' distance_to_target'], axis=1)
    run_combo = pd.concat([run_1,
                           run_2['average'].reindex(run_1.index),
                           run_3['average'].reindex(run_1.index),
                           run_4['average'].reindex(run_1.index)], axis=1)
    run_combo.set_axis(['average_1', 'average_2', 'average_3', 'average_4'], axis=1, inplace=True)

    run_combo['min'] = run_combo.min(axis=1)
    run_combo['max'] = run_combo.max(axis=1)
    run_combo['average'] = run_combo.mean(axis=1)

    plt.plot(run_combo['average_1'], label='Run 1')
    plt.plot(run_combo['average_2'], label='Run 2')
    plt.plot(run_combo['average_3'], label='Run 3')
    plt.plot(run_combo['average_4'], label='Run 4')
    plt.plot(run_combo['average'], color='black', label='Average', linestyle='dashed')
    plt.fill_between(run_combo.index, run_combo['min'], run_combo['max'], alpha=0.5, label='Range')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.title('Total Average Episode Reward')
    plt.legend(loc='lower right')
    plt.savefig('graphs/total-average-episode-reward-minmax.png')
    plt.show()


def sd_plot_2(run_1, run_2, run_3, run_4):
    run_1 = run_1.drop(['episode'], axis=1)
    run_combo = pd.concat([run_1,
                           run_2['average'].reindex(run_1.index),
                           run_3['average'].reindex(run_1.index),
                           run_4['average'].reindex(run_1.index)], axis=1)
    run_combo.set_axis(['average_1', 'average_2', 'average_3', 'average_4'], axis=1, inplace=True)

    run_combo['min'] = run_combo.min(axis=1)
    run_combo['max'] = run_combo.max(axis=1)
    run_combo['average'] = run_combo.mean(axis=1)

    plt.plot(run_combo['average_1'], label='Run 1')
    plt.plot(run_combo['average_2'], label='Run 2')
    plt.plot(run_combo['average_3'], label='Run 3')
    plt.plot(run_combo['average_4'], label='Run 4')
    plt.plot(run_combo['average'], color='black', label='Average', linestyle='dashed')
    plt.fill_between(run_combo.index, run_combo['min'], run_combo['max'], alpha=0.5, label='Range')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    # plt.title('Total Average Episode Reward')
    plt.legend(loc='lower right')
    plt.savefig('graphs/total-average-episode-reward-minmax-new.png')
    plt.show()


if __name__ == "__main__":
    sns.set()
    sns.set_style('whitegrid')

    run1 = pd.read_csv('data-files/run1a.csv')
    run2 = pd.read_csv('data-files/run2a.csv')
    run3 = pd.read_csv('data-files/run3a.csv')
    run4 = pd.read_csv('data-files/run4a.csv')
    runs = [run1, run2, run3, run4]

    run1 = run1.drop(['Steps_In_Episode',
                      'Total_Steps',
                      'Mean_Episode_Reward',
                      'Mean_Reward_All',
                      ' Memory_used',
                      'Time_Elapsed',
                      'Solved',
                      'Solved.1'], axis=1)
    run1.set_axis(['episode', 'average'], axis=1, inplace=True)
    run2 = run2.drop(['Steps_In_Episode',
                      'Total_Steps',
                      'Mean_Episode_Reward',
                      'Mean_Reward_All',
                      ' Memory_used',
                      'Time_Elapsed',
                      'Solved',
                      'Solved.1'], axis=1)
    run2.set_axis(['episode', 'average'], axis=1, inplace=True)
    run3 = run3.drop(['Steps_In_Episode',
                      'Total_Steps',
                      'Mean_Episode_Reward',
                      'Mean_Reward_All',
                      ' Memory_used',
                      'Time_Elapsed',
                      'Solved',
                      'Solved.1'], axis=1)
    run3.set_axis(['episode', 'average'], axis=1, inplace=True)
    run4 = run4.drop(['Steps_In_Episode',
                      'Total_Steps',
                      'Mean_Episode_Reward',
                      'Mean_Reward_All',
                      ' Memory_used',
                      'Time_Elapsed',
                      'Solved',
                      'Solved.1'], axis=1)
    run4.set_axis(['episode', 'average'], axis=1, inplace=True)

    sd_plot_2(run1, run2, run3, run4)

    ra_plot(run1, run2, run3, run4)
    ra_plot(run1, run2, run3, run4, window=100)
    '''
    run1 = pd.read_csv('data-files/run1.csv')
    run2 = pd.read_csv('data-files/run2.csv')
    run3 = pd.read_csv('data-files/run3.csv')
    run4 = pd.read_csv('data-files/run4.csv')

    frames = [run1, run2, run3, run4]

    long_file = pd.concat(frames)

    columns = long_file.columns

    mean = long_file[columns[2]].mean()
    print(mean)
    std = long_file[columns[2]].std()
    print(std)
    min = long_file[columns[2]].min()
    print(min)

    run1 = run1.drop(['total_timesteps'], axis=1)
    run1 = run1.groupby('episode').mean()
    run1['average'] = run1.expanding().mean()

    run2 = run2.drop(['total_timesteps'], axis=1)
    run2 = run2.groupby('episode').mean()
    run2['average'] = run2.expanding().mean()

    run3 = run3.drop(['total_timesteps'], axis=1)
    run3 = run3.groupby('episode').mean()
    run3['average'] = run3.expanding().mean()

    run4 = run4.drop(['total_timesteps'], axis=1)
    run4 = run4.groupby('episode').mean()
    run4['average'] = run4.expanding().mean()

    # ra_plot(run1, run2, run3, run4)
    # ra_plot(run1, run2, run3, run4, window=100)
    # sd_plot(run1, run2, run3, run4)
    '''

import time


class report:

    def __init__(self):
        # open files
        self.step_file = open("td3/results/TD3_results_step.csv", "w")
        self.episode_file = open("td3/results/TD3_results_episode.csv", "w")

        # write headers
        self.step_file.write("Step,Reward,Avg_Reward_Last_100,Avg_Reward_Last_1000,Avg_Reward_All,Time_Elapsed")
        self.episode_file.write("Episode,Steps_In_Episode,Total_Steps,Mean_Episode_Reward,Mean_Reward_All,"
                                "Solved, Reward, Memory_used,Time_Elapsed,Solved")

    def write_step(self, values, elapsed_time):
        reward = values[-1]
        mean_all = sum(values)/len(values)

        # mean rewards
        if len(values) >= 100:
            mean_100 = sum(values[-100:])/100
        else:
            mean_100 = 'null'

        if len(values) >= 1000:
            mean_1000 = sum(values[-1000:])/1000
        else:
            mean_1000 = 'null'

        if len(values) >= 10000:
            mean_10000 = sum(values[-10000:])/10000
        else:
            mean_10000 = 'null'

        self.step_file.write('\n{},{},{},{},{},{},{}'.format(len(values), reward, mean_100, mean_1000, mean_10000,
                                                             mean_all, time.strftime("%H:%M:%S",
                                                                                     time.gmtime(elapsed_time))))

    def write_episode(self, episode, steps, total, mean_episode, mean_all, solved, reward, sys, t):

        self.episode_file.write('\n{},{},null,null,{},{},{},{},{},{}'.format(episode, steps, total, mean_episode,
                                                                             mean_all, solved, reward, sys,
                                                                             time.strftime("%H:%M:%S", time.gmtime(t))))

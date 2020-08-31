from datetime import datetime
import time


class report:

    def __init__(self):
        # open files
        timestr = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")
        self.step_file = open("td3/results/reports/TD3_results_step_{}.csv".format(timestr), "w+")
        self.episode_file = open("td3/results/reports/TD3_results_episode_{}.csv".format(timestr), "w+")
        self.dist_file = open("td3/results/reports/TD3_distance_to_target_{}.csv".format(timestr), "w+")
        self.actor_loss_file_1 = open("td3/results/reports/TD3_actor_loss_1_{}.csv".format(timestr), "w+")
        self.actor_loss_file_2 = open("td3/results/reports/TD3_actor_loss_2_{}.csv".format(timestr), "w+")
        self.critic_loss_file_1 = open("td3/results/reports/TD3_critic_loss_1_{}.csv".format(timestr), "w+")
        self.critic_loss_file_2 = open("td3/results/reports/TD3_critic_loss_2_{}.csv".format(timestr), "w+")
        self.solved_file = open("td3/results/reports/TD3_solved_{}.csv".format(timestr), "w+")
        self.evaluation_file = open("td3/results/reports/TD3_evaluation_{}.csv".format(timestr), "w+")
        # write headers
        self.step_file.write("Step,Reward,Avg_Reward_Last_100,Avg_Reward_Last_1000,Avg_Reward_All,Time_Elapsed")
        self.episode_file.write("Episode,Steps_In_Episode,Total_Steps,Mean_Episode_Reward,Mean_Reward_All,"
                                "Solved, Reward, Memory_used,Time_Elapsed,Solved")
        self.dist_file.write("episode,total_timesteps, distance_to_target")
        self.actor_loss_file_1.write('total_iterations, iteration_num, actor_loss')
        self.actor_loss_file_2.write('total_iterations, iteration_num, actor_loss')
        self.critic_loss_file_1.write('total_iterations, iteration_num, critic_loss')
        self.critic_loss_file_1.write('total_iterations, iteration_num, critic_loss')
        self.solved_file.write('episode,solved_on_step')
        self.evaluation_file.write('episode,reward')

    def write_eval_reward(self, episode, reward):
        self.evaluation_file.write('\n{}'.format(episode, reward))
        self.evaluation_file.flush()

    def write_solved(self, episode, solved_on_step):
        self.solved_file.write('\n{},{}'.format(episode, solved_on_step))
        self.solved_file.flush()

    def write_actor_loss(self, total_iterations, iteration_num, actor_loss, actor_num):
        if actor_num == 2:
            actor_file = self.actor_loss_file_2
        else:
            actor_file = self.actor_loss_file_1
        actor_file.write('\n{},{},{}'.format(total_iterations, iteration_num, actor_loss))
        actor_file.flush()

    def write_critic_loss(self, total_iterations, iteration_num, critic_loss, critic_num=1):
        if critic_num == 1:
            self.critic_loss_file_1.write('\n{},{},{}'.format(total_iterations, iteration_num, critic_loss))
            self.critic_loss_file_1.flush()
        if critic_num == 2:
            self.critic_loss_file_2.write('\n{},{},{}'.format(total_iterations, iteration_num, critic_loss))
            self.critic_loss_file_2.flush()

    def write_dist_to_target(self, episode, total_timesteps, distance_to_target):
        self.dist_file.write('\n{},{},{}'.format(episode, total_timesteps, distance_to_target))
        self.dist_file.flush()

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
        self.step_file.flush()

    def write_episode(self, episode, steps, total, mean_episode, mean_all, solved, reward, sys, t):

        self.episode_file.write('\n{},{},null,null,{},{},{},{},{},{}'.format(episode, steps, total, mean_episode,
                                                                             mean_all, solved, reward, sys,
                                                                             time.strftime("%H:%M:%S", time.gmtime(t))))
        self.episode_file.flush()

from datetime import datetime
import td3.constants as cons


class Reports:

    def __init__(self):
        """
        Reports Needed:
        + step report: episode, step, reward, step_distance_moved, step_distance_target , solved, time_elapsed
        + actor loss report: episode, step, actor_1_loss, actor_2_loss
        + critic loss report: episode, step, critic_1_loss, critic_2_loss
        + error report: episode, step, error
        """
        # get date for the run
        timestr = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")

        # create files for each episode
        self.step_report = open("results/reports/A_{}_step_report_{}.csv".format(cons.MODE, timestr), "w+")
        self.actor_report = open("results/reports/B_{}_actor_loss_report_{}.csv".format(cons.MODE, timestr), "w+")
        self.critic_report = open("results/reports/C_{}_critic_loss_report_{}.csv".format(cons.MODE, timestr), "w+")
        self.error_report = open("results/reports/D_{}_error_report_{}.csv".format(cons.MODE, timestr), "w+")

        # write headers for files
        self.step_report.write("episode,step,reward,step_distance_moved,step_distance_target,solved,time_elapsed")
        self.actor_report.write("episode,step,actor_1_loss,actor_2_loss")
        self.critic_report.write("episode,step,critic_1_loss,critic_2_loss")
        self.error_report.write("episode,step,error")

        # create temp storage lists
        self.step_list = []
        self.actor_list = []
        self.critic_list = []
        self.error_list = []

        # store data for each timestep in the report
        def report_step(episode, step, reward, step_distance_moved, step_distance_target, solved, time_elapsed):
            """
            Records the episode training results at each timestep
            :param int episode: the episode number
            :param int step: the current timestep
            :param float reward: reward at current timestep
            :param float step_distance_moved: distance moved by the end joint during the episode
            :param float step_distance_target: distance between the target and the end joint at the end of the episode
            :param boolean solved: whether the episode is solved or not
            :param time_elapsed: training time elapsed since the start of training
            """
            # add all the data to the list
            self.step_list.append([episode, step, reward, step_distance_moved, step_distance_target, solved, time_elapsed])
            # check for save interval if interval, write to file, reset storage list
            if len(self.step_list) is 100:
                # write to the file
                write_report(self.step_report, self.step_list)
                # reset file for next batch
                self.step_list = []

        def report_actor(episode, step, actor_1_loss, actor_2_loss):
            """
            Records the actor loss for each training step
            :param int episode: the current episode number
            :param int step: the current timestep
            :param float actor_1_loss: loss value from actor 1
            :param float actor_2_loss: loss value from actor 2
            """
            self.actor_list.append([episode, step, actor_1_loss, actor_2_loss])
            if len(self.actor_list) is 100:
                write_report(self.actor_report, self.actor_list)
                self.actor_list = []

        def report_critic(episode, step, critic_1_loss, critic_2_loss):
            """
            Records the critic loss for each training step
            :param int episode: the current episode number
            :param int step: the current timestep
            :param float critic_1_loss: loss value from critic 1
            :param float critic_2_loss: loss value from critic 2
            """
            self.critic_list.append([episode, step, critic_1_loss, critic_2_loss])
            if len(self.critic_list) is 100:
                write_report(self.critic_report, self.critic_list)
                self.critic_list = []

        def report_error(episode, step, error):
            """
            Records the error at each training step
            :param int episode: the current episode number
            :param int step: the current step
            :param float error: the error value
            :return:
            """
            # TODO need to get the error values in TD3, then update this function
            self.error_list.append([episode, step, error])
            if len(self.error_list) is 100:
                write_report(self.error_report, self.error_list)
                self.error_list = []

        # periodically write the data to the file, reinitialize the list
        def write_report(report_file, write_list):
            """
            writes list of values to the specified report
            :param String report_file: name of the report to write
            :param list write_list: list of the values to append to the report
            """
            # write all lines to file
            for row in write_list:
                report_file.write("%s\n" % ','.join(str(col) for col in row))



import file_names as names
import globals


class Reports:

    def __init__(self):
        """
        Reports Needed:
        + step report: episode, step, reward, step_distance_moved, step_distance_target , solved, time_elapsed
        + actor loss report: episode, step, actor_1_loss, actor_2_loss
        + critic loss report: episode, step, critic_1_loss, critic_2_loss
        + error report: episode, step, error
        """
        # write headers for files
        names.STEP_REPORT.write("episode,step,reward,step_distance_moved,step_distance_target,solved,time_elapsed\n")
        names.ACTOR_REPORT.write("episode,step,actor_1_loss,actor_2_loss\n")
        names.CRITIC_REPORT.write("episode,step,critic_1_loss,critic_2_loss\n")
        names.ERROR_REPORT.write("episode,step,error\n")

        # create temp storage lists
        self.step_list = []
        self.actor_list = []
        self.critic_list = []
        self.error_list = []

    # store data for each timestep in the report
    def write_report_step(self, episode, step, reward, step_distance_moved, step_distance_target, solved, time_elapsed):
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
        record = [episode, step, reward, step_distance_moved, step_distance_target, solved, time_elapsed]
        # add all the data to the list
        self.step_list.append(record)
        globals.STEP_LIST.append(record)
        # check for save interval if interval, write to file, reset storage list
        if len(self.step_list) is 100:
            # write to the file
            self.write_report(names.STEP_REPORT, self.step_list)
            # reset file for next batch
            self.step_list = []

    # TODO deal with varying amounts of actors
    def write_report_actor(self, episode, step, actor_1_loss, actor_2_loss):
        """
        Records the actor loss for each training step
        :param int episode: the current episode number
        :param int step: the current timestep
        :param float actor_1_loss: loss value from actor 1
        :param float actor_2_loss: loss value from actor 2
        """
        record = [episode, step, actor_1_loss, actor_2_loss]
        self.actor_list.append(record)
        globals.ACTOR_LIST.append(record)
        if len(self.actor_list) is 100:
            self.write_report(names.ACTOR_REPORT, self.actor_list)
            self.actor_list = []

    # TODO deal with varying amounts of actors
    def write_report_critic(self, episode, step, critic_1_loss, critic_2_loss):
        """
        Records the critic loss for each training step
        :param int episode: the current episode number
        :param int step: the current timestep
        :param float critic_1_loss: loss value from critic 1
        :param float critic_2_loss: loss value from critic 2
        """
        record = [episode, step, critic_1_loss, critic_2_loss]
        self.critic_list.append(record)
        globals.CRITIC_LIST.append(record)
        if len(self.critic_list) is 100:
            self.write_report(names.CRITIC_REPORT, self.critic_list)
            self.critic_list = []

    def write_report_error(self, episode, step, error):
        """
        Records the error at each training step
        :param int episode: the current episode number
        :param int step: the current step
        :param float error: the error value
        :return:
        """
        # TODO need to get the error values in TD3, then update this function
        record = [episode, step, error]
        self.error_list.append(record)
        globals.ERROR_LIST.append(record)
        if len(self.error_list) is 100:
            self.write_report(names.ERROR_REPORT, self.error_list)
            self.error_list = []

    def write_report(self, report_file, write_list):
        """
        writes list of values to the specified report
        :param report_file: name of the report to write
        :param list write_list: list of the values to append to the report
        """
        # write all lines to file
        for row in write_list:
            report_file.write("%s\n" % ','.join(str(col) for col in row))


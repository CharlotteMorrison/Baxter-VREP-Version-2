import numpy as np
import td3.constants as cons
from utils import output_video
import time
import psutil
import torch
import td3.rewards as rew
import globals as glo
import set_mode
from statistics import mean
import file_names as names
from graphs import Graphs
import GPUtil
import gc


def train(agent, sim, replay_buffer):
    """Train the agent for exploration steps
        Args:
            :param replay_buffer: (ReplayBuffer) replay buffer for arm
            :param sim: (robot environment) vrep simulation
            :param agent: (Agent) agent to use, TD3 Algorithm
    """
    best_avg = -10
    start_time = time.time()

    # initialize graphs
    graphs = Graphs()

    while glo.TIMESTEP < cons.EXPLORATION:
        if glo.EPISODE == cons.MAX_EPISODE:
            print('Max Episode Reached: {}'.format(cons.MAX_EPISODE))
            break
        print('Timesteps: {}/{}.'.format(glo.TIMESTEP, cons.EXPLORATION))

        glo.EPISODE += 1  # start at episode 1, after the test so the last episode runs

        # get the current state  of the robot arms (joint angles)
        right_pos, left_pos = sim.get_current_position()
        state = right_pos + left_pos

        # video recording
        video_array = []
        if glo.EPISODE % cons.VIDEO_INTERVAL == 0:
            video_record = True
            video_array.append(sim.get_video_image())
        else:
            video_record = False

        # reset episode variables
        solved = False  # reset each episode as unsolved
        rewards = []  # used to store the episode rewards, used to average the rewards.
        collision_count = 0
        episode_length = 0
        # -------------------------------------------------------------------------------------------------
        # Start Episode
        # -------------------------------------------------------------------------------------------------
        while True:
            glo.TIMESTEP += 1
            episode_length += 1

            # check memory utilization
            '''
            GPUtil.showUtilization()
            print(torch.cuda.memory_allocated())

            count_objs = 0

            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        # print(type(obj), obj.size())
                        count_objs += 1
                except:
                    pass
            print("Total objects in GPU memory: {}".format(count_objs))
            '''

            # get the initial location of the target object
            target_start = sim.get_target_position()

            new_state = []
            if set_mode.MODE == 'cooperative':
                # get a new action based on policy
                action = agent.select_action(np.array(state), noise=cons.POLICY_NOISE).tolist()
                # apply the action and get the new state
                right_state, left_state = sim.step_arms(action[:7], action[7:])
                new_state = right_state + left_state

            elif set_mode.MODE == 'independent' or set_mode.MODE == 'partial':
                right_action = agent.select_action(np.array(state)[:7], 'right', noise=cons.POLICY_NOISE).tolist()
                left_action = agent.select_action(np.array(state)[7:], 'left', noise=cons.POLICY_NOISE).tolist()

                # apply the action and get the new state
                right_state, left_state = sim.step_arms(right_action, left_action)
                new_state = right_state + left_state

                # store the actions together in the replay
                action = right_action + left_action

            # add the image to the video array
            if video_record:
                video_array.append(sim.get_video_image())

            # get the new position of the target
            target_end = sim.get_target_position()
            target_x, target_y, target_z = target_end

            # calculate the reward and distance to target for the step
            reward, distance_moved, distance_to_target = rew.target_movement_reward(target_start, target_end,
                                                                                    cons.XYZ_GOAL)

            # round the x, y, z to compare with the goal positions
            round_target_x = round(target_x, 1)
            round_target_y = round(target_y, 1)
            round_target_z = round(target_z, 1)

            # check and see if the target is within the goal range
            if round_target_x == cons.XYZ_GOAL[0] and round_target_y == cons.XYZ_GOAL[1] and \
                    round_target_z == cons.XYZ_GOAL[2]:
                # end episode if the goal is reached
                done = True
            else:
                done = False

            # check for multiple collisions in a row (5), reset sim if so
            object_collision_table = sim.object_collision_state()
            if object_collision_table:
                # if it collides with the table 5 times in a row - end episode
                collision_count += 1
                if collision_count > 4:
                    done = True
                    collision_count = 0
            else:
                collision_count = 0

            # check distance between grippers to see if dropped
            if sim.check_suction_distance() > .32:
                done = True
                time.sleep(1)  # wait to allow the sim to catch up
                reward = - 1  # was zero, try a big, bad reward when you drop it
                # moved back to 0, since priority reply can't handle negs, this is only
                # giving me an out-sized negative value, not really helpful.
                # will make the graphs look a little better, will be around the x=0 axis, not only below

            # if it is dropped, reward is zero. end the episode and start a new one, it was very bad to drop it.
            if not sim.check_suction_prox():
                done = True
                time.sleep(1)  # wait to allow the sim to catch up
                reward = 0  # was zero, try a big, bad reward when you drop it

            # update the replay buffer with new tuple
            replay_buffer.add(state, action, reward, new_state, done)

            # training step
            agent.train(replay_buffer, episode_length)

            # set the state to the new state for the next step
            state = new_state

            if solved:
                print('Solved on Episode: {}'.format(glo.EPISODE))

            # add the reward to the reward list
            rewards.append(reward)

            if episode_length == 50:   # stop after 50 attempts, 30 was too low to reach goal, tried 45.
                done = True  # stop after 50 attempts, was getting stuck flipping from bad to good.

            # calculate the elapsed time
            elapsed_time = time.time() - start_time

            if cons.WRITE_TO_FILE:
                cons.report.write_report_step(glo.EPISODE, glo.TIMESTEP, reward, distance_moved, distance_to_target,
                                              solved, elapsed_time)

            if done:
                # get the average reward for the episode
                mean_reward_episode = mean(rewards)

                # if current episodes reward better, becomes new save
                if best_avg < mean_reward_episode:
                    best_avg = mean_reward_episode
                    agent.save()

                if video_record and episode_length > 10:  # only record episodes over 10
                    output_video(video_array, cons.SIZE, names.EPISODE_VIDEO)
                if solved:
                    output_video(video_array, cons.SIZE, names.EPISODE_VIDEO_SOLVED)

                system_info = psutil.virtual_memory()
                if True:
                    print("\n*** Episode " + str(glo.EPISODE) + " ***")
                    print("Avg_Reward [last episode of " + str(episode_length) + " steps]: " + str(mean_reward_episode))
                    print("Total Timesteps: " + str(glo.TIMESTEP))
                    print("Elapsed Time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                    print("Memory Usage: " + str(system_info.percent) + "%")

                # reset the simulation at the end of the episode
                sim.reset_sim()

                # at the end of the episode, visualize the charts.
                graphs.update_step_list_graphs()

                # end the episode
                break

        # check for system overload, if memory over-utilized, quit
        system_info = psutil.virtual_memory()
        if system_info.percent > 98:
            break

    # write any remaining values
    cons.report.write_final_values()

    # at the end do a final update of the graphs
    graphs.update_step_list_graphs()


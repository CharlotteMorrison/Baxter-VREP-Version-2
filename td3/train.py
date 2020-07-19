import numpy as np
import td3.constants as cons
from utils import output_video, plot_results
import time
import psutil
import torch
from td3.rewards import *
import td3.rewards as rew


def train(agent, sim, replay_buffer):
    """Train the agent for exploration steps
        Args:
            :param replay_buffer: (ReplayBuffer) replay buffer for arm
            :param sim: (robot environment) vrep simulation
            :param agent: (Agent) agent to use, TD3 Algorithm
    """
    total_timesteps = 0
    best_avg = -10
    episode = 0
    start_time = time.time()

    loc = 'td3/saves'

    # lists storage for result tracking
    rewards = []                # every step reward
    episode_rewards = []        # the average reward for each episode
    all_episode_rewards = []    # the average reward for all episodes, after each new one
    end_reward = []             # the reward at the end of the episode
    avg_last_10_episodes = []   # sliding window of the previous 10 episode rewards
    episode_length = []         # the length of each episode

    if cons.WRITE_TO_FILE:
        td3_report = cons.TD3_REPORT

    while total_timesteps < cons.EXPLORATION:
        if episode == cons.MAX_EPISODE:
            print('Max Episode Reached: {}'.format(cons.MAX_EPISODE))
            break
        print('Timesteps: {}/{}.'.format(total_timesteps, cons.EXPLORATION))
        episode += 1
        steps_in_episode = 0  # counts the number of step in the episode

        # get the current state  of the robot arms (joint angles)
        right_pos, left_pos = sim.get_current_position()
        state = right_pos + left_pos

        # video recording
        video_array = []
        if episode % cons.VIDEO_INTERVAL == 0:
            video_record = True
        else:
            video_record = False
        video_array.append(sim.get_video_image())

        # reset variables
        solved = False  # reset each episode as unsolved
        index = 0  # track the number of bad moves made.
        temp_steps = 0  # tracks the number of tries- if above 45, is done, reset.
        collision_count = 0

        while True:
            total_timesteps += 1
            steps_in_episode += 1

            # get the initial location of the target object
            target_start = sim.get_target_position()

            new_state = []
            if cons.MODE == 'cooperative':
                loc = 'td3/saves/shared_agent'
                # get a new action based on policy
                action = agent.select_action(np.array(state), noise=cons.POLICY_NOISE).tolist()
                # apply the action and get the new state
                right_state, left_state = sim.step_arms(action[:7], action[7:])
                new_state = right_state + left_state

            elif cons.MODE == 'independent' or cons.MODE == 'separate':
                if cons.MODE == 'independent':
                    loc = 'td3/saves/dual_agent'
                elif cons.MODE == 'separate':
                    loc = 'td3/saves/separate_agent'
                right_action = agent.select_action(np.array(state)[:7], 'right', noise=cons.POLICY_NOISE).tolist()
                left_action = agent.select_action(np.array(state)[7:], 'left', noise=cons.POLICY_NOISE).tolist()

                # apply the action and get the new state
                right_state, left_state = sim.step_arms(right_action, left_action)
                new_state = right_state + left_state

                # store the actions together in the replay
                action = right_action + left_action

            # add the image to the video array
            video_array.append(sim.get_video_image())

            # check for collision state/ if done
            # right_arm_collision_state = sim.right_collision_state()
            # left_arm_collision_state = sim.left_collision_state()

            ''' code for moving arms towards a target    
            # calculate reward
            right_reward, left_reward = sim.calc_distance()
            reward = arm_distance_reward(right_reward, left_reward)

            # check if solved
            if right_reward > cons.SOLVED_DISTANCE and left_reward > cons.SOLVED_DISTANCE:
                done = True
                solved = True

            # check for collisions
            elif right_arm_collision_state or left_arm_collision_state:
                done = True
                solved = False
            else:
                done = False
            '''
            target_end = sim.get_target_position()
            target_x, target_y, target_z = target_end
            reward, dist_to_target = rew.target_movement_reward(target_start, target_end, cons.XYZ_GOAL)

            # removed collision state checking may add back in to check for box collision with the table.

            round_target_x = round(target_x, 1)
            round_target_y = round(target_y, 1)
            round_target_z = round(target_z, 1)

            if round_target_x == cons.XYZ_GOAL[0] and round_target_y == cons.XYZ_GOAL[1] and \
                    round_target_z == cons.XYZ_GOAL[2]:
                # end the episode if the target is reached, might be too restrictive- maybe round all to 1 decimal place
                done = True
            else:
                done = False

            # if robot makes more than 10 bad moves in a row, end the episode
            # changed to 5 to speed up ending episodes
            '''
            if reward < 0:
                index += 1
            else:
                index = 0

            if index >= 10:  # increased to 10 from 5
                done = True
                solved = False
            '''
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
                reward = -1  # was zero, try a big, bad reward when you drop it

            # if it is dropped, reward is zero. end the episode and start a new one, it was very bad to drop it.
            if not sim.check_suction_prox():
                done = True
                time.sleep(1)  # wait to allow the sim to catch up
                reward = -1  # was zero, try a big, bad reward when you drop it

            # store the reward for the step
            rewards.append(reward)

            # update the replay buffer with new tuple
            replay_buffer.add(state, torch.tensor(action, dtype=torch.float32), reward, new_state, done)

            # training step
            agent.train(replay_buffer, cons.BATCH_SIZE)

            state = new_state

            if solved:
                print('Solved on Episode: {}'.format(episode))

            temp_steps += 1
            if temp_steps == 100:   # stop after 50 attempts, 30 was too low to reach goal, tried 45.
                done = True  # stop after 50 attempts, was getting stuck flipping from bad to good.
                temp_steps = 0

            system_info = psutil.virtual_memory()

            if cons.WRITE_TO_FILE:
                elapsed_time_frame = time.time() - start_time
                td3_report.write_step(rewards, elapsed_time_frame)
                td3_report.write_dist_to_target(episode, total_timesteps, dist_to_target)

            if done:
                # append the final positions reward
                end_reward.append(reward)
                episode_length.append(steps_in_episode)

                # get the average reward for the episode and the overall average reward
                mean_reward_episode = sum(rewards[-steps_in_episode:])/steps_in_episode
                mean_reward_all = round(sum(rewards) / len(rewards), 4)

                # if current episodes reward better, becomes new save
                if best_avg < mean_reward_episode:
                    best_avg = mean_reward_episode
                    agent.save("best_avg", loc)

                # save the progress for graphing
                episode_rewards.append(mean_reward_episode)
                all_episode_rewards.append(mean_reward_all)

                elapsed_time = time.time() - start_time

                if video_record and temp_steps > 30:  # only record episodes over 30
                    output_video(episode, video_array, cons.SIZE, "td3/videos/" + cons.DEFAULT_NAME)
                if solved:
                    output_video(episode, video_array, cons.SIZE, "td3/videos/" + cons.DEFAULT_NAME + "_solved")

                if cons.WRITE_TO_FILE:
                    td3_report.write_episode(episode, steps_in_episode, total_timesteps, mean_reward_episode,
                                             mean_reward_all, solved, reward, system_info.used, elapsed_time)

                # plot rewards
                if episode % cons.REPORT_INTERVAL == 0 and episode > 0:

                    # get the average reward of last 10 episodes
                    if len(episode_rewards) > 9:
                        avg_last_10_episodes.append(sum(episode_rewards[-10:]) / 10)
                        plot_results(avg_last_10_episodes, cons.AVG_10_PLOT_NAME, 'Average Reward Previous 10 Episodes')

                    plot_results(episode_length, cons.EPISODE_LENGTH_NAME, 'Episode Total Average Reward')
                    plot_results(episode_rewards, cons.EPISODE_PLOT_NAME, 'Episode Total Average Reward')
                    plot_results(all_episode_rewards, cons.ALL_PLOT_NAME, 'Total Average Reward')
                    plot_results(end_reward, cons.END_PLOT_NAME, 'Reward For Last Move')

                    print("\n*** Episode " + str(episode) + " ***")
                    print("Avg_Reward [last episode of " + str(steps_in_episode) + " steps]: " + str(
                        mean_reward_episode) + ", [all]: " + str(mean_reward_all))
                    print("Total Timesteps: " + str(total_timesteps))
                    print("Elapsed Time: ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
                    print("Memory Usage: " + str(system_info.percent) + "%")
                sim.reset_sim()
                break

        system_info = psutil.virtual_memory()

        if system_info.percent > 98:
            break

    if len(episode_rewards) > 9:
        avg_last_10_episodes.append(sum(episode_rewards[-10:]) / 10)
        plot_results(avg_last_10_episodes, cons.AVG_10_PLOT_NAME, 'Average Reward Previous 10 Episodes')
    plot_results(episode_rewards, cons.EPISODE_PLOT_NAME, 'Episode Total Average Reward')
    plot_results(all_episode_rewards, cons.ALL_PLOT_NAME, 'Total Average Reward')
    plot_results(end_reward, cons.END_PLOT_NAME, 'Reward For Last Move')


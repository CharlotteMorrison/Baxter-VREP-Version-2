import td3.constants as cons
from utils import output_video
import td3.rewards as rew
import numpy as np


def evaluate_policy(policy, sim, eval_episodes=50, episode_length=50):
    """run several episodes with the best agent policy"""
    avg_reward = []

    for i in range(eval_episodes):
        video_array = []
        episode_reward = []
        sim.reset_sim()
        video_array.append(sim.get_video_image())

        right_pos, left_pos = sim.get_current_position()
        state = right_pos + left_pos
        done = False
        num_of_steps = 0

        # run policy
        while not done:
            num_of_steps += 1
            # get the starting target location
            target_start = sim.get_target_position()
            # get action by policy
            action = policy.select_action(state, noise=0)
            if cons.MODE == 'cooperative':
                action = policy.select_action(state, noise=0)

            elif cons.MODE == 'independent':
                loc = 'td3/saves/dual_agent'
                right_action = policy.select_action(np.array(state)[:7], 'right', noise=0)
                left_action = policy.select_action(np.array(state)[7:], 'left', noise=0)
                action = right_action + left_action

            # apply action and get new state
            right_state, left_state = sim.step_arms(action[:7], action[7:])
            state = right_state + left_state
            video_array.append(sim.get_video_image())
            ''' old reward and collision detection
            # calculate reward
            right_reward, left_reward = sim.calc_distance()
            reward = (right_reward + left_reward) / 2
            episode_reward.append(reward)
            print(reward)
            # determine if done
            right_arm_collision_state = sim.right_collision_state()
            left_arm_collision_state = sim.left_collision_state()

            # check if solved
            if right_reward > cons.SOLVED_DISTANCE and left_reward > cons.SOLVED_DISTANCE:
                done = True
            if right_arm_collision_state or left_arm_collision_state:
                done = True
            if num_of_steps > episode_length:  # needs to run longer to get near the target
                done = True
            '''
            target_end = sim.get_target_position()
            target_x, target_y, target_z = target_end
            reward = rew.target_movement_reward(target_start, target_end, cons.XYZ_GOAL)
            episode_reward.append(reward)

            if round(target_x, 2) == cons.XYZ_GOAL[0] and round(target_y, 2) == cons.XYZ_GOAL[1] and \
                    round(target_z, 2) == cons.XYZ_GOAL[2]:
                done = True
            elif num_of_steps > episode_length:
                done = True
            else:
                done = False
        avg_reward.append(sum(episode_reward)/len(episode_reward))
        output_video(i, video_array, cons.SIZE, "td3/videos/evaluate/" + cons.DEFAULT_NAME)
    total_avg_reward = sum(avg_reward)/len(avg_reward)

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, total_avg_reward))
    print("---------------------------------------")
    return avg_reward

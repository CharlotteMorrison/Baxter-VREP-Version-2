import td3.constants as cons
from utils import output_video
import td3.rewards as rew
import statistics as sta
import set_mode
import file_names
import graphs


def evaluate_policy(policy, sim, eval_episodes=5, episode_length=50):
    """run several episodes with the best agent policy"""
    avg_reward = []

    for i in range(eval_episodes):
        print("Evaluation episode: {}".format(i + 1))
        video_array = []
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
            if set_mode.MODE == 'cooperative':
                action = policy.select_action(state, noise=0)
                # apply action and get new state
                right_state, left_state = sim.step_arms(action[:7], action[7:])

            elif set_mode.MODE == 'independent' or set_mode.MODE == 'partial':
                right_action = policy.select_action(right_pos, 'right', noise=0)
                left_action = policy.select_action(left_pos, 'left', noise=0)
                right_state, left_state = sim.step_arms(right_action, left_action)

            state = right_state + left_state

            video_array.append(sim.get_video_image())

            target_end = sim.get_target_position()
            target_x, target_y, target_z = target_end
            reward, _, _ = rew.target_movement_reward(target_start, target_end, cons.XYZ_GOAL)

            if round(target_x, 2) == cons.XYZ_GOAL[0] and round(target_y, 2) == cons.XYZ_GOAL[1] and \
                    round(target_z, 2) == cons.XYZ_GOAL[2]:
                done = True
            elif num_of_steps > episode_length:
                done = True
            else:
                done = False
            if not sim.check_suction_prox():
                done = True

            cons.report.write_evaluate_step(i, num_of_steps, reward, eval_episodes)
            avg_reward.append(reward)
        output_video(video_array, cons.SIZE, file_names.EVALUATION_VIDEO, evaluate=True, eval_num=i)

    graph = graphs.Graphs()
    graph.avg_evaluation_episode_reward()

    total_avg_reward = sta.mean(avg_reward)

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, total_avg_reward))
    print("---------------------------------------")
    return avg_reward

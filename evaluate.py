import td3.constants as cons
from utils import output_video


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
            # get action by policy
            action = policy.select_action(state, noise=0)

            # apply action and get new state
            right_state = sim.step_right(action[:7])
            left_state = sim.step_left(action[7:])
            state = right_state + left_state
            video_array.append(sim.get_video_image())

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
        avg_reward.append(sum(episode_reward)/len(episode_reward))
        output_video(i, video_array, cons.SIZE, "td3/videos/evaluate/" + cons.DEFAULT_NAME)
    total_avg_reward = sum(avg_reward)/len(avg_reward)

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, total_avg_reward))
    print("---------------------------------------")
    return avg_reward

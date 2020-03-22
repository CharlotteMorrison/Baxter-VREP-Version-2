import td3.constants as cons


def evaluate_policy(policy, sim, eval_episodes=100):
    """run several episodes with the best agent policy"""
    avg_reward = 0
    for i in range(eval_episodes):
        sim.reset()
        right_pos, left_pos = sim.get_current_position()
        state = right_pos + left_pos
        done = False
        index = 0

        # run policy
        while not done:
            # get action by policy
            action = policy.select_action(state, noise=0)

            # apply action and get new state
            right_state = sim.step_right(action[:7])
            left_state = sim.step_left(action[7:])
            state = right_state + left_state

            # calculate reward
            right_reward, left_reward = sim.calc_distance()
            reward = right_reward + left_reward / 2
            avg_reward += reward

            # determine if done
            right_arm_collision_state = sim.right_collision_state()
            left_arm_collision_state = sim.left_collision_state()

            # check if solved
            if right_reward > cons.SOLVED_DISTANCE and left_reward > cons.SOLVED_DISTANCE:
                done = True
            elif right_arm_collision_state or left_arm_collision_state:
                done = True
            if index >= 5:
                done = True

    avg_reward /= eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

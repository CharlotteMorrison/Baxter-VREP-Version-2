import random
import sys
import torch
import td3.constants as cons


def initialize_min_buffer(sim, replay_buffer):
    print("\nInitializing experience replay buffer...")

    # store and load the initial replay values up to 100 to have an initial batch of values
    for x in range(10):
        right_pos, left_pos = sim.get_current_position()
        state = right_pos + left_pos

        value = [-0.1, 0, 0.1]
        right_action = []
        left_action = []
        for i in range(7):
            right_action.append(random.choice(value))
            left_action.append(random.choice(value))

        action = right_action + left_action
        if cons.MODE == 'cooperative':
            right_state = sim.step_right(right_action)
            left_state = sim.step_left(left_action)
            next_state = right_state + left_state

        elif cons.MODE == 'independent':
            right_state = sim.step_right(right_action)
            left_state = sim.step_left(left_action)
            next_state = right_state + left_state

        right_reward, left_reward = sim.calc_distance()

        right_arm_collision_state = sim.right_collision_state()
        left_arm_collision_state = sim.left_collision_state()

        if right_reward > cons.SOLVED_DISTANCE or left_reward > cons.SOLVED_DISTANCE:
            done = True
        elif right_arm_collision_state or left_arm_collision_state:
            done = True
        else:
            done = False

        # set new reward as average of both rewards
        reward = right_reward + left_reward / 2

        replay_buffer.add(state, torch.tensor(action, dtype=torch.float32), reward,
                          next_state, done)
        if done:
            sim.reset_sim()
    print("\nExperience replay buffer initialized.")
    sys.stdout.flush()

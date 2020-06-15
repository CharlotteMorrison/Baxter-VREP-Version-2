import random
import sys
import pickle
import torch
import td3.constants as cons
import psutil
import platform
import td3.rewards as rew


def populate_buffer(sim, replay_buffer):
    print("\nInitializing experience replay buffer...")
    buffer_storage = []

    # store and load the initial replay values
    # once replay buffer is full, use the pre-made one to populate observe step
    replay_counter = 0

    if platform.system() == 'Windows':
        file_loc = "D:\\git\\PythonProjects\\Baxter-VREP-Version-2\\td3\\temp\\buffer.pkl"
    else:
        file_loc = "td3/temp/buffer.pkl"
    with open(file_loc, "rb") as pk_file:
        while True:
            try:
                data = pickle.load(pk_file)
                for test in data:
                    replay_buffer.add(test[0], torch.tensor(test[1], dtype=torch.float32), test[2], test[3], test[4])
                    buffer_storage.append([test[0], test[1], test[2], test[3], test[4]])
                    replay_counter += 1
                    if replay_counter >= cons.BUFFER_SIZE:
                        break
                    # break if too many resources used
                    system_info = psutil.virtual_memory()
                    if system_info.percent > 98:
                        # print("System at 98%")
                        break

            except EOFError:
                print('Reached end of file.')
                break
            except pickle.UnpicklingError:
                print('Incomplete record {} was ignored.'.format(replay_counter + 1))
                break
    # uncomment to update the whole stored replay
    # save_buffer = open("D:\\git\\PythonProjects\\Baxter-VREP\\td3\\temp\\buffer-dist.pkl", "wb")
    # pickle.dump(buffer_storage, save_buffer)
    # save_buffer.close()
    buffer_storage = []
    buffer = cons.BUFFER_SIZE - replay_counter
    print('Buffer size {}/{} loaded from previous session'.format(replay_counter, cons.BUFFER_SIZE))
    collision_count = 0 # keep track of the number of table hits in a row, reset sim if it keeps hitting table
    done = False  # start with initial state of done == false

    for x in range(buffer):

        target_start = sim.get_target_position()

        right_pos, left_pos = sim.get_current_position()
        state = right_pos + left_pos

        value = [-0.1, 0, 0.1]
        right_action = []
        left_action = []
        for i in range(7):
            right_action.append(random.choice(value))
            left_action.append(random.choice(value))

        action = right_action + left_action

        right_state, left_state = sim.step_arms(right_action, left_action)
        next_state = right_state + left_state

        object_collision_table = sim.object_collision_state()

        # right_arm_collision_state = sim.right_collision_state()
        # left_arm_collision_state = sim.left_collision_state()

        ''' old reward for moving arms towards a target
        right_reward, left_reward = sim.calc_distance()
        if right_reward > cons.SOLVED_DISTANCE or left_reward > cons.SOLVED_DISTANCE:
            done = True
        elif right_arm_collision_state or left_arm_collision_state:
            done = True
        else:
            done = False
        # set new reward as average of both rewards
        reward = (right_reward + left_reward) / 2
        '''

        target_end = sim.get_target_position()
        target_x, target_y, target_z = target_end
        reward, _ = rew.target_movement_reward(target_start, target_end, cons.XYZ_GOAL)

        if round(target_x, 2) == cons.XYZ_GOAL[0] and round(target_y, 2) == cons.XYZ_GOAL[1] and \
                round(target_z, 2) == cons.XYZ_GOAL[2]:
            # end the episode if the target is reached, might be too restrictive- maybe round all to one decimal place
            done = True
        elif x != 0 and x % 50 == 0:
            # end the episode after 100 movements
            done = True
        else:
            done = False

        if object_collision_table:
            # if it collides with the table 5 times in a row - end episode
            collision_count += 1
            if collision_count > 4:
                done = True
                collision_count = 0
        else:
            collision_count = 0

        # if it is dropped, reward is zero. end the episode and start a new one, it was very bad to drop it.
        if not sim.check_suction_prox():
            done = True
            reward = 0

        replay_buffer.add(state, torch.tensor(action, dtype=torch.float32), reward,
                          next_state, done)

        buffer_storage.append([state, action, reward, next_state, done])

        if done:
            sim.reset_sim()

        if x % 50 == 0:
            save_buffer = open(file_loc, "ab")
            pickle.dump(buffer_storage, save_buffer)
            save_buffer.close()
            buffer_storage = []
            # sim.reset_sim()  # reset simulation after 25 movements
        if x % 1000 == 0 and x < cons.BUFFER_SIZE - 100 and x != 0:
            sim.reset_sim()  # moved to every 1000 iterations, to allow for more diverse movement sets
            print("{} of {} loaded".format(x + replay_counter, cons.BUFFER_SIZE))
        elif x == cons.BUFFER_SIZE:
            print("{} of {} loaded".format(x + replay_counter, cons.BUFFER_SIZE))
    save_buffer = open(file_loc, "ab")
    pickle.dump(buffer_storage, save_buffer)
    save_buffer.close()
    print("\nExperience replay buffer initialized.")

    sys.stdout.flush()

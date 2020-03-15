#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sim as vrep
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import cv2


class VrepSim(object):

    def __init__(self):
        # Close any open connections
        vrep.simxFinish(-1)

        # Create Var for client connection
        self.clientID = vrep.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

        if self.clientID != -1:
            print('Connected to remote API server')

            self.right_joint_array = []
            self.left_joint_array = []
            self.right_joint_org_position = []
            self.left_joint_org_position = []

            # input videos and campera
            error_code, self.input_cam = vrep.simxGetObjectHandle(self.clientID, 'input_camera',
                                                                  vrep.simx_opmode_oneshot_wait)
            error_code, self.video_cam = vrep.simxGetObjectHandle(self.clientID, 'video_camera',
                                                                  vrep.simx_opmode_oneshot_wait)
            # main right_target
            error_code, self.main_target = vrep.simxGetObjectHandle(self.clientID, 'right_target',
                                                                    vrep.simx_opmode_oneshot_wait)

            # right arm
            error_code, self.right_hand = vrep.simxGetObjectHandle(self.clientID, 'Baxter_rightArm_camera',
                                                                   vrep.simx_opmode_oneshot_wait)
            error_code, self.right_target = vrep.simxGetObjectHandle(self.clientID, 'right_target',
                                                                     vrep.simx_opmode_oneshot_wait)
            error, self.right_arm_collision_target = vrep.simxGetCollisionHandle(self.clientID,
                                                                                 "right_arm_collision_target#",
                                                                                 vrep.simx_opmode_blocking)
            error, self.right_arm_collision_table = vrep.simxGetCollisionHandle(self.clientID,
                                                                                "right_arm_collision_table#",
                                                                                vrep.simx_opmode_blocking)

            # left arm
            error_code, self.left_hand = vrep.simxGetObjectHandle(self.clientID, 'Baxter_leftArm_camera',
                                                                  vrep.simx_opmode_oneshot_wait)
            error_code, self.left_target = vrep.simxGetObjectHandle(self.clientID, 'leftt_target',
                                                                    vrep.simx_opmode_oneshot_wait)
            error, self.left_arm_collision_target = vrep.simxGetCollisionHandle(self.clientID,
                                                                                "left_arm_collision_target#",
                                                                                vrep.simx_opmode_blocking)
            error, self.left_arm_collision_table = vrep.simxGetCollisionHandle(self.clientID,
                                                                               "left_arm_collision_table#",
                                                                               vrep.simx_opmode_blocking)

            # Used to translate action to joint array position
            self.joint_switch = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 5, 11: 5, 12: 6, 13: 6}

            for x in range(1, 8):
                # right arm
                error_code, right_joint = vrep.simxGetObjectHandle(self.clientID, 'Baxter_rightArm_joint' + str(x),
                                                                   vrep.simx_opmode_oneshot_wait)
                self.right_joint_array.append(right_joint)
                # left arm
                error_code, left_joint = vrep.simxGetObjectHandle(self.clientID, 'Baxter_leftArm_joint' + str(x),
                                                                  vrep.simx_opmode_oneshot_wait)
                self.left_joint_array.append(left_joint)

            for x in range(0, 7):
                vrep.simxGetJointPosition(self.clientID, self.right_joint_array[x], vrep.simx_opmode_streaming)
                vrep.simxGetJointPosition(self.clientID, self.left_joint_array[x], vrep.simx_opmode_streaming)

            for x in range(0, 7):
                # right arm
                error_code, right_temp_pos = vrep.simxGetJointPosition(self.clientID, self.right_joint_array[x],
                                                                       vrep.simx_opmode_buffer)
                self.right_joint_org_position.append(right_temp_pos)
                # left arm
                error_code, left_temp_pos = vrep.simxGetJointPosition(self.clientID, self.left_joint_array[x],
                                                                      vrep.simx_opmode_buffer)
                self.left_joint_org_position.append(left_temp_pos)

            # right hand
            error_code, self.right_xyz_hand = vrep.simxGetObjectPosition(self.clientID, self.right_hand, -1,
                                                                         vrep.simx_opmode_streaming)
            error_code, self.right_xyz_target = vrep.simxGetObjectPosition(self.clientID, self.right_target, -1,
                                                                           vrep.simx_opmode_streaming)
            # left hand
            error_code, self.left_xyz_hand = vrep.simxGetObjectPosition(self.clientID, self.left_hand, -1,
                                                                        vrep.simx_opmode_streaming)
            error_code, self.left_xyz_target = vrep.simxGetObjectPosition(self.clientID, self.left_target, -1,
                                                                          vrep.simx_opmode_streaming)

            error_code, self.xyz_main_target = vrep.simxGetObjectPosition(self.clientID, self.main_target, -1,
                                                                          vrep.simx_opmode_streaming)
            vrep.simxGetVisionSensorImage(self.clientID, self.input_cam, 0, vrep.simx_opmode_streaming)
            vrep.simxGetVisionSensorImage(self.clientID, self.video_cam, 0, vrep.simx_opmode_streaming)

            # right hand
            error_code, self.right_arm_collision_state_target = vrep.simxReadCollision(self.clientID,
                                                                                       self.right_arm_collision_target,
                                                                                       vrep.simx_opmode_streaming)
            error_code, self.right_arm_collision_state_table = vrep.simxReadCollision(self.clientID,
                                                                                      self.right_arm_collision_table,
                                                                                      vrep.simx_opmode_streaming)
            # left hand
            error_code, self.left_arm_collision_state_target = vrep.simxReadCollision(self.clientID,
                                                                                      self.left_arm_collision_target,
                                                                                      vrep.simx_opmode_streaming)
            error_code, self.left_arm_collision_state_table = vrep.simxReadCollision(self.clientID,
                                                                                     self.left_arm_collision_table,
                                                                                     vrep.simx_opmode_streaming)
            time.sleep(1)

        else:
            print('Failed connecting to remote API server')
            sys.exit('Could not connect')

    def right_move_joint(self, action):
        if action == 0 or action % 2 == 0:
            move_interval = 0.03
        else:
            move_interval = -0.03
        joint_num = self.joint_switch.get(action, -1)
        error_code, position = vrep.simxGetJointPosition(self.clientID, self.right_joint_array[joint_num],
                                                         vrep.simx_opmode_buffer)
        error_code = vrep.simxSetJointTargetPosition(self.clientID, self.right_joint_array[joint_num],
                                                     position + move_interval, vrep.simx_opmode_oneshot_wait)
        return error_code

    def left_move_joint(self, action):
        if action == 0 or action % 2 == 0:
            move_interval = -0.03
        else:
            move_interval = 0.03
        joint_num = self.joint_switch.get(action, -1)
        error_code, position = vrep.simxGetJointPosition(self.clientID, self.left_joint_array[joint_num],
                                                         vrep.simx_opmode_buffer)
        error_code = vrep.simxSetJointTargetPosition(self.clientID, self.left_joint_array[joint_num],
                                                     position + move_interval, vrep.simx_opmode_oneshot_wait)
        return error_code

    def get_current_position(self):
        right_pos = []
        left_pos = []
        for x in range(0, 7):
            error_code, right_temp_pos = vrep.simxGetJointPosition(self.clientID, self.right_joint_array[x],
                                                                   vrep.simx_opmode_buffer)
            right_pos.append(right_temp_pos)
            error_code, left_temp_pos = vrep.simxGetJointPosition(self.clientID, self.left_joint_array[x],
                                                                  vrep.simx_opmode_buffer)
            left_pos.append(left_temp_pos)
        return right_pos, left_pos

    def step_right(self, action):
        """Applies an array of actions to all right joint positions.
           Args:
               action (list): increments to add to robot position (-0.1, 0, 0.1)
        """

        # get position of arm, increment by values, then move robot
        start_position = []
        new_position = []

        for x in range(0, 7):
            error_code, temp_pos = vrep.simxGetJointPosition(self.clientID, self.right_joint_array[x],
                                                             vrep.simx_opmode_buffer)
            start_position.append(temp_pos)
            new_position.append(temp_pos + action[x])
            vrep.simxSetJointTargetPosition(self.clientID, self.right_joint_array[x], new_position[x],
                                            vrep.simx_opmode_oneshot_wait)
        return new_position
        # time.sleep(0.1)

    def step_left(self, action):
        """Applies an array of actions to all left joint positions.
           Args:
               action (list): increments to add to robot position (-0.1, 0, 0.1)
        """

        # get position of arm, increment by values, then move robot
        start_position = []
        new_position = []

        for x in range(0, 7):
            error_code, temp_pos = vrep.simxGetJointPosition(self.clientID, self.left_joint_array[x],
                                                             vrep.simx_opmode_buffer)
            start_position.append(temp_pos)
            new_position.append(temp_pos + action[x])
            vrep.simxSetJointTargetPosition(self.clientID, self.left_joint_array[x], new_position[x],
                                            vrep.simx_opmode_oneshot_wait)
        return new_position
        # time.sleep(0.1)

    def calc_distance(self):
        """Calculates the distance between the end effector and a target position
            Args:
            Returns:
                right_distance
                left_distance
        """
        error_code, self.right_xyz_hand = vrep.simxGetObjectPosition(self.clientID, self.right_hand, -1,
                                                                     vrep.simx_opmode_buffer)  # right hand
        error_code, self.left_xyz_hand = vrep.simxGetObjectPosition(self.clientID, self.left_hand, -1,
                                                                     vrep.simx_opmode_buffer)  # left hand

        # removed getting right_target location each time- using the static initial location
        # error_code, self.right_xyz_target = vrep.simxGetObjectPosition(self.clientID, self.right_target, -1,
        #                                                          vrep.simx_opmode_buffer)
        # TODO set this to main target, find static point.
        # need to check if this formula is calculating distance properly
        right_distance = 1 / math.sqrt(
            pow((self.right_xyz_hand[0] - self.right_xyz_target[0]), 2) +
            pow((self.right_xyz_hand[1] - self.right_xyz_target[1]), 2) +
            pow((self.right_xyz_hand[2] - self.right_xyz_target[2]), 2))
        left_distance = 1 / math.sqrt(
            pow((self.left_xyz_hand[0] + self.left_xyz_target[0]), 2) +
            pow((self.left_xyz_hand[1] + self.left_xyz_target[1]), 2) +
            pow((self.left_xyz_hand[2] + self.left_xyz_target[2]), 2))

        return right_distance, left_distance

    def right_collision_state(self):
        error_code, self.right_arm_collision_state_target = vrep.simxReadCollision(self.clientID,
                                                                                   self.right_arm_collision_target,
                                                                                   vrep.simx_opmode_buffer)
        error_code, self.right_arm_collision_state_table = vrep.simxReadCollision(self.clientID,
                                                                                  self.right_arm_collision_table,
                                                                                  vrep.simx_opmode_buffer)
        if self.right_arm_collision_state_target or self.right_arm_collision_state_table:
            return True
        else:
            return False

    def left_collision_state(self):
        error_code, self.left_arm_collision_state_target = vrep.simxReadCollision(self.clientID,
                                                                                  self.left_arm_collision_target,
                                                                                  vrep.simx_opmode_buffer)
        error_code, self.left_arm_collision_state_table = vrep.simxReadCollision(self.clientID,
                                                                                 self.left_arm_collision_table,
                                                                                 vrep.simx_opmode_buffer)
        if self.left_arm_collision_state_target or self.left_arm_collision_state_table:
            return True
        else:
            return False

    def get_input_image(self):
        error_code, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.input_cam, 0,
                                                                      vrep.simx_opmode_buffer)
        image = np.array(image, dtype=np.uint8)
        image.resize([resolution[0], resolution[1], 3])
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        return image

    def get_video_image(self):
        error_code, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.video_cam, 0,
                                                                      vrep.simx_opmode_buffer)
        image = np.array(image, dtype=np.uint8)
        image.resize([resolution[0], resolution[1], 3])
        image = cv2.rotate(image, cv2.ROTATE_180)
        return image

    def display_image(self):
        image = self.get_input_image()
        plt.imshow(image)

    def reset_sim(self):
        for x in range(0, 7):
            vrep.simxSetJointTargetPosition(self.clientID, self.right_joint_array[x], self.right_joint_org_position[x],
                                            vrep.simx_opmode_oneshot_wait)
            vrep.simxSetJointTargetPosition(self.clientID, self.left_joint_array[x], self.left_joint_org_position[x],
                                            vrep.simx_opmode_oneshot_wait)
        time.sleep(1)

    def full_sim_reset(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        is_running = True
        while is_running:
            error_code, ping_time = vrep.simxGetPingTime(self.clientID)
            error_code, server_state = vrep.simxGetInMessageInfo(self.clientID, vrep.simx_headeroffset_server_state)
            if server_state == 0:
                is_running = False
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)
        time.sleep(1)


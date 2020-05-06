import math


def arm_distance_reward(right_dist, left_dist):
    """
    Calculates the average distance of the right and left end joints to a given target.
    :param left_dist: distance of left arm to target
    :param right_dist: distance of right arm to target
    :return: average of the right and left arm distances
    """
    return (left_dist + right_dist)/2


def target_movement_reward(start_loc, end_loc,  target_loc):
    """
    Calculates the distance an object has moved from an (x,y,z) position to another (x,y,z) position.
    :param start_loc: (x,y,z) float values, initial location in 3D plane
    :param end_loc:  (x,y,z) float values, final location in 3D plane
    :param target_loc: (x,y,z) float values, target location in 3D plane
    :return:
    """
    # The distance moved in the step
    (x1, y1, z1) = start_loc
    (x2, y2, z2) = end_loc
    (x3, y3, z3) = target_loc

    distance_moved = math.sqrt(math.pow(x2 - x1, 2) +
                               math.pow(y2 - y1, 2) +
                               math.pow(z2 - z1, 2) * 1.0)
    distance_to_target = math.sqrt(math.pow(x3 - x2, 2) +
                                   math.pow(y3 - y2, 2) +
                                   math.pow(z3 - z2, 2) * 1.0)

    # this will provide the largest rewards for actions that move the largest distances and end closer to the goal
    # two actions that end at the same point will be rated differently, the large action towards the point will recieve
    # a higher reward, this will promote larger actions and demote smaller actions over time.
    reward = distance_moved / distance_to_target
    return reward

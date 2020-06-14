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

    # starting location to target location distance
    initial_distance = math.sqrt(math.pow(x3 - x1, 2) +
                                 math.pow(y3 - y1, 2) +
                                 math.pow(z3 - z1, 2) * 1.0)
    # ending location to target location distance
    new_distance = math.sqrt(math.pow(x3 - x2, 2) +
                             math.pow(y3 - y2, 2) +
                             math.pow(z3 - z2, 2) * 1.0)

    # difference between starting distance and ending distance from the target
    reward = initial_distance - new_distance
    return reward, new_distance

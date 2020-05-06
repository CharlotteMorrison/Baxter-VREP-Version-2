import keyboard  # using module keyboard
from vrepsim import VrepSim


if __name__ == '__main__':

    sim = VrepSim()
    sim.reset_sim()

    # right, left = sim.calc_distance()

    # sim.step_right([.8, 0, 0, 0, 0, 0, 0])
    # sim.step_left([-1, .3, .1, 0, 0, 0, -.1])  # yields a collision

    sim.step_right([.1, 0, 0, 0, 0, 0, 0])
    sim.step_right([-.1, 0, 0, 0, 0, 0, 0])
    sim.step_right([0, .1, 0, 0, 0, 0, 0])
    sim.step_right([0, -.1, 0, 0, 0, 0, 0])
    sim.step_right([0, 0, .1, 0, 0, 0, 0])
    sim.step_right([0, 0, -.1, 0, 0, 0, 0])
    sim.step_right([0, 0, 0, .1, 0, 0, 0])
    sim.step_right([0, 0, 0, -.1, 0, 0, 0])
    sim.step_right([0, 0, 0, 0, .1, 0, 0])
    sim.step_right([0, 0, 0, 0, -.1, 0, 0])
    sim.step_right([0, 0, 0, 0, 0, .1, 0])
    sim.step_right([0, 0, 0, 0, 0, -.1, 0])
    sim.step_right([0, 0, 0, 0, 0, 0, .1])
    sim.step_right([0, 0, 0, 0, 0, 0, -.1])

    # move left joints

    sim.step_left([.1, 0, 0, 0, 0, 0, 0])
    sim.step_left([-.1, 0, 0, 0, 0, 0, 0])
    sim.step_left([0, .1, 0, 0, 0, 0, 0])
    sim.step_left([0, -.1, 0, 0, 0, 0, 0])
    sim.step_left([0, 0, .1, 0, 0, 0, 0])
    sim.step_left([0, 0, -.1, 0, 0, 0, 0])
    sim.step_left([0, 0, 0, .1, 0, 0, 0])
    sim.step_left([0, 0, 0, -.1, 0, 0, 0])
    sim.step_left([0, 0, 0, 0, .1, 0, 0])
    sim.step_left([0, 0, 0, 0, -.1, 0, 0])
    sim.step_left([0, 0, 0, 0, 0, .1, 0])
    sim.step_left([0, 0, 0, 0, 0, -.1, 0])
    sim.step_left([0, 0, 0, 0, 0, 0, .1])
    sim.step_left([0, 0, 0, 0, 0, 0, -.1])

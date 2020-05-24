from vrepsim import VrepSim
import time

if __name__ == '__main__':

    sim = VrepSim()
    sim.reset_sim()
    time.sleep(1)

    # demo both arms lifting
    sim.step_arms([.605, .45, 0, 0, -1.6, -1.5, 0], [-.605, .45, 0, 0, 1.6, -1.5, 0])
    time.sleep(2)
    sim.step_arms([0, -.6, 0, 0, 0, 0, 0], [0, -.6, 0, 0, 0, 0, 0])
    time.sleep(5)

    # demo one arm lifting
    sim.reset_sim()
    time.sleep(1)
    sim.step_arms([.608, .45, 0, 0, -1.6, -1.5, 0], [0, 0, 0, 0, 0, 0, 0])
    time.sleep(2)
    sim.step_arms([0, -.6, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0])

    sim.reset_sim()


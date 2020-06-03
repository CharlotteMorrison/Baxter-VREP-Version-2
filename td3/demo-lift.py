from vrepsim import VrepSim
import time

if __name__ == '__main__':

    sim = VrepSim()
    sim.reset_sim()
    time.sleep(1)
    print(sim.get_target_angles())
    # demo both arms lifting
    sim.step_arms([0, .45, 0, 0, -1.6, -1.5, 0], [0, .45, 0, 0, 1.6, -1.5, 0])
    sim.step_arms([.65, 0, 0, 0, 0, 0, 0], [-.65, 0, 0, 0, 0, 0, 0])
    time.sleep(1)
    sim.step_arms([0, -.6, 0, 0, 0, 0, 0], [0, -.6, 0, 0, 0, 0, 0])


    '''
    # demo one arm lifting
    sim.reset_sim()
    time.sleep(1)
    print(sim.get_target_angles())
    sim.step_arms([.608, .45, 0, 0, -1.6, -1.5, 0], [0, 0, 0, 0, 0, 0, 0])
    print(sim.get_target_angles())
    time.sleep(2)
    sim.step_arms([0, -.1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0])
    print(sim.get_target_angles())
    '''
    sim.reset_sim()


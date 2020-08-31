from vrepsim import VrepSim
import time

if __name__ == '__main__':
    # [0.231, 0.105, 1.261]
    sim = VrepSim()
    # sim.check_suction_prox()
    # time.sleep(5)
    sim.reset_sim()
    # time.sleep(1)
    print(sim.get_target_position())
    # demo both arms lifting
    # sim.step_arms([0, .45, 0, 0, -1.6, -1.5, 0], [0, .45, 0, 0, 1.6, -1.5, 0])


    # sim.check_suction_prox()
    # sim.step_arms([1., 0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0, 0])

    # time.sleep(1)
    # sim.check_suction_prox()
    sim.step_arms([0, -.4, 0, 0, 0, 0, 0], [0, -.4, 0, 0, 0, 0, 0])
    # sim.check_suction_prox()
    print(sim.get_target_position())
    time.sleep(1)
    sim.step_arms([.45, 0, 0, 0, 0, -1, 0], [.4, 0, 0, 0, 0, -1, 0])
    time.sleep(1)
    print(sim.get_target_position())

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
    # sim.reset_sim()


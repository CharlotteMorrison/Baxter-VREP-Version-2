from vrepsim import VrepSim
import time

if __name__ == '__main__':

    sim = VrepSim()
    sim.reset_sim()
    time.sleep(3)
    dist = sim.check_suction_distance()
    print(dist)





from vrepsim import VrepSim
import time
from evaluate import evaluate_policy
from td3.td3_shared_critic import TD3SharedCritic
from td3.td3 import TD3


if __name__ == '__main__':

    agent = TD3()
    sim = VrepSim()
    sim.reset_sim()
    agent.load()
    evaluate_policy(agent, sim)

    sim.reset_sim()


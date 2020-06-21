from vrepsim import VrepSim
import time
from evaluate import evaluate_policy
from td3.td3_shared_critic import TD3SharedCritic


if __name__ == '__main__':

    agent = TD3SharedCritic()
    sim = VrepSim()
    sim.reset_sim()
    agent.load()
    evaluate_policy(agent, sim)

    sim.reset_sim()


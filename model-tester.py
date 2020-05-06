from vrepsim import VrepSim
import time
from evaluate import evaluate_policy
from td3.td3 import TD3


if __name__ == '__main__':

    agent = TD3()
    sim = VrepSim()
    sim.reset_sim()
    agent.load()
    evaluate_policy(agent, sim, eval_episodes=10, episode_length=50)

    sim.reset_sim()


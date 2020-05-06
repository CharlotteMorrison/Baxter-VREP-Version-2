from vrepsim import VrepSim
from td3.td3 import TD3
from evaluate import evaluate_policy

if __name__ == '__main__':

    agent = TD3()

    sim = VrepSim()
    sim.reset_sim()

    # possible evaluation
    agent.load()
    evaluate_policy(agent, sim, eval_episodes=10, episode_length=100)

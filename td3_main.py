import numpy as np
import torch
from vrepsim import VrepSim
import td3.constants as cons
from evaluate import evaluate_policy
from td3.experience.priority_replay_buffer import PrioritizedReplayBuffer
from td3.experience.replay_buffer import ReplayBuffer
from td3.td3 import TD3
from td3.td3_shared_critic import TD3SharedCritic
from td3.train import train
from td3.populate import populate_buffer
from td3.experience.schedules import LinearSchedule


if __name__ == '__main__':

    # Set seeds
    if cons.set_seed:
        torch.manual_seed(cons.SEED)
        np.random.seed(cons.SEED)
    if cons.MODE == 'cooperative':
        agent = TD3()
    elif cons.MODE == 'independent':
        agent = TD3SharedCritic()
    else:  # can add fully independent here...
        agent = TD3()

    sim = VrepSim()
    sim.reset_sim()

    if cons.PRIORITY:
        replay_buffer = PrioritizedReplayBuffer(cons.BUFFER_SIZE, alpha=cons.ALPHA)
        if cons.BETA_ITERS is None:
            cons.BETA_ITERS = cons.EXPLORATION
        cons.BETA_SCHED = LinearSchedule(cons.BETA_ITERS, initial_p=cons.BETA, final_p=1.0)
    else:
        replay_buffer = ReplayBuffer()

    # initialize_min_buffer(sim, replay_buffer)
    populate_buffer(sim, replay_buffer)
    train(agent, sim, replay_buffer)

    # possible evaluation

    if cons.MODE == 'cooperative':
        agent.load(directory="td3/saves/dual_agent")
    elif cons.MODE == 'independent':
        agent.load(directory="td3/saves/shared_agent")

    evaluate_policy(agent, sim)

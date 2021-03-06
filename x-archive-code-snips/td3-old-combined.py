import constants as cons
from actor import Actor
from critic import Critic
import numpy as np
import torch
import torch.nn.functional as F


class TD3(object):
    """Agent class that handles the training of the networks
    and provides outputs as actions.
    """

    def __init__(self):
        state_dim = cons.STATE_DIM.flatten().shape[0]
        action_dim = cons.ACTION_DIM
        self.actor = Actor(state_dim, action_dim, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target = Actor(state_dim,  action_dim, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)  # or 1e-3

        self.critic = Critic(state_dim,  action_dim).to(cons.DEVICE)
        self.critic_target = Critic(state_dim,  action_dim).to(cons.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)  # or 1e-3

        self.total_it = 0
        self.critic_loss_plot = []
        self.actor_loss_plot = []

    def select_action(self, state, noise=cons.POLICY_NOISE):
        """Select an appropriate action from the agent policy
            Args:
                state (array): current state of environment
                noise (float): how much noise to add to actions
            Returns:
                action (list): nn action results
        """
        state = torch.FloatTensor(state).to(cons.DEVICE)
        action = self.actor(state)
        # action space noise introduces noise to change the likelihoods of each action the agent might take
        if noise != 0:
            # creates tensor of gaussian noise
            noise = torch.clamp(torch.randn(14, dtype=torch.float32, device='cuda') * noise,
                                min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
        action = action + noise
        torch.clamp(action, min=cons.MIN_ACTION, max=cons.MAX_ACTION)
        return action

    def train(self, replay_buffer, iterations):
        """Train and update actor and critic networks
            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                iterations (int): how many times to run training
            Return:
                actor_loss (float): loss from actor network
                critic_loss (float): loss from critic network
        """
        for it in range(iterations):
            self.total_it += 1  # keep track of the total training iterations
            # Sample replay buffer (priority replay)
            # choose type of replay
            if cons.PRIORITY:
                state, action, reward, next_state, done, weights, indexes = replay_buffer.sample(cons.BATCH_SIZE,
                                                                                     beta=cons.BETA_SCHED.value(it))
            else:
                state, action, reward, next_state, done = replay_buffer.sample(cons.BATCH_SIZE)

            state = torch.from_numpy(state).float().to(cons.DEVICE)                 # torch.Size([100, 14])
            next_state = torch.from_numpy(next_state).float().to(cons.DEVICE)       # torch.Size([100, 14])
            action = torch.from_numpy(action).float().to(cons.DEVICE)               # torch.Size([100, 14])
            reward = torch.as_tensor(reward, dtype=torch.float32).to(cons.DEVICE)   # torch.Size([100])
            done = torch.as_tensor(done, dtype=torch.float32).to(cons.DEVICE)       # torch.Size([100])

            with torch.no_grad():
                # select an action according to the policy and add clipped noise
                next_action = self.actor_target(next_state)
                noise = torch.clamp(torch.randn((100, 14), dtype=torch.float32, device='cuda') * cons.POLICY_NOISE,
                                    min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
                next_action = torch.clamp((next_action + noise), min=cons.MIN_ACTION, max=cons.MAX_ACTION)

                # Compute the target Q value
                target_q1, target_q2 = self.critic(state.float(), next_action.float())
                target_q = torch.min(target_q1, target_q2)
                gamma = torch.ones((100, 1), dtype=torch.float32, device='cuda')
                gamma = gamma.new_full((100, 1), cons.GAMMA)
                target_q = reward.unsqueeze(1) + (done.unsqueeze(1) * gamma * target_q).detach()

            # get current Q estimates
            current_q1, current_q2 = self.critic(state.float(), action.float())

            # compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            cons.TD3_REPORT.write_critic_loss(self.total_it, it, critic_loss)

            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # using the minimum of the q values as the weight, use min to prevent overestimation
            if cons.PRIORITY:
                new_priorities = torch.flatten(torch.min(current_q1, current_q2))
                # convert any negative priorities to a minimum value, can't have a negative priority
                new_priorities = torch.clamp(new_priorities, min=0.0000001).tolist()  # convert to a list for storage
                replay_buffer.update_priorities(indexes, new_priorities)

            # delayed policy updates
            if it % cons.POLICY_FREQ == 0:  # update the actor policy less frequently

                # compute the actor loss
                q_action = self.actor(state).float().detach()
                actor_loss = -self.critic.get_q(state, q_action).mean()
                cons.TD3_REPORT.write_actor_loss(self.total_it, it, actor_loss, 1)

                # optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_loss_plot.append(actor_loss.item())

                # Update the frozen right_target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="td3/saves/shared_agent"):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

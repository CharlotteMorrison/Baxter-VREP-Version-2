import td3.constants as cons
from td3.actor import Actor
from td3.critic import Critic
import numpy as np
import torch
import torch.nn.functional as F


class TD3SharedCritic(object):
    """Agent class that handles the training of the networks
    and provides outputs as actions.
    """
    # notes- still using 1 replay buffer
    # both actors are trained in each step together, right then left
    #
    def __init__(self):
        state_dim = cons.STATE_DIM.flatten().shape[0]
        self.action_dim_actor = cons.ACTION_DIM / 2
        self.action_dim_critic = cons.ACTION_DIM

        # actor 1 right arm
        self.actor_1 = Actor(state_dim, self.action_dim_actor, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_1 = Actor(state_dim,  self.action_dim_actor, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_1.load_state_dict(self.actor_1.state_dict())
        self.actor_optimizer_1 = torch.optim.Adam(self.actor_1.parameters(), lr=3e-4)  # or 1e-3

        # actor 2 left arm
        self.actor_2 = Actor(state_dim, self.action_dim_actor, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_2 = Actor(state_dim,  self.action_dim_actor, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_2.load_state_dict(self.actor_2.state_dict())
        self.actor_optimizer_2 = torch.optim.Adam(self.actor_2.parameters(), lr=3e-4)  # or 1e-3

        # shared critic
        self.critic = Critic(state_dim,  self.action_dim_critic).to(cons.DEVICE)
        self.critic_target = Critic(state_dim,  self.action_dim_critic).to(cons.DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)  # or 1e-3

        self.total_it = 0
        self.critic_loss_plot = []
        self.actor_loss_plot_1 = []
        self.actor_loss_plot_2 = []

    def select_action(self, state, actor,  noise=cons.POLICY_NOISE):
        """Select an appropriate action from the agent policy
            Args:
                state (array): current state of environment
                noise (float): how much noise to add to actions
                actor: right (actor_1) left (actor_2)
            Returns:
                action (list): nn action results
        """

        state = torch.FloatTensor(state).to(cons.DEVICE)
        if actor == "right":
            action = self.actor_1(state)
        else:
            action = self.actor_2(state)
        # action space noise introduces noise to change the likelihoods of each action the agent might take
        if noise != 0:
            # creates tensor of gaussian noise
            noise = torch.clamp(torch.randn(self.action_dim_actor, dtype=torch.float32, device='cuda') * noise,
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
                actor_loss_1 (float): loss from actor network 1
                actor_loss_2 (float): loss from actor network 2
                critic_loss (float): loss from critic network
        """
        for it in range(iterations):
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

            # split the state, next_state, and action into 2 stored in a list
            states = torch.chunk(state, 2, 1)
            next_states = torch.chunk(next_state, 2, 1)
            actions = torch.chunk(action, 2, 1)

            with torch.no_grad():
                # select an action according to the policy and add clipped noise
                next_action_1 = self.actor_target_1(next_states[0])
                next_action_2 = self.actor_target_2(next_states[1])

                noise_1 = torch.clamp(torch.randn((100, 14), dtype=torch.float32, device='cuda') * cons.POLICY_NOISE,
                                      min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
                next_action_1 = torch.clamp((next_action_1 + noise_1), min=cons.MIN_ACTION, max=cons.MAX_ACTION)

                noise_2 = torch.clamp(torch.randn((100, 7), dtype=torch.float32, device='cuda') * cons.POLICY_NOISE,
                                      min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
                next_action_2 = torch.clamp((next_action_2 + noise_2), min=cons.MIN_ACTION, max=cons.MAX_ACTION)



                # Compute the target Q value
                next_action = torch.cat((next_action_1, next_action_2), 1)
                target_q1, target_q2 = self.critic(state.float(), next_action.float())
                target_q = torch.min(target_q1, target_q2)
                gamma = torch.ones((100, 1), dtype=torch.float32, device='cuda')
                gamma = gamma.new_full((100, 1), cons.GAMMA)
                target_q = reward.unsqueeze(1) + (done.unsqueeze(1) * gamma * target_q).detach()

            # get current Q estimates
            current_q1, current_q2 = self.critic(state.float(), action.float())

            # compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

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
            if self.total_it % cons.POLICY_FREQ == 0:  # update the actor policy less frequently

                # compute the actor 1 loss
                q_action_1 = self.actor_1(states[0]).float().detach()
                q_action_2 = self.actor_2(states[1]).float().detach()

                actor_loss_1 = -self.critic.get_q(states[0], q_action_1).mean()
                actor_loss_2 = -self.critic.get_q(states[1], q_action_2).mean()

                # optimize the actor
                self.actor_optimizer_1.zero_grad()
                self.actor_optimizer_2.zero_grad()
                actor_loss_1.backward()
                actor_loss_2.backward()
                self.actor_optimizer_1.step()
                self.actor_optimizer_2.step()

                # store the loss
                self.actor_loss_plot_1.append(actor_loss_1.item())
                self.actor_loss_plot_2.append(actor_loss_2.item())

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                for param, target_param in zip(self.actor_1.parameters(), self.actor_target_1.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                for param, target_param in zip(self.actor_2.parameters(), self.actor_target_2.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor_1.state_dict(), '%s/%s_actor_1.pth' % (directory, filename))
        torch.save(self.actor_2.state_dict(), '%s/%s_actor_2.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="td3/saves"):
        self.actor_1.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_2.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

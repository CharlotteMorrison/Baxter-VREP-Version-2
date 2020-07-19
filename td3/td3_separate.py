import td3.constants as cons
from td3.actor import Actor
from td3.critic import Critic
import numpy as np
import torch
import torch.nn.functional as F


class TD3Separate(object):
    """Agent class that handles the training of the networks
    and provides outputs as actions.
    """
    #
    def __init__(self):

        self.state_dim = int(cons.STATE_DIM.flatten().shape[0] / 2)
        self.action_dim = int(cons.ACTION_DIM / 2)

        # actor 1 right arm
        self.actor_1 = Actor(self.state_dim, self.action_dim , cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_1 = Actor(self.state_dim,  self.action_dim, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_1.load_state_dict(self.actor_1.state_dict())
        self.actor_optimizer_1 = torch.optim.Adam(self.actor_1.parameters(), lr=3e-4)  # or 1e-3

        # actor 2 left arm
        self.actor_2 = Actor(self.state_dim, self.action_dim , cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_2 = Actor(self.state_dim,  self.action_dim , cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_2.load_state_dict(self.actor_2.state_dict())
        self.actor_optimizer_2 = torch.optim.Adam(self.actor_2.parameters(), lr=3e-4)  # or 1e-3

        # critic 1 right arm
        self.critic_1 = Critic(self.state_dim,  self.action_dim).to(cons.DEVICE)
        self.critic_1_target = Critic(self.state_dim,  self.action_dim).to(cons.DEVICE)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)  # or 1e-3

        # critic 2 left arm
        self.critic_2 = Critic(self.state_dim,  self.action_dim).to(cons.DEVICE)
        self.critic_2_target = Critic(self.state_dim,  self.action_dim).to(cons.DEVICE)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)  # or 1e-3

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
            noise = torch.clamp(torch.randn(self.action_dim, dtype=torch.float32, device='cuda') * noise,
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
            self.total_it += 1
            if cons.PRIORITY:
                state, action, reward, next_state, done, weights, indexes = replay_buffer.sample(cons.BATCH_SIZE,
                                                                                                 beta=cons.BETA_SCHED.value(it))
            else:
                state, action, reward, next_state, done = replay_buffer.sample(cons.BATCH_SIZE)
                indexes = 0

            state = torch.from_numpy(state).float().to(cons.DEVICE)                 # torch.Size([100, 14])
            next_state = torch.from_numpy(next_state).float().to(cons.DEVICE)       # torch.Size([100, 14])
            action = torch.from_numpy(action).float().to(cons.DEVICE)               # torch.Size([100, 14])
            reward = torch.as_tensor(reward, dtype=torch.float32).to(cons.DEVICE)   # torch.Size([100])
            done = torch.as_tensor(done, dtype=torch.float32).to(cons.DEVICE)       # torch.Size([100])

            # split the state, next_state, and action into 2 stored in a list
            states = torch.chunk(state, 2, 1)
            next_states = torch.chunk(next_state, 2, 1)
            #this is split on the wrong axis
            actions = torch.chunk(action, 2, 1)

            with torch.no_grad():
                # select an action according to the policy and add clipped noise
                next_action_1 = self.actor_target_1(next_states[0])
                next_action_2 = self.actor_target_2(next_states[1])

                noise_1 = torch.clamp(torch.randn((100, 7), dtype=torch.float32, device='cuda') * cons.POLICY_NOISE,
                                      min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
                next_action_1 = torch.clamp((next_action_1 + noise_1), min=cons.MIN_ACTION, max=cons.MAX_ACTION)

                noise_2 = torch.clamp(torch.randn((100, 7), dtype=torch.float32, device='cuda') * cons.POLICY_NOISE,
                                      min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
                next_action_2 = torch.clamp((next_action_2 + noise_2), min=cons.MIN_ACTION, max=cons.MAX_ACTION)

                # Compute the target Q value critic 1
                target_1_q1, target_1_q2 = self.critic_1(states[0].float(), next_action_1.float())
                target_1_q = torch.min(target_1_q1, target_1_q2)
                gamma_1 = torch.ones((100, 1), dtype=torch.float32, device='cuda')
                gamma_1 = gamma_1.new_full((100, 1), cons.GAMMA)
                target_1_q = reward.unsqueeze(1) + (done.unsqueeze(1) * gamma_1 * target_1_q).detach()

                # Compute the target Q value critic 2
                target_2_q1, target_2_q2 = self.critic_2(states[1].float(), next_action_2.float())
                target_2_q = torch.min(target_2_q1, target_2_q2)
                gamma_2 = torch.ones((100, 1), dtype=torch.float32, device='cuda')
                gamma_2 = gamma_2.new_full((100, 1), cons.GAMMA)
                target_2_q = reward.unsqueeze(1) + (done.unsqueeze(1) * gamma_2 * target_2_q).detach()

            # get current Q estimates
            current_1_q1, current_1_q2 = self.critic_1(states[0].float(), actions[0].float())
            current_2_q1, current_2_q2 = self.critic_2(states[1].float(), actions[1].float())

            # compute critic loss
            critic_1_loss = F.mse_loss(current_1_q1, target_1_q) + F.mse_loss(current_1_q2, target_1_q)
            critic_2_loss = F.mse_loss(current_2_q1, target_2_q) + F.mse_loss(current_2_q2, target_2_q)
            cons.TD3_REPORT.write_critic_loss(self.total_it, it, critic_1_loss, critic_num=1)
            cons.TD3_REPORT.write_critic_loss(self.total_it, it, critic_2_loss, critic_num=2)

            # optimize the critics
            self.critic_1_optimizer.zero_grad()
            self.critic_2_optimizer.zero_grad()
            critic_1_loss.backward()
            critic_2_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.step()

            # using the minimum of the q values as the weight, use min to prevent overestimation
            if cons.PRIORITY:
                new_priorities = torch.flatten(torch.min(torch.min(current_1_q1, current_1_q2),
                                                         torch.min(current_2_q1, current_2_q2)))
                # convert any negative priorities to a minimum value, can't have a negative priority
                new_priorities = torch.clamp(new_priorities, min=0.0000001).tolist()  # convert to a list for storage
                replay_buffer.update_priorities(indexes, new_priorities)

            # delayed policy updates
            if it % cons.POLICY_FREQ == 0:  # update the actor policy less frequently

                # compute the actor loss
                q_action_1 = self.actor_1(states[0]).float().detach()
                q_action_2 = self.actor_2(states[1]).float().detach()
                # q_action = torch.cat((q_action_1, q_action_2), 1)
                actor_loss_1 = -self.critic_1.get_q(states[0], q_action_1).mean()
                actor_loss_2 = -self.critic_2.get_q(states[1], q_action_2).mean()
                cons.TD3_REPORT.write_actor_loss(self.total_it, it, actor_loss_1, 1)
                cons.TD3_REPORT.write_actor_loss(self.total_it, it, actor_loss_2, 2)

                # optimize the actors
                self.actor_optimizer_1.zero_grad()
                actor_loss_1.backward(retain_graph=True)
                self.actor_optimizer_1.step()
                self.actor_loss_plot_1.append(actor_loss_1.item())

                self.actor_optimizer_2.zero_grad()
                actor_loss_2.backward(retain_graph=True)
                self.actor_optimizer_2.step()
                self.actor_loss_plot_2.append(actor_loss_2.item())

                # Update the frozen right_target models
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                for param, target_param in zip(self.actor_1.parameters(), self.actor_target_2.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                for param, target_param in zip(self.actor_2.parameters(), self.actor_target_2.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor_1.state_dict(), '%s/%s_actor_1.pth' % (directory, filename))
        torch.save(self.actor_2.state_dict(), '%s/%s_actor_2.pth' % (directory, filename))
        torch.save(self.critic_1.state_dict(), '%s/%s_critic_1.pth' % (directory, filename))
        torch.save(self.critic_2.state_dict(), '%s/%s_critic_2.pth' % (directory, filename))

    def load(self, filename="best_avg", directory="td3/saves/dual_agent/"):
        self.actor_1.load_state_dict(torch.load('%s/%s_actor_1.pth' % (directory, filename)))
        self.actor_2.load_state_dict(torch.load('%s/%s_actor_2.pth' % (directory, filename)))
        self.critic_1.load_state_dict(torch.load('%s/%s_critic_1.pth' % (directory, filename)))
        self.critic_2.load_state_dict(torch.load('%s/%s_critic_2.pth' % (directory, filename)))

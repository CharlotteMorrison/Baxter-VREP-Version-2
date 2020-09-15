import td3.constants as cons
from td3.actor import Actor
from td3.critic import Critic
import torch
import torch.nn.functional as F
import globals as glo
import file_names as names
import gc


class TD3(object):
    """Agent class that handles the training of the networks
    and provides outputs as actions.
    """
    #
    def __init__(self, mode):
        state_dim = cons.STATE_DIM.flatten().shape[0]  # 14
        action_dim = cons.ACTION_DIM                   # 14
        self.mode = mode

        # setup for each mode case:
        if self.mode == 'cooperative':        # cooperative (TD3)
            # uses actor_1 and critic_1, define the dimensions- is repetitive, but keeps this organized
            # for me so I don't forget to have these values defined.
            self.state_dim_a1 = self.state_dim_c1 = state_dim
            self.action_dim_a1 = self.action_dim_c1 = action_dim

        elif self.mode == 'partial':           # partial (td3_shared_critic)
            # Uses actor_1 and actor_2 only uses critic_1
            self.state_dim_a1 = self.state_dim_a2 = int(state_dim / 2)
            self.action_dim_a1 = self.action_dim_a2 = int(action_dim / 2)
            self.state_dim_c1 = state_dim
            self.action_dim_c1 = action_dim

        elif self.mode == 'independent':       # independent (td3separate)
            # needs half the state dims of the cooperative
            self.state_dim_a1 = self.state_dim_a2 = int(state_dim / 2)
            self.action_dim_a1 = self.action_dim_a2 = int(action_dim / 2)
            self.state_dim_c1 = self.state_dim_c2 = int(state_dim / 2)
            self.action_dim_c1 = self.action_dim_c2 = int(action_dim / 2)
        else:
            print('incorrect mode')

        # -----------------------------------------------------------------------------------------------------
        # Setup the actor/critic networks
        # -----------------------------------------------------------------------------------------------------

        # actor 1: shared actor or right arm
        self.actor_1 = Actor(self.state_dim_a1, self.action_dim_a1, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_1 = Actor(self.state_dim_a1,  self.action_dim_a1, cons.MAX_ACTION).to(cons.DEVICE)
        self.actor_target_1.load_state_dict(self.actor_1.state_dict())
        self.actor_optimizer_1 = torch.optim.Adam(self.actor_1.parameters(),  lr=1e-3)  # or 3e-4

        if mode is not 'cooperative':
            # actor 2: left arm
            self.actor_2 = Actor(self.state_dim_a2, self.action_dim_a2, cons.MAX_ACTION).to(cons.DEVICE)
            self.actor_target_2 = Actor(self.state_dim_a2,  self.action_dim_a2, cons.MAX_ACTION).to(cons.DEVICE)
            self.actor_target_2.load_state_dict(self.actor_2.state_dict())
            self.actor_optimizer_2 = torch.optim.Adam(self.actor_2.parameters(), lr=1e-3)  # or 3e-4

        # critic 1: shared critic or  right arm critic
        self.critic_1 = Critic(self.state_dim_c1,  self.action_dim_c1).to(cons.DEVICE)
        self.critic_target_1 = Critic(self.state_dim_c1,  self.action_dim_c1).to(cons.DEVICE)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(),  lr=1e-3)  # or 3e-4

        if mode is 'independent':
            # critic 2 left arm
            self.critic_2 = Critic(self.state_dim_c2,  self.action_dim_c2).to(cons.DEVICE)
            self.critic_target_2 = Critic(self.state_dim_c2,  self.action_dim_c2).to(cons.DEVICE)
            self.critic_target_2.load_state_dict(self.critic_2.state_dict())
            self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(),  lr=1e-3)  # or 3e-4

    def select_action(self, state, actor='combined', noise=cons.POLICY_NOISE):
        """Select an appropriate action from the agent policy
            Args:
                state (array): current state of environment
                noise (float): how much noise to add to actions
                actor:  if two actors- right (actor_1) left (actor_2), combined
            Returns:
                action (list): nn action results
        """
        state = torch.FloatTensor(state).to(cons.DEVICE)  # ignore the state param warning
        if actor == "left":
            action = self.actor_2(state).cpu()
        else:
            action = self.actor_1(state).cpu()
        # action space noise introduces noise to change the likelihoods of each action the agent might take
        if noise != 0:
            # creates tensor of gaussian noise, use action_dim_a1, if only 1, then it is action_dim_a1
            # if two, they are the same dimensions, so just use action_dim_a1
            noise = torch.clamp(torch.randn(self.action_dim_a1, dtype=torch.float32, device='cpu') * noise,
                                min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
        action = action + noise
        torch.clamp(action, min=cons.MIN_ACTION, max=cons.MAX_ACTION).cpu()

        del state, noise
        gc.collect()

        return action

    def train(self, replay_buffer, iterations):
        """Train and update actor and critic networks
            Args:
                replay_buffer (ReplayBuffer): buffer for experience replay
                iterations (int): how many times to run training
            Return:
                actor_loss_1 (float): loss from actor network 1 right arm, or combined both
                actor_loss_2 (float): loss from actor network 2 left arm
                critic_loss_1 (float): loss from critic network right arm, or combined both
                critic_loss_2 (float): loss from critic network left arm, or combined both
        """
        for it in range(iterations):
            # Sample replay buffer (priority replay)
            # choose type of replay
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
            split_state = torch.chunk(state, 2, 1)
            split_next_state = torch.chunk(next_state, 2, 1)
            # this is split on the wrong axis- # TODO figure out what this comment is about...
            split_action = torch.chunk(action, 2, 1)

            # with torch.no_grad():
            # TODO get creative and refactor this so it isn't so repetitive.
            # select an action according to the policy and add clipped noise
            if self.mode == 'cooperative':
                next_action_1 = self.actor_target_1(next_state)
                noise_1 = torch.clamp(torch.randn((100, self.action_dim_a1), dtype=torch.float32, device='cuda') *
                                      cons.POLICY_NOISE, min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
                next_action_1 = torch.clamp((next_action_1 + noise_1), min=cons.MIN_ACTION, max=cons.MAX_ACTION)
            else:
                next_action_1 = self.actor_target_1(split_next_state[0])
                next_action_2 = self.actor_target_2(split_next_state[1])
                noise_1 = torch.clamp(torch.randn((100, self.action_dim_a1), dtype=torch.float32, device='cuda') *
                                                   cons.POLICY_NOISE, min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
                next_action_1 = torch.clamp((next_action_1 + noise_1), min=cons.MIN_ACTION, max=cons.MAX_ACTION)
                noise_2 = torch.clamp(torch.randn((100, self.action_dim_a2), dtype=torch.float32, device='cuda') *
                                      cons.POLICY_NOISE, min=-cons.NOISE_CLIP, max=cons.NOISE_CLIP)
                next_action_2 = torch.clamp((next_action_2 + noise_2), min=cons.MIN_ACTION, max=cons.MAX_ACTION)

            # Compute the target Q value
            if self.mode != 'independent':  # partial and cooperative have only one critic
                if self.mode == 'partial':  # need to combine the action from both actors
                    next_action_1 = torch.cat((next_action_1, next_action_2), 1)
                target_1_q1, target_1_q2 = self.critic_1(state.float(), next_action_1.float())
                target_1_q = torch.min(target_1_q1, target_1_q2)
                gamma_1 = torch.ones((100, 1), dtype=torch.float32, device='cuda')
                gamma_1 = gamma_1.new_full((100, 1), cons.GAMMA)
                target_1_q = reward.unsqueeze(1) + (done.unsqueeze(1) * gamma_1 * target_1_q).detach()
            else:
                # Compute the target Q value critic 1
                target_1_q1, target_1_q2 = self.critic_1(split_state[0].float(), next_action_1.float())
                target_1_q = torch.min(target_1_q1, target_1_q2)
                gamma_1 = torch.ones((100, 1), dtype=torch.float32, device='cuda')
                gamma_1 = gamma_1.new_full((100, 1), cons.GAMMA)
                target_1_q = reward.unsqueeze(1) + (done.unsqueeze(1) * gamma_1 * target_1_q).detach()
                # Compute the target Q value critic 2
                target_2_q1, target_2_q2 = self.critic_2(split_state[1].float(), next_action_2.float())
                target_2_q = torch.min(target_2_q1, target_2_q2)
                gamma_2 = torch.ones((100, 1), dtype=torch.float32, device='cuda')
                gamma_2 = gamma_2.new_full((100, 1), cons.GAMMA)
                target_2_q = reward.unsqueeze(1) + (done.unsqueeze(1) * gamma_2 * target_2_q).detach()
            # get current Q estimates
            if self.mode != 'independent':
                current_1_q1, current_1_q2 = self.critic_1(state.float(), action.float())
                # compute critic loss
                critic_1_loss = F.mse_loss(current_1_q1, target_1_q) + F.mse_loss(current_1_q2, target_1_q)
                cons.report.write_report_critic(glo.EPISODE, glo.TIMESTEP, critic_1_loss)
                # optimize the critic
                self.critic_optimizer_1.zero_grad()
                critic_1_loss.backward()
                self.critic_optimizer_1.step()
                # just renaming to use later for priority weighting
                priority_q1 = current_1_q1
                priority_q2 = current_1_q2
            else:
                current_1_q1, current_1_q2 = self.critic_1(split_state[0].float(), split_action[0].float())
                current_2_q1, current_2_q2 = self.critic_2(split_state[1].float(), split_action[1].float())
                # compute critic loss
                critic_1_loss = F.mse_loss(current_1_q1, target_1_q) + F.mse_loss(current_1_q2, target_1_q)
                critic_2_loss = F.mse_loss(current_2_q1, target_2_q) + F.mse_loss(current_2_q2, target_2_q)

                # write to the report
                cons.report.write_report_critic(glo.EPISODE, glo.TIMESTEP, critic_1_loss, critic_2_loss)

                # optimize the critics
                self.critic_optimizer_1.zero_grad()
                self.critic_optimizer_2.zero_grad()
                critic_1_loss.backward()
                critic_2_loss.backward()
                self.critic_optimizer_1.step()
                self.critic_optimizer_2.step()

                # get the minimum from each critic for use in the priority
                priority_q1 = torch.min(current_1_q1, current_1_q2)
                priority_q2 = torch.min(current_2_q1, current_2_q2)

            # using the minimum of the q values as the weight, use min to prevent overestimation
            if cons.PRIORITY:
                new_priorities = torch.flatten(torch.min(priority_q1, priority_q2))
                # convert any negative priorities to a minimum value, can't have a negative priority
                new_priorities = torch.clamp(new_priorities, min=0.000000001).tolist()  # convert to a list for storage
                replay_buffer.update_priorities(indexes, new_priorities)

            # delayed policy updates
            if it % cons.POLICY_FREQ == 0:  # update the actor policy less frequently (2)

                if self.mode == 'cooperative':
                    q_action_1 = self.actor_1(state).float().detach()
                    actor_1_loss = -self.critic_1.get_q(state, q_action_1).mean()
                    cons.report.write_report_actor(glo.EPISODE, glo.TIMESTEP, actor_1_loss)

                    # optimize the actors
                    self.actor_optimizer_1.zero_grad()
                    actor_1_loss.backward()
                    self.actor_optimizer_1.step()
                else:
                    # compute the actor loss
                    q_action_1 = self.actor_1(split_state[0]).float().detach()
                    q_action_2 = self.actor_2(split_state[1]).float().detach()

                    if self.mode == 'combined':
                        actor_1_loss = -self.critic_1.get_q(split_state[0], q_action_1).mean()
                        actor_2_loss = -self.critic_2.get_q(split_state[1], q_action_2).mean()
                    else:
                        q_action = torch.cat((q_action_1, q_action_2), 1)
                        actor_1_loss = -self.critic_1.get_q(state, q_action).mean()
                        actor_2_loss = -self.critic_1.get_q(state, q_action).mean()
                        del q_action

                    cons.report.write_report_actor(glo.EPISODE, glo.TIMESTEP, actor_1_loss, actor_2_loss)

                    # optimize the actors
                    self.actor_optimizer_1.zero_grad()
                    actor_1_loss.backward()
                    self.actor_optimizer_1.step()

                    self.actor_optimizer_2.zero_grad()
                    actor_2_loss.backward()
                    self.actor_optimizer_2.step()

                # update the frozen target parameters
                for param, target_param in zip(self.actor_1.parameters(), self.actor_target_1.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                if self.mode != 'cooperative':
                    for param, target_param in zip(self.actor_2.parameters(), self.actor_target_2.parameters()):
                        target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                    target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                if self.mode == 'independent':
                    for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                        target_param.data.copy_(cons.TAU * param.data + (1 - cons.TAU) * target_param.data)

                # garbage collection, gpu storage is building, just remove all locals

                # independent, uses actor 1, actor 2, critic 1, critic 2
                if self.mode == 'independent':  # include only critic 2 variables
                    del target_2_q1, target_2_q2, target_2_q, gamma_2, current_2_q1, current_2_q2, critic_2_loss

                # partial, uses actor 1, actor 2, critic 1
                if self.mode == 'partial' or self.mode == 'independent':  # include only actor 2 variables
                    del next_action_2, noise_2, q_action_2, actor_2_loss

                # remove all the initial batches
                del state, next_state, action, reward, done, split_state, split_next_state, split_action

                # cooperative, uses actor 1 and critic 1
                # include all actor 1 and critic 1 variables
                del next_action_1, noise_1, target_1_q1, target_1_q2, target_1_q, gamma_1
                del current_1_q1, current_1_q2, critic_1_loss
                del q_action_1, actor_1_loss

                if cons.PRIORITY:
                    del new_priorities, indexes, priority_q1, priority_q2
                gc.collect()
                torch.cuda.empty_cache()

    def save(self):
        # everyone saves these two
        torch.save(self.actor_1.state_dict(), names.ACTOR_1)
        torch.save(self.critic_1.state_dict(), names.CRITIC_1)
        if self.mode == 'partial':
            torch.save(self.actor_2.state_dict(), names.ACTOR_2)
        elif self.mode == 'independent':
            torch.save(self.actor_2.state_dict(), names.ACTOR_2)
            torch.save(self.critic_2.state_dict(), names.CRITIC_2)

    def load(self):
        self.actor_1.load_state_dict(torch.load(names.ACTOR_1))
        self.critic_1.load_state_dict(torch.load(names.CRITIC_1))
        if self.mode == 'partial':
            self.actor_2.load_state_dict(torch.load(names.ACTOR_2))
        elif self.mode == 'independent':
            self.actor_2.load_state_dict(torch.load(names.ACTOR_2))
            self.critic_2.load_state_dict(torch.load(names.CRITIC_2))

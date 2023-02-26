N_EPISODES = 100_001  # 10_000
BATCH_SIZE = 32  # 128
DISCOUNT_FACTOR = 0.999
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 0.9999
TARGET_UPDATE = 10
REPLAY_MEMORY_SIZE = 5_000  # If I keep replay buffer (partly) after updating value function, I am OFF-POLICY. If I always generate new data based on my value function, I am ON-POLICY
# One step updates: Here, on- and off-policy are mathematically the same (magic)
# n-step updates: Using old data becomes mathematically wrong and is thus forbidden. You CAN make it off-policy, but then you
#    need to include likelihoods. You have to a) check with which likelihood your old policy chose an action. Then b) you check how
#    the likelihoos of your new policy look like. Now c) you have to weigh the new policy for off-policy learning. This is called
#    IMPORTANT sampling (re-weighing old data instead of recording new ones). There are mathematical issues with 0 or very small
#    likelihoods (very instable).

import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

from dqn_model import DQN
from dqn_model import DQN_DQN

from env_carla import N_ACTIONS

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(
            Transition(*args)
        )  # *args: pass a variable number of arguments to a function. It is used to pass a non-key worded, variable-length argument list.

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Training:
    def __init__(self, writer, device: torch.device, concatAE):
        self.device = device
        self.writer = writer

        if concatAE:
            self.policy_net = DQN_DQN().to(self.device)
            self.target_net = DQN_DQN().to(self.device)
        else:
            self.policy_net = DQN().to(self.device)
            self.target_net = DQN().to(self.device)

        self.target_net.eval()  # switch target_net to inference mode (normalisation layers use running statistics, de-activates Dropout layers)
        self.optimizer = optim.Adam(self.policy_net.parameters())  # was RMSprop
        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)  # (current_state, action, reward, new_state, done)

        iterations_until_EPS_Min = math.log(EPS_END / EPS_START) / math.log(EPS_DECAY)
        print(str(iterations_until_EPS_Min) + " iterations until EPS_END is reached")

    def select_action(self, state, epsilon):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():  # PyTorch will not calculate the gradients (makes all the operations in the block have no gradients)
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                policy_prediction = self.policy_net(state.to(self.device))
                max_pred = policy_prediction.max(1)[1]
                view = max_pred.view(1, 1)
                return view
        else:
            return torch.tensor([[random.randrange(N_ACTIONS)]], device=self.device, dtype=torch.long)

    def decay_epsilon(self, epsilon):
        if epsilon > EPS_END:
            epsilon *= EPS_DECAY
            epsilon = max(EPS_END, epsilon)
        return epsilon

    def optimize(self, epoch):
        if len(self.replay_memory) < BATCH_SIZE:
            return

        transitions = self.replay_memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))  

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # print("nfns: ", non_final_next_states.shape)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # print(non_final_mask.shape)             # I don't really get how these look like
        # print(non_final_next_states.shape)
        # print(state_batch.shape)

        # Compute Q(s_t, a)
        # These are the actions which would've been taken for each batch state according to policy_net
        q_current_list = self.policy_net(state_batch.to(self.device))  # Model computes Q(s_t)
        state_action_values = q_current_list.gather(
            1, action_batch
        )  # Gathers values along an axis (select columns of actions taken)

        # print(q_current_list)
        # print(action_batch)
        # print(state_action_values)

        # Compute max(Q(s_{t+1},a))
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.to(self.device)).max(1)[0].detach()
        # print(next_state_values)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch
        # print(reward_batch)
        # print(expected_state_action_values)
        # print("#####")
        # Compute Huber loss
        # The Huber loss acts like the mean squared error when the error is small,
        # but like the mean absolute error when the error is large.
        # this makes it more robust to outliers when the estimates of Q are very noisy

        # Minimize temporal difference error (prediction - target)
        # Bellmann euqation: Q_s_a = r + gamma * max_future_q
        # d = Q_s_a - (r + gamma * max_future_q)

        criterion = nn.HuberLoss()  # was SmoothL1Loss
        loss = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )  # x,y is used as x-y to compute d
        self.writer.add_scalar("Loss/train", loss, epoch)

        # Optimize the model
        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Perform backward pass

        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)           # Stabilize training (in a way "best practice", but mostly with gradient norm clipping)
        self.optimizer.step()


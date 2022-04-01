from msilib.schema import Directory
import numpy as np
import random
from numpy.core.numeric import indices
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from  torch.autograd import Variable
#from kerastuner.tuners import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameters
import time
from collections import namedtuple

LOG_DIR = f"models/{int(time.time())}"
#from tensorflow.keras.
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def softmax(x, temp):
    ''' Helper function from Helper.py of the first assignment '''
    x = x / temp
    z = x - max(x)
    return np.exp(z)/np.sum(np.exp(z))

def argmax(x):
    ''' Helper function from Helper.py of the first assignment '''
    try:
        return np.random.choice(np.where(x == np.max(x))[0])
    except:
        return np.argmax(x)

def linear_anneal(t,T,start,final,percentage):
    ''' Helper function from Helper.py of the first assignment
    Linear annealing scheduler
    t: current timestep
    T: total timesteps
    start: initial value
    final: value after percentage*T steps
    percentage: percentage of T after which annealing finishes
    '''
    final_from_T = int(percentage*T)
    if t > final_from_T:
        return final
    else:
        return final + (start - final) * (final_from_T - t)/final_from_T

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
    
        super(QNetwork, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)            
        """
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final(x)

        return x

class DQNagent():

    def __init__(self, state_shape, n_possible_actions, use_target = False, use_buffer = False, batch_size = 32, learning_rate = 0.01, future_reward_discount_factor = 0.95, exploration_parameter = 0.1, hidden_dim = 16):
        self.state_shape = state_shape
        self.n_possible_actions = n_possible_actions
        self.learning_rate = learning_rate
        self.gamma = future_reward_discount_factor
        self.epsilon = exploration_parameter
        self.memory = []
        self.hidden_dim = hidden_dim
        self.use_target = use_target
        self.use_buffer = use_buffer
        self.batch_size = batch_size
        self.model = QNetwork(state_shape, n_possible_actions, hidden_dim)
        if use_target:
            self.target_model = QNetwork(state_shape, n_possible_actions, hidden_dim)
        self.mse_loss = torch.nn.MSELoss()
        self.optim = optim.Adam(self.model.parameters(), lr=learning_rate)
    
        #self.tuner = RandomSearch(
        #    self._tune_model,
        #    objective = "accuracy",
        #    max_trials = 3,
        #    executions_per_trial = 1,
        #    directory = LOG_DIR
        #)
    
    def action_selection(self, state, method ="egreedy"):
        greedy_action = self.model(state).argmax().item()

        #Epsilon greedy
        if method == 'egreedy':
            if np.random.rand() < self.epsilon:
                #Take random action
                return np.random.randint(self.n_possible_actions)
            else:
                return greedy_action

        elif method == 'boltzmann':
  
            return argmax(softmax(self.model.predict(state)[0], self.epsilon))

        elif method == 'anneal_egreedy':
            epsilon = linear_anneal() # this needs more work
            if np.random.rand() < epsilon:
                #Take random action
                return np.random.randint(self.n_possible_actions)
            else:
                return greedy_action

        #Error catch
        else:
            raise KeyError("ERROR: not a valid method. Use: egreedy, boltzmann or anneal_egreedy")
    
    #def forward_pass(self, state, action = None):
    #    #Returns the network Q-value(s) of the given state (and action if specified)
    #    state = np.copy(state[np.newaxis, :])
    #    if self.use_target:
    #        if action == None:
    #            return self.target_model.predict(state)
    #        else:
    #            return self.target_model.predict(state)[0][action]
    #    else:
    #        if action == None:
    #            return self.model.predict(state)
    #        else:
    #            return self.model.predict(state)[0][action]

    def train(self):
        
        #When buffer is used we sample randomly from memory
        #When buffer isn't used the memory is just the last n=batch_size examples
        if self.use_buffer:
            transitions = random.sample(self.memory,self.batch_size)
        else:
            transitions = self.memory
        batch = Transition(*zip(*transitions))

        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)

        Q_max_action = self.model(next_states).detach().max(1)[1].unsqueeze(1)

        if self.use_target:
            Q_targets_next = self.target_model(next_states).gather(1, Q_max_action).reshape(-1)
        else:
            Q_targets_next = self.model(next_states).gather(1, Q_max_action).reshape(-1)
        
        # Compute the expected Q values
        Q_targets = rewards + (self.gamma * Q_targets_next * (1-dones))
        Q_expected = self.model(states).gather(1, actions) ## current 
        
        self.optim.zero_grad()
        loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))
        
        # backpropagation of loss to NN        
        loss.backward()
        self.optim.step()

    def memorize(self, state, action, reward, next_state, done):
        if self.use_buffer:
            if len(self.memory) >= 1e6:
                self.memory.pop(0)
        else:
            if len(self.memory) >= self.batch_size:
                self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, done))
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict()) #When to update, if using replay is it n/batch_size?


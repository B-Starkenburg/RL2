#from msilib.schema import Directory
import numpy as np
import random
from numpy.core.numeric import indices
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
#from kerastuner.tuners import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameters
import time
import sys
import argparse

LOG_DIR = f"models/{int(time.time())}"
#from tensorflow.keras.

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experience_replay', action='store_true', required=False, default=False)
parser.add_argument('-t', '--target_network',action='store_true',  required=False, default=False)

args = parser.parse_args()

TARGET_NETWORK = args.target_network
EXPERIENCE_REPLAY = args.experience_replay

print("Settings:")
print(f"Experience replay: {EXPERIENCE_REPLAY}")
print(f"Target network: {TARGET_NETWORK}")

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


class DQNagent():

    def __init__(self, state_shape, n_possible_actions, learning_rate = 0.01, future_reward_discount_factor = 0.95, exploration_parameter = 0.1, network_params = [(24,'relu'),(24,'relu')]):
        self.state_shape = state_shape
        self.n_possible_actions = n_possible_actions
        self.learning_rate = learning_rate
        self.gamma = future_reward_discount_factor
        self.epsilon = exploration_parameter
        self.memory = []
        self.network_params = network_params
        self.model = self._build_model()
        self.target_model = self._build_model()
    
        #self.tuner = RandomSearch(
        #    self._tune_model,
        #    objective = "accuracy",
        #    max_trials = 3,
        #    executions_per_trial = 1,
        #    directory = LOG_DIR
        #)

    def _build_model(self):
        #Build the neural network, might be better to have it outside of the class for experimentation...
        #Right now its just a simple one with keras but we will have to experiment with number of layers, nodes, activation function, learning rate etc.

        model = Sequential()
        model.add(Input(shape=self.state_shape))
        for par in self.network_params:
            model.add(Dense(par[0], activation=par[1]))
        model.add(Dense(self.n_possible_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def _tune_model(self, hp):
        #Build the neural network, might be better to have it outside of the class for experimentation...
        #Right now its just a simple one with keras but we will have to experiment with number of layers, nodes, activation function, learning rate etc.

        model = Sequential()
        model.add(Dense(hp.Int("input_units", 32, 256, 32), input_dim=self.state_shape, activation='relu'))
        for i in range(hp.Int("n_layers0", 1, 4)):
            model.add(Dense(hp.Int(f"dens_{i}_units0", 32, 256, 32), input_dim=self.state_shape, activation='relu'))
        model.add(Dense(self.n_possible_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model
    
    def action_selection(self, state, method ="egreedy", temp=None, curT = None, totT = None, startE = None, finalE = None, percent = None):
        greedy_action = np.argmax(self.model.predict(state)[0])

        #Epsilon greedy
        if method == 'egreedy':
            if np.random.rand() < self.epsilon:
                #Take random action
                return np.random.randint(self.n_possible_actions)
            else:
                return greedy_action
        elif method == 'boltzmann':
  
            return argmax(softmax(self.model.predict(state)[0], temp))

        elif method == 'anneal_egreedy':
            if curT is None or totT is None or startE is None or finalE is None or percent is None:
                raise KeyError("annealing is selected, but not all parameters are given")
            epsilon = linear_anneal(curT, totT, startE, finalE, percent)
            if np.random.rand() < epsilon:
                #Take random action
                return np.random.randint(self.n_possible_actions)
            else:
                return greedy_action

        #Error catch
        else:
            raise KeyError("ERROR: not a valid method. Use: egreedy, boltzmann or anneal_egreedy")
    
    def forward_pass(self, state, action = None, use_target_network = False):
        #Returns the network Q-value(s) of the given state (and action if specified)

        if use_target_network:
            if action == None:
                return self.target_model.predict(state)
            else:
                return self.target_model.predict(state)[0][action]
        else:
            if action == None:
                return self.model.predict(state)
            else:
                return self.model.predict(state)[0][action]
    
    def train(self, state, action, reward, next_state, done, use_target_network = False):
        
        #Calculate target    
        Q_values_target = self.forward_pass(state)
        if done:
            target = reward
        else:
            target = reward + self.gamma*np.max(self.forward_pass(next_state, use_target_network = use_target_network)[0]) #correct? not DDQN right (greedy action from original model)
        Q_values_target[0][action] = target

        #self.tuner.search(x = state, y = Q_values_target, epochs = 1, batch_size = 64)
        self.model.fit(state, Q_values_target, epochs = 1, verbose = 0)
    
    def memorize(self, state, action, reward, next_state, done):

        if len(self.memory) >= 1e6:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay_train(self, timestep, target_network_update_frequency, batch_size = 32, use_target_network = False):
        minibatch = random.sample(self.memory,batch_size)
        i = 0
        for state, action, reward, next_state, done in minibatch:
            self.train(state, action, reward, next_state, done, use_target_network)
            if (timestep*batch_size+i)%target_network_update_frequency == 0:
                self.update_target_network()
            i+=1

    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights()) #When to update, if using replay is it n/batch_size?



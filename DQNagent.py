from msilib.schema import Directory
import numpy as np
import random
from numpy.core.numeric import indices
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
#from kerastuner.tuners import RandomSearch
#from kerastuner.engine.hyperparameters import HyperParameters
import time

LOG_DIR = f"models/{int(time.time())}"
#from tensorflow.keras.

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

    def __init__(self, state_shape, n_possible_actions, use_target = False, use_buffer = False, batch_size = 32, learning_rate = 0.01, future_reward_discount_factor = 0.95, exploration_parameter = 0.1, network_params = [(24,'relu'),(24,'relu')]):
        self.state_shape = state_shape
        self.n_possible_actions = n_possible_actions
        self.learning_rate = learning_rate
        self.gamma = future_reward_discount_factor
        self.epsilon = exploration_parameter
        self.memory = []
        self.network_params = network_params
        self.use_target = use_target
        self.use_buffer = use_buffer
        self.batch_size = batch_size
        self.model = self._build_model()
        if use_target:
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
    
    def action_selection(self, state, method ="egreedy"):
        state = np.copy(state[np.newaxis, :])
        greedy_action = np.argmax(self.model.predict(state)[0])

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
    
    def forward_pass(self, state, action = None):
        #Returns the network Q-value(s) of the given state (and action if specified)
        state = np.copy(state[np.newaxis, :])
        if self.use_target:
            if action == None:
                return self.target_model.predict(state)
            else:
                return self.target_model.predict(state)[0][action]
        else:
            if action == None:
                return self.model.predict(state)
            else:
                return self.model.predict(state)[0][action]

    def train(self):
        
        #When buffer is used we sample randomly from memory
        #When buffer isn't used the memory is just the last n=batch_size examples
        if self.use_buffer:
            minibatch = random.sample(self.memory,self.batch_size)
        else:
            minibatch = self.memory
        
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            #Calculate targets    
            Q_values_target = self.forward_pass(state)
            if done:
                target = reward
            else:
                target = reward + self.gamma*np.max(self.forward_pass(next_state)[0])
            Q_values_target[0][action] = target
            states.append(state)
            targets.append(Q_values_target)
        states = np.array(states)
        targets = np.array(targets)

        self.model.fit(states, targets, epochs = 1, verbose = 0)

    def memorize(self, state, action, reward, next_state, done):
        if self.use_buffer:
            if len(self.memory) >= 1e6:
                self.memory.pop(0)
        else:
            if len(self.memory) >= self.batch_size:
                self.memory.pop(0)

        self.memory.append((state, action, reward, next_state, done))
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights()) #When to update, if using replay is it n/batch_size?


import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



class DQNagent():

    def __init__(self, state_shape, n_possible_actions, learning_rate = 0.01, future_reward_discount_factor = 0.95, exploration_parameter = 0.1):
        self.state_shape = state_shape
        self.n_possible_actions = n_possible_actions
        self.learning_rate = learning_rate
        self.gamma = future_reward_discount_factor
        self.epsilon = exploration_parameter
        self.memory = []
        self.model = self._build_model()
        self.target_model = self._build_model()
    
    def _build_model(self):
        #Build the neural network, might be better to have it outside of the class for experimentation...
        #Right now its just a simple one with keras but we will have to experiment with number of layers, nodes, activation function, learning rate etc.

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_shape, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_possible_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def action_selection(self, state, method ="egreedy"):
        greedy_action = np.argmax(self.model.predict(state)[0])

        #Epsilon greedy
        if method == 'egreedy':
            if np.random.rand() < self.epsilon:
                #Take random action
                return np.random.randrange(self.n_possible_actions)
            else:
                return greedy_action

        #Error catch
        else:
            print("ERROR: not a valid method, using greedy action. Use: egreedy")
            return greedy_action
    
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

        #With keras the backpropagation is done automatically with the fit() command
        self.model.fit(state, Q_values_target, epochs = 1, verbose = 0)
    
    def memorize(self, state, action, reward, next_state, done):
        
        if len(self.memory) >= 1e6:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay_train(self, batch_size = 32, use_target_network = False):
        
        minibatch = np.random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            self.train(state, action, reward, next_state, done, use_target_network)
    
    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights()) #When to update, if using replay is it n/batch_size?


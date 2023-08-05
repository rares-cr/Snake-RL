import numpy as np
import random
from collections import deque
import tensorflow as tf

# Set up the game window
WIDTH, HEIGHT = 640, 480
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

ENABLE_OBSTACLES = True
ENABLE_WALLS = True

# Define the DQN class
class DQNAgent:
    def __init__(self, state_size, action_size, memory_limit):
        # Define the size of the state and action spaces
        self.state_size = state_size
        self.action_size = action_size
        self.memory_limit = memory_limit
        
        # Define hyperparameters
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995  # Exploration rate decay
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.gamma = 0.9  # Discount factor
        self.learning_rate = 0.00025  # Learning rate
        
        # Define the replay memory buffer
        self.memory = deque(maxlen=memory_limit)
        
        # Build the Q-network and target network
        self.q_network = self.build_network()
        self.target_network = self.build_network()

    def save_model(self, path):
        # save weights
        self.q_network.save_weights(path)

    def load_model(self, path):
        # load weights
        self.q_network.load_weights(path)
        
    def build_network(self):
        # build neural network
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_dim=self.state_size),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target_network(self):
        # update target network
        self.target_network.set_weights(self.q_network.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # add experience to memory
        experience = (state, action, reward, next_state, done)
        if len(self.memory) >= self.memory_limit:
            self.memory.popleft()
        self.memory.append(experience)

    def choose_action(self, state):
        # choose action 
        # exploration - random action
        if np.random.rand() <= self.epsilon:  
            return random.randrange(self.action_size)
        # exploitation - action based on  q_network
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, batch_size):
        # sample a batch
        minibatch = random.sample(self.memory, batch_size)
        # update network
        for state, action, reward, next_state, done in minibatch:
            target = self.q_network.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)
            self.q_network.fit(state, target, epochs=1, verbose=0)
        # decrease epsilon after training
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_relative_direction(self, snake, food):
        # get direction towards food
        head_x, head_y = snake.body[0]
        food_x, food_y = food.position

        dx = food_x - head_x
        dy = food_y - head_y

        if dx > 0:
            x_dir = 'RIGHT'
        elif dx < 0:
            x_dir = 'LEFT'
        else:
            x_dir = None

        if dy > 0:
            y_dir = 'DOWN'
        elif dy < 0:
            y_dir = 'UP'
        else:
            y_dir = None

        return (x_dir, y_dir)
    
    def get_collision_info(self, snake, obstacles):
        # collect information on adjacent obstacles
        head_x, head_y = snake.body[0]
        collisions = [
            (head_x, head_y - 1) in snake.body[1:] or head_y - 1 < 0,  # UP
            (head_x, head_y + 1) in snake.body[1:] or head_y + 1 >= GRID_HEIGHT,  # DOWN
            (head_x - 1, head_y) in snake.body[1:] or head_x - 1 < 0,  # LEFT
            (head_x + 1, head_y) in snake.body[1:] or head_x + 1 >= GRID_WIDTH,  # RIGHT
        ]
        
        if ENABLE_OBSTACLES:
            for obstacle in obstacles:
                if (head_x, head_y - 1) == obstacle.position:
                    collisions[0] = True
                if (head_x, head_y + 1) == obstacle.position:
                    collisions[1] = True
                if (head_x - 1, head_y) == obstacle.position:
                    collisions[2] = True
                if (head_x + 1, head_y) == obstacle.position:
                    collisions[3] = True

        return tuple(collisions)

    def get_state(self, snake, food, obstacles):
        # Get the relative direction to the food
        x_dir, y_dir = self.get_relative_direction(snake, food)

        # Convert to numerical values
        if x_dir == 'LEFT':
            x_dir = -1
        elif x_dir == 'RIGHT':
            x_dir = 1
        else:
            x_dir = 0

        if y_dir == 'UP':
            y_dir = -1
        elif y_dir == 'DOWN':
            y_dir = 1
        else:
            y_dir = 0

        # Get the body direction
        body_direction = snake.direction
        if body_direction == 'UP':
            body_direction = [1, 0, 0, 0]
        elif body_direction == 'DOWN':
            body_direction = [0, 1, 0, 0]
        elif body_direction == 'LEFT':
            body_direction = [0, 0, 1, 0]
        elif body_direction == 'RIGHT':
            body_direction = [0, 0, 0, 1]
        else:
            body_direction = [0, 0, 0, 0]
        
        # Get the collision info
        collision_info = self.get_collision_info(snake, obstacles)

        # Convert to numerical values
        collision_info = [1 if collision else 0 for collision in collision_info]

        state = [x_dir] + [y_dir] + body_direction + collision_info
        state = np.array([state], dtype=np.float32)

        return state


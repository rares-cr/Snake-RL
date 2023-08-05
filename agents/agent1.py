import random
import itertools
import math
from collections import deque
import pickle

# Set up the game window
WIDTH, HEIGHT = 640, 480
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

ENABLE_OBSTACLES = True
ENABLE_WALLS = True

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, memory_limit=1000):
        # Define parameters
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}  # Q-table to store Q-values for each state-action pair
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum value of epsilon
        self.memory = deque(maxlen=memory_limit)  # Replay buffer to store experiences for experience replay

        self.load_q_table()  # Load or initialize the Q-table

    def load_q_table(self):
        # Load q table
        try:
            with open("q_table.pkl", "rb") as file:
                self.q_table = pickle.load(file)
                print("Loaded Q-table from file.")
        except FileNotFoundError:
            self.init_q_table()
            print("Initialized new Q-table.")

    def save_q_table(self):
        # Save q table
        with open("q_table.pkl", "wb") as file:
            pickle.dump(self.q_table, file)
            print("Saved Q-table to file.")

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Exploration; random choice
            return random.choice(self.actions)
        else:
            # Exploitation; choose action based on q table
            if state not in self.q_table:
                self.q_table[state] = {action: 0 for action in self.actions}
            max_q_value = max(self.q_table[state].values())
            best_actions = [action for action, q_value in self.q_table[state].items() if q_value == max_q_value]
            return random.choice(best_actions)

    def get_relative_direction(self, snake, food):
        # Determine the relative direction between the snake's head and the food item
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
        # Collect information about adjacent obstacles
        head_x, head_y = snake.body[0]
        # collisions with walls or body
        collisions = [
            (head_x, head_y - 1) in snake.body[1:] or head_y - 1 < 0,  # UP
            (head_x, head_y + 1) in snake.body[1:] or head_y + 1 >= GRID_HEIGHT,  # DOWN
            (head_x - 1, head_y) in snake.body[1:] or head_x - 1 < 0,  # LEFT
            (head_x + 1, head_y) in snake.body[1:] or head_x + 1 >= GRID_WIDTH,  # RIGHT
        ]
        
        # collision with obstacles
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

    def get_free_space_info(self, snake, obstacles):
        # Collect information about free adjacent spaces
        head_x, head_y = snake.body[0]
        # free space info based on walls and body
        free_spaces = [
            (head_x, head_y - 1) not in snake.body and head_y - 1 >= 0,  # UP
            (head_x, head_y + 1) not in snake.body and head_y + 1 < GRID_HEIGHT,  # DOWN
            (head_x - 1, head_y) not in snake.body and head_x - 1 >= 0,  # LEFT
            (head_x + 1, head_y) not in snake.body and head_x + 1 < GRID_WIDTH,  # RIGHT
        ]
        # free space info based on obstacles
        if ENABLE_OBSTACLES:
            for obstacle in obstacles:
                if (head_x, head_y - 1) == obstacle.position:
                    free_spaces[0] = False
                if (head_x, head_y + 1) == obstacle.position:
                    free_spaces[1] = False
                if (head_x - 1, head_y) == obstacle.position:
                    free_spaces[2] = False
                if (head_x + 1, head_y) == obstacle.position:
                    free_spaces[3] = False

        return tuple(free_spaces)

    def update_q_table(self, state, action, reward, new_state, snake, food, obstacles):
        max_next_action_value = max(self.q_table[new_state].values()) if new_state in self.q_table else 0

        # Calculate the reward based on the distance to the food
        old_distance = self.calculate_distance(snake, food)
        new_snake = snake.copy()
        new_snake.change_direction(action)
        new_snake.move(obstacles, food) 
        new_distance = self.calculate_distance(new_snake, food)

        if new_distance < old_distance:
            distance_reward = 1
        else:
            distance_reward = 0

        # Additional rewards for moving into spaces with zero adjacent obstacles
        x, y = snake.body[0]
        adjacent_obstacles = sum((x+dx, y+dy) in snake.body[1:] or (x+dx, y+dy) == obstacle.position for obstacle in obstacles for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)])
        if adjacent_obstacles == 0:
            additional_reward = 1
        else:
            additional_reward = 0

        # update q table
        self.q_table[state][action] += self.learning_rate * (reward + distance_reward + additional_reward + self.discount_factor * max_next_action_value - self.q_table[state][action])

    def get_state(self, snake, food, obstacles):
        # get the game state: relative direction to food, body direction and collision info
        relative_direction = self.get_relative_direction(snake, food)
        body_direction = snake.direction
        collision_info = self.get_collision_info(snake, obstacles)
        return (body_direction, relative_direction, collision_info)

    
    def init_q_table(self):
        # initialise q_table
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        relative_directions = [(dx, dy) for dx in [None, "LEFT", "RIGHT"] for dy in [None, "UP", "DOWN"] if dx != dy]

        for direction in directions:
            for relative_direction in relative_directions:
                for collision_info in itertools.product([True, False], repeat=4):
                    state = (direction, relative_direction, collision_info)
                    self.q_table[state] = {action: 0 for action in self.actions}

    def calculate_distance(self, snake, food):
        # Calculate the Euclidean distance between the snake's head and the food item
        head_x, head_y = snake.body[0]
        food_x, food_y = food.position

        dx = food_x - head_x
        dy = food_y - head_y

        return math.sqrt(dx * dx + dy * dy)
    
    def add_experience(self, experience):
        # Add the experience to the replay buffer
        self.memory.append(experience)

    def sample_experiences(self, batch_size):
        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def train(self, batch_size):
        # Perform training using the sampled experiences
        states, actions, rewards, next_states, dones = self.sample_experiences(batch_size)
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            max_next_action_value = max(self.q_table[next_state].values()) if next_state in self.q_table else 0
            self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * max_next_action_value - self.q_table[state][action])

        # decrease epsilon after training
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


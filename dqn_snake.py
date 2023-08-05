import numpy as np
import random
import tensorflow as tf
import pygame
import os
import pandas as pd
import gc
from agent2 import DQNAgent

seed_value = 0
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

class Snake:
    def __init__(self):
        # Initialize the snake's attributes
        self.body = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
        self.alive = True
        self.ate_food = False 

    def move(self, obstacles, food):
        # Move the snake according to its current direction
        head = self.body[0]
        x, y = head

        if self.direction == "UP":
            y -= 1
        elif self.direction == "DOWN":
            y += 1
        elif self.direction == "LEFT":
            x -= 1
        elif self.direction == "RIGHT":
            x += 1

        new_head = (x % GRID_WIDTH, y % GRID_HEIGHT)

        # Check for collision with snake body or walls
        if new_head in self.body[1:] or (ENABLE_WALLS and (x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT)):
            self.alive = False

        # Check for collision with obstacles
        for obstacle in obstacles:
            if new_head == obstacle.position:
                self.alive = False


        self.ate_food = False
        self.body.insert(0, new_head)
        # Check if the snake has eaten the food
        if new_head == food.position:
            self.ate_food = True
            food.spawn()
        else:
            self.body.pop()

    def change_direction(self, new_direction):
        # Change the direction of the snake
        self.direction = new_direction

    def draw(self):
        # Draw the snake on the screen
        for segment in self.body:
            pygame.draw.rect(screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

    def copy(self):
        # Create a copy of the snake object
        new_snake = Snake()
        new_snake.body = [part for part in self.body]
        new_snake.direction = self.direction
        new_snake.alive = self.alive
        new_snake.ate_food = self.ate_food
        return new_snake
    
# Food class
class Food:
    def __init__(self, snake, obstacles):
        # Initialize the food's attributes
        self.position = (0, 0)
        self.snake = snake
        self.obstacles = obstacles
        self.spawn()

    def spawn(self):
        # Spawn the food in a valid position
        while True:
            position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if position != self.snake.body[0] and all(position != obstacle.position for obstacle in self.obstacles):
                self.position = position
                break

    def draw(self):
        # Draw the food on the screen
        pygame.draw.rect(screen, BLUE, (self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

# Obstacle class
class Obstacle:
    def __init__(self, food):
        # Initialize the obstacle's attributes
        self.position = (0, 0)
        self.food = food
        self.spawn()

    def spawn(self):
        # Spawn the food in a valid position
        self.position = self.generate_position()

    def generate_position(self):
        # Generate a valid position
        while True:
            position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if position != self.food.position:
                return position

    def draw(self):
        # Draw obstacle on screen
        pygame.draw.rect(screen, RED, (self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def initialize_pygame(train=True):
    # Initialize Pygame
    if not train:
        pygame.init()

    # Set up the game window
    global WIDTH, HEIGHT, GRID_SIZE, GRID_WIDTH, GRID_HEIGHT, FPS, WHITE, GREEN, RED, BLUE
    global FONT_SIZE, FONT_COLOR, FONT, ENABLE_OBSTACLES, ENABLE_WALLS, screen, clock

    WIDTH, HEIGHT = 640, 480
    GRID_SIZE = 20
    GRID_WIDTH = WIDTH // GRID_SIZE
    GRID_HEIGHT = HEIGHT // GRID_SIZE
    FPS = 10
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    # Font settings
    if not train:
        FONT_SIZE = 24
        FONT_COLOR = (0, 0, 0)
        FONT = pygame.font.SysFont(None, FONT_SIZE)

    ENABLE_OBSTACLES = True  
    ENABLE_WALLS = True

    if not train:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        clock = pygame.time.Clock()

def initialize_game(num_obstacles = 5):
    # Create snake instance
    snake = Snake()
    
    # Create an initial food instance. This may need to be recreated later if obstacles overlap.
    food = Food(snake, [])

    obstacles = []
    
    if ENABLE_OBSTACLES:
        # Create obstacles
        for _ in range(num_obstacles):  # Adjust the number of obstacles as desired
            obstacle = Obstacle(food)  # Pass food to the Obstacle
            obstacles.append(obstacle)

    # Now we need to ensure that the food isn't on an obstacle, so create a new food instance with the obstacles
    food = Food(snake, obstacles)

    return snake, food, obstacles

def learn(agent, episodes, batch_size):
    # initialise metrics lists
    scores = []
    steps = []
    rewards = []
    for e in range(episodes):
        # init game
        initialize_pygame()
        # init game objects
        snake, food, obstacles = initialize_game()
        state = agent.get_state(snake, food, obstacles)

        # initialize metrics
        score = 0
        step = 0
        total_reward = 0

        done = False
        while not done:
            action = agent.choose_action(state)

            # map action to direction
            action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
            action_dir = action_map[action]
        
            # Update the snake's direction based on the chosen action
            if action_dir == "UP" and snake.direction != "DOWN":
                snake.change_direction("UP")
            elif action_dir == "DOWN" and snake.direction != "UP":
                snake.change_direction("DOWN")
            elif action_dir == "LEFT" and snake.direction != "RIGHT":
                snake.change_direction("LEFT")
            elif action_dir == "RIGHT" and snake.direction != "LEFT":
                snake.change_direction("RIGHT")

            # move snake, update steps, get next_state
            snake.move(obstacles, food)
            step += 1
            next_state = agent.get_state(snake, food, obstacles)

            # manage rewards and termination conditions
            if not snake.alive:
                reward = -10
                total_reward += -10
                done = True
            elif snake.ate_food:
                score += 1
                reward = 100
                total_reward += 100
                food = Food(snake, obstacles)
            else:
                reward = -1
                total_reward += -1

            # Additional rewards for moving into spaces with zero adjacent obstacles
            x, y = snake.body[0]
            adjacent_obstacles = sum((x+dx, y+dy) in snake.body[1:] or (x+dx, y+dy) == obstacle.position for obstacle in obstacles for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)])
            if adjacent_obstacles == 0:
                reward += 1
                total_reward += 1

            # Additional rewards for moving closer to the food
            current_dist = abs(x - food.position[0]) + abs(y - food.position[1])
            new_dist = abs(next_state[0][0] - food.position[0]) + abs(next_state[0][1] - food.position[1])
            if new_dist < current_dist:
                reward += 1
                total_reward += 1


            # The agent remembers the transition
            agent.remember(state, action, reward, next_state, done)

            # Transition to the next state
            state = next_state

        # append metrics to lists
        scores.append(score)
        steps.append(step)
        rewards.append(total_reward)
        
        # train agent
        if len(agent.memory) >= batch_size:
                agent.train(batch_size)

        # Update the target network every 10 episodes
        if e % 10 == 0:
            agent.update_target_network()
            agent.save_model(f'dqn_model.h5')

        # Delete objects and variables to free up memory
        del snake
        del food
        del obstacles

        # Call garbage collection explicitly
        gc.collect()
        
        print(f'Episode: {e}/{episodes}, Score: {score}')
        pygame.quit()

    # save metrics as csv
    metrics = pd.DataFrame({'Scores':scores, 'Steps':steps, 'Rewards':rewards})
    metrics.to_csv('metrics.csv', index=False)
    agent.save_model('dqn_model.h5')

def play(agent, episodes, num_obstacles=5):
    # set epsilon to 0
    agent.epsilon = 0
    # initialise metrics lists
    scores = []
    steps = []
    rewards = []
    for e in range(episodes):
        # init game
        initialize_pygame()
        # init game objects
        snake, food, obstacles = initialize_game(num_obstacles)
        state = agent.get_state(snake, food, obstacles)

        # initialize metrics
        score = 0
        step = 0
        total_reward = 0

        done = False
        while not done:
            action = agent.choose_action(state)

            # map action to direction
            action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
            action_dir = action_map[action]
        
            # Update the snake's direction based on the chosen action
            if action_dir == "UP" and snake.direction != "DOWN":
                snake.change_direction("UP")
            elif action_dir == "DOWN" and snake.direction != "UP":
                snake.change_direction("DOWN")
            elif action_dir == "LEFT" and snake.direction != "RIGHT":
                snake.change_direction("LEFT")
            elif action_dir == "RIGHT" and snake.direction != "LEFT":
                snake.change_direction("RIGHT")

            # move snake, update steps, get next_state
            snake.move(obstacles, food)
            step += 1
            next_state = agent.get_state(snake, food, obstacles)

            # manage rewards and termination conditions
            if not snake.alive:
                total_reward += -10
                done = True
            elif step > 2000:
                done = True
            elif snake.ate_food:
                score += 1
                total_reward += 100
                food = Food(snake, obstacles)
            else:
                total_reward += -1

            # Additional rewards for moving into spaces with zero adjacent obstacles
            x, y = snake.body[0]
            adjacent_obstacles = sum((x+dx, y+dy) in snake.body[1:] or (x+dx, y+dy) == obstacle.position for obstacle in obstacles for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)])
            if adjacent_obstacles == 0:
                total_reward += 1

            # Additional rewards for moving closer to the food
            current_dist = abs(x - food.position[0]) + abs(y - food.position[1])
            new_dist = abs(next_state[0][0] - food.position[0]) + abs(next_state[0][1] - food.position[1])
            if new_dist < current_dist:
                total_reward += 1

            # Transition to the next state
            state = next_state

        # append metrics to lists
        scores.append(score)
        steps.append(step)
        rewards.append(total_reward)

        # Delete objects and variables to free up memory
        del snake
        del food
        del obstacles

        # Call garbage collection explicitly
        gc.collect()
        
        print(f'Episode: {e}/{episodes}, Score: {score}')
        pygame.quit()

    # save metrics as csv
    results = pd.DataFrame({'Scores':scores, 'Steps':steps, 'Rewards':rewards})
    return results
    

if __name__ == "__main__":

    # Set the parameters
    state_size = 10  # The size of the state representation 
    action_size = 4  # The number of possible actions: [UP, DOWN, LEFT, RIGHT]
    episodes = 1000  # The number of episodes to train for
    batch_size = 64 # The size of the batches used for learning

    # Create the agent
    agent = DQNAgent(state_size, action_size, memory_limit=5000)

    # Load pre-trained weights, if they exist
    model_path = 'dqn_model_final.h5' 
    if os.path.isfile(model_path):
        agent.load_model(model_path)

    # Train the agent
    learn(agent, episodes, batch_size)

    list_of_obstacles = [0, 5, 15]
    # Play
    for o in list_of_obstacles:
        # Perform the play with the agent
        results = play(agent, 100, num_obstacles=o)
        # Save the metrics as CSV with a unique name
        file_name = f'results_obstacles_{o}.csv'  # Use a different name for each iteration
        results.to_csv(file_name, index=False)

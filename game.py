import pygame
import random
import numpy as np

class SnakeGameAI:
    def __init__(self, width=400, height=400, block_size=20, visualize=True):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.visualize = visualize

        # Initialize pygame if visualization is enabled
        if self.visualize:
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake AI Training")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.direction = "RIGHT"
        self.head = [self.width // 2, self.height // 2]
        self.snake = [self.head[:], [self.head[0] - self.block_size, self.head[1]],
                      [self.head[0] - 2 * self.block_size, self.head[1]]]
        self.food = self._place_food()
        self.score = 0
        self.frame_iteration = 0
        return self._get_state()

    def render(self):
        """
        Render the game using Pygame and handle events to prevent freezing.
        """
        if not self.visualize:
            return  # Skip rendering if visualization is disabled

        # Process events to prevent the window from becoming unresponsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # Handle window close
                pygame.quit()
                quit()

        # Clear the screen
        self.display.fill((0, 0, 0))

        # Draw the snake
        for segment in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(segment[0], segment[1], self.block_size, self.block_size))

        # Draw the food
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))

        # Update the display
        pygame.display.flip()
        
    def step(self, action):
        self.frame_iteration += 1

        # Remember the current head position
        old_head = self.head[:]

        # Update direction based on action
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        idx = directions.index(self.direction)
        if action == [1, 0, 0]:  # Straight
            pass
        elif action == [0, 1, 0]:  # Right turn
            self.direction = directions[(idx + 1) % 4]
        elif action == [0, 0, 1]:  # Left turn
            self.direction = directions[(idx - 1) % 4]

        # Move the snake
        x, y = self.head
        if self.direction == "UP":
            y -= self.block_size
        elif self.direction == "DOWN":
            y += self.block_size
        elif self.direction == "LEFT":
            x -= self.block_size
        elif self.direction == "RIGHT":
            x += self.block_size
        self.head = [x, y]
        self.snake.insert(0, self.head)

        # Check for collisions
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            return self._get_state(), -10, True, self.score  # Game over with penalty

        # Check if snake eats food
        if self.head == self.food:
            self.score += 1
            reward = 10  # Reward for eating food
            self.food = self._place_food()
        else:
            # Small penalty for each step
            reward = -0.1
            self.snake.pop()

            # Additional reward/penalty based on proximity to food
            food_x, food_y = self.food
            food_distance_before = abs(old_head[0] - food_x) + abs(old_head[1] - food_y)
            food_distance_after = abs(self.head[0] - food_x) + abs(self.head[1] - food_y)

            if food_distance_after < food_distance_before:
                reward += 0.5  # Reward for moving closer to the food
            else:
                reward -= 0.5  # Penalty for moving farther from the food

        return self._get_state(), reward, False, self.score

    def _is_collision(self):
        x, y = self.head
        # Check wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # Check self-collision
        if self.head in self.snake[1:]:
            return True
        return False

    def _place_food(self):
        while True:
            x = random.randint(0, (self.width // self.block_size) - 1) * self.block_size
            y = random.randint(0, (self.height // self.block_size) - 1) * self.block_size
            if [x, y] not in self.snake:
                return [x, y]

    def _get_state(self):
        x, y = self.head
        food_x, food_y = self.food

        state = [
            x < food_x,  # Food is to the right
            x > food_x,  # Food is to the left
            y < food_y,  # Food is below
            y > food_y,  # Food is above
            self.direction == "LEFT",  # Moving left
            self.direction == "RIGHT",  # Moving right
            self.direction == "UP",  # Moving up
            self.direction == "DOWN",  # Moving down
            self._is_obstacle_ahead(),  # Danger ahead
            self._is_obstacle_left(),   # Danger to the left
            self._is_obstacle_right()   # Danger to the right
        ]
        return np.array(state, dtype=int)
        
    def _is_obstacle_ahead(self):
        x, y = self.head
        if self.direction == "UP":
            y -= self.block_size
        elif self.direction == "DOWN":
            y += self.block_size
        elif self.direction == "LEFT":
            x -= self.block_size
        elif self.direction == "RIGHT":
            x += self.block_size
        return [x, y] in self.snake or x < 0 or x >= self.width or y < 0 or y >= self.height

    def _is_obstacle_left(self):
        x, y = self.head
        if self.direction == "UP":
            x -= self.block_size
        elif self.direction == "DOWN":
            x += self.block_size
        elif self.direction == "LEFT":
            y += self.block_size
        elif self.direction == "RIGHT":
            y -= self.block_size
        return [x, y] in self.snake or x < 0 or x >= self.width or y < 0 or y >= self.height

    def _is_obstacle_right(self):
        x, y = self.head
        if self.direction == "UP":
            x += self.block_size
        elif self.direction == "DOWN":
            x -= self.block_size
        elif self.direction == "LEFT":
            y -= self.block_size
        elif self.direction == "RIGHT":
            y += self.block_size
        return [x, y] in self.snake or x < 0 or x >= self.width or y < 0 or y >= self.height

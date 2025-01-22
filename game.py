import pygame
import random

class SnakeGameAI:
    """
    Represents the Snake game environment for AI training.

    Args:
        width (int): Width of the game window.
        height (int): Height of the game window.
        block_size (int): Size of each block in the game grid.
        visualize (bool): Whether to visualize the game during training.
        timeout_multiplier (int): Multiplier for calculating frame timeout based on snake length.
    """
    def __init__(self, width=640, height=480, block_size=20, visualize=False, timeout_multiplier=50):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.visualize = visualize
        self.timeout_multiplier = timeout_multiplier
        self.reset()

        if self.visualize:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

    def reset(self):
        """
        Reset the game state to start a new episode.
        
        Returns:
            list: Initial state of the game.
        """
        self.direction = "RIGHT"
        self.head = [self.width // 2, self.height // 2]
        self.snake = [self.head[:], [self.head[0] - self.block_size, self.head[1]], [self.head[0] - 2 * self.block_size, self.head[1]]]
        self.food = self._place_food()
        self.score = 0
        self.frame_iteration = 0
        return self._get_state()

    def step(self, action):
        """
        Execute one step in the game based on the selected action.

        Args:
            action (list): Action vector indicating direction.

        Returns:
            tuple: Next state, reward, whether the game ended, and the score.
        """
        self.frame_iteration += 1
        old_head = self.head[:]
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        idx = directions.index(self.direction)

        # Update the directions
        if action == [1, 0, 0]:  # Go straight
            pass
        elif action == [0, 1, 0]:  # Turn right
            self.direction = directions[(idx + 1) % 4]
        elif action == [0, 0, 1]:  # Turn left
            self.direction = directions[(idx - 1) % 4]

        # move the snake
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

        # Check for collisions or timeouts
        collision_penalty = -5000
        if self._is_collision() or self.frame_iteration > 50 * len(self.snake):
            return self._get_state(), collision_penalty, True, self.score

        # Check if snake eats food
        if self.head == self.food:
            self.score += 1
            reward = 5000  # Food reward
            self.food = self._place_food()
        else:
            reward = self._get_reward(old_head)
            self.snake.pop()

        return self._get_state(), reward, False, self.score

    def _place_food(self):
        """
        Randomly place food in the game grid.

        Returns:
            list: Coordinates of the placed food.
        """
        while True:
            x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
            y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
            if [x, y] not in self.snake:
                return [x, y]
            
    def _get_reward(self, old_head):
        """
        Calculate the reward for the current step.

        Args:
            old_head (list): The previous position of the snake's head.

        Returns:
            float: Reward for the current step.
        """
        food_distance_before = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        food_distance_after = abs(self.head[0] - self.food[0]) + abs(self.head[1] - self.food[1])

        # Reward shaping
        if self.head == self.food:
            return 1000  # Reward for eating food
        elif self._is_collision():
            return -100  # Penalty for collision
        elif food_distance_after < food_distance_before:
            return 10  # Reward for moving closer to food
        else:
            return -1  # Small penalty for moving farther or idle movement

    def _is_collision(self, point=None):
        """
        Check for collisions with walls or the snake's own body.

        Args:
            point (list): Point to check for collision. Defaults to the snake's head.

        Returns:
            bool: True if collision occurred, False otherwise.
        """
        if point is None:
            point = self.head
        if point[0] < 0 or point[0] >= self.width or point[1] < 0 or point[1] >= self.height:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def _get_state(self):
        """
        Retrieve the current state of the game.

        Returns:
            list: Current state vector with size 15 (expanded features).
        """
        head_x, head_y = self.head
        food_x, food_y = self.food

        # Danger in each direction
        danger_straight = self._is_collision(self._next_head_pos(self.direction))
        danger_right = self._is_collision(self._next_head_pos(self._rotate_right(self.direction)))
        danger_left = self._is_collision(self._next_head_pos(self._rotate_left(self.direction)))

        # Direction as one-hot encoding
        direction_up = self.direction == "UP"
        direction_down = self.direction == "DOWN"
        direction_left = self.direction == "LEFT"
        direction_right = self.direction == "RIGHT"

        # Relative position of food
        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y

        # Distance to walls
        distance_up = head_y // self.block_size
        distance_down = (self.height - head_y) // self.block_size
        distance_left = head_x // self.block_size
        distance_right = (self.width - head_x) // self.block_size

        return [
            danger_straight, danger_right, danger_left,
            direction_up, direction_down, direction_left, direction_right,
            food_left, food_right, food_up, food_down,
            distance_up, distance_down, distance_left, distance_right
        ]

    def render(self):
        """
        Render the game visualization.
        """
        self.display.fill((0, 0, 0))
        for block in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(block[0], block[1], self.block_size, self.block_size))
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        pygame.display.flip()

    def _next_head_pos(self, direction):
        """
        Calculate the next head position based on the given direction.

        Args:
            direction (str): Current direction ("UP", "DOWN", "LEFT", "RIGHT").

        Returns:
            list: Next head position [x, y].
        """
        x, y = self.head
        if direction == "UP":
            y -= self.block_size
        elif direction == "DOWN":
            y += self.block_size
        elif direction == "LEFT":
            x -= self.block_size
        elif direction == "RIGHT":
            x += self.block_size
        return [x, y]

    def _rotate_right(self, direction):
        """
        Rotate the direction 90 degrees to the right.

        Args:
            direction (str): Current direction.

        Returns:
            str: New direction after rotating right.
        """
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        return directions[(directions.index(direction) + 1) % 4]

    def _rotate_left(self, direction):
        """
        Rotate the direction 90 degrees to the left.

        Args:
            direction (str): Current direction.

        Returns:
            str: New direction after rotating left.
        """
        directions = ["UP", "RIGHT", "DOWN", "LEFT"]
        return directions[(directions.index(direction) - 1) % 4]

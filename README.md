
# Snake AI with Deep Q-Learning

This project implements an AI agent to play the classic Snake game using **Deep Q-Learning** (DQN), a popular reinforcement learning algorithm. The agent learns to navigate the environment, eat food, and avoid collisions through trial and error.

---

## General Information

### Features
- **AI Agent**: The AI controls the snake using a trained neural network.
- **State Representation**: The snake's environment is represented as a vector of features, including food location, movement direction, and obstacles.
- **Reward System**: Rewards are designed to encourage eating food, penalize collisions, and improve gameplay over time.
- **Visualization Toggle**: Choose to visualize the game during training or run in headless mode for faster learning.
- **Training Metrics**: Tracks scores and rewards across episodes, with a graphical display of training progress.

---

## Prerequisites

### Requirements
- Python 3.7 or higher
- Virtual environment (optional but recommended)
- Required Python packages:
  ```bash
  pip install pygame torch numpy matplotlib
  ```

### How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Snake_DQN.git
   cd Snake_DQN
   ```
2. Set up a virtual environment (optional):
   ```bash
   python -m venv snake_ai_env
   source snake_ai_env/bin/activate  # For Linux/Mac
   snake_ai_env\Scripts\activate    # For Windows
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python main.py
   ```
4. When prompted, choose whether to enable visualization (`y` or `n`).

---

## Detailed Explanation of Deep Q-Learning (DQN)

Deep Q-Learning is a reinforcement learning algorithm that enables an agent to learn an optimal policy for interacting with an environment. In this project, the Snake AI uses DQN to maximize rewards over time by learning from its actions.

### Key Concepts

#### **State**
The state is a representation of the current environment. For the Snake game, the state includes:
- Food's relative position to the snake's head (4 features: `food_up`, `food_down`, `food_left`, `food_right`).
- Snake's current movement direction (4 features: `moving_up`, `moving_down`, `moving_left`, `moving_right`).
- Obstacles relative to the snake's head (3 features: `danger_ahead`, `danger_left`, `danger_right`).

**State Vector Example (11 features):**
```python
[food_up, food_down, food_left, food_right,
 moving_up, moving_down, moving_left, moving_right,
 danger_ahead, danger_left, danger_right]
```

#### **Action**
The snake can choose one of three actions:
- `[1, 0, 0]`: Move straight.
- `[0, 1, 0]`: Turn right.
- `[0, 0, 1]`: Turn left.

#### **Reward**
Rewards are used to guide the learning process:
- **+10** for eating food.
- **-10** for colliding with walls or itself.
- **-0.1** for each step taken without progress.
- **+1** for moving closer to the food.

#### **Neural Network (DQN)**
The DQN is a fully connected neural network that maps states to Q-values for each action. It predicts the expected cumulative reward for each action in a given state.

**Architecture:**
- Input Layer: 11 neurons (state size).
- Hidden Layers: 2 layers with 128 neurons each.
- Output Layer: 3 neurons (one for each action).

---

## Visualization

- **Real-Time Visualization**:
  - The game is rendered during training if visualization is enabled, showing the snake's movements, food, and obstacles.
- **Training Metrics**:
  - At the end of training, a graph of scores per episode and rolling average is displayed.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contributions

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository and submit a pull request.

---

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the neural network implementation.
- [Pygame](https://www.pygame.org/) for the game rendering.

---

Feel free to reach out if you have any questions or suggestions!

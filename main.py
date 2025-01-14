from game import SnakeGameAI
from trainer import SnakeAITrainer
from utils import save_model, load_model, plot_training_progress, calculate_mean
import torch
import random

# Main training loop
def train(visualize):
    game = SnakeGameAI(visualize=visualize)
    trainer = SnakeAITrainer(state_size=11, action_size=3)

    # Load a pre-trained model if available
    trainer.model, trainer.optimizer = load_model(trainer.model, trainer.optimizer, "snake_model.pth")

    # Define exploration parameters
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995

    EPISODES = 1000  # Number of training episodes
    scores = []
    mean_scores = []

    for episode in range(EPISODES):
        state = game.reset()
        done = False
        total_reward = 0

        # Calculate epsilon for this episode
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:  # Exploration
                action = random.randint(0, 2)
            else:  # Exploitation
                action = torch.argmax(
                    trainer.model(torch.tensor(state, dtype=torch.float32))
                ).item()

            # Take a step in the environment
            next_state, reward, done, score = game.step(
                [1 if i == action else 0 for i in range(3)]
            )
            trainer.store_transition(state, action, reward, next_state, done)
            trainer.train_step()
            state = next_state
            total_reward += reward

            # Render the game (if visualization is enabled)
            if visualize:
                game.render()
                game.clock.tick(10)

        # Track training progress
        scores.append(score)
        mean_scores = calculate_mean(scores)

        print(f"Episode {episode + 1}/{EPISODES}, Score: {score}, Reward: {round(total_reward, 2)}")

        # Save the model every 50 episodes
        if (episode + 1) % 50 == 0:
            save_model(trainer.model, trainer.optimizer, "snake_model.pth")

    # Visualize the training progress at the end
    plot_training_progress(scores, mean_scores)


if __name__ == "__main__":
    # Ask the user if they want to enable visualization
    visualize_input = input("Enable visualization? (y/n): ").strip().lower()
    visualize = visualize_input == "y"

    train(visualize)

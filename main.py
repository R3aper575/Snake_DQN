from game import SnakeGameAI
from trainer import SnakeAITrainer
from utils import save_model, load_model, plot_training_progress, calculate_mean, save_run_results
import torch
import random
import time

# Main training loop
def train(visualize):
    """
    Executes the main training loop for the Snake AI.

    Args:
        visualize (bool): Whether to enable game visualization during training.

    Returns:
        tuple: Scores from all episodes and total training time.
    """
    # Initialize game and trainer
    game = SnakeGameAI(visualize=visualize)
    trainer = SnakeAITrainer(state_size=11, action_size=3)

    # Load pre-trained model if available
    trainer.model, trainer.optimizer = load_model(trainer.model, trainer.optimizer, "snake_model.pth")

    # Exploration parameters
    epsilon_start = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995

    EPISODES = 1000  # Total training episodes
    scores = []
    mean_scores = []

    # Track training time
    start_time = time.time()

    for episode in range(EPISODES):
        # Reset the game for a new episode
        state = game.reset()
        done = False
        total_reward = 0

        # Calculate epsilon for exploration-exploitation balance
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** episode))

        while not done:
            # Select action using epsilon-greedy policy
            if random.random() < epsilon:  # Explore
                action = random.randint(0, 2)
            else:  # Exploit
                action = torch.argmax(
                    trainer.model(torch.tensor(state, dtype=torch.float32))
                ).item()

            # Perform the action and observe the next state and reward
            next_state, reward, done, score = game.step(
                [1 if i == action else 0 for i in range(3)]
            )
            trainer.store_transition(state, action, reward, next_state, done)
            trainer.train_step()
            state = next_state
            total_reward += reward

            # Render game visualization if enabled
            if visualize:
                game.render()
                game.clock.tick(10)

        # Track scores and calculate rolling average
        scores.append(score)
        mean_scores = calculate_mean(scores)

        # Log progress for the current episode
        print(f"Episode {episode + 1}/{EPISODES}, Score: {score}, Reward: {round(total_reward, 2)}")

        # Save model periodically
        if (episode + 1) % 50 == 0:
            save_model(trainer.model, trainer.optimizer, "snake_model.pth")

    # Calculate total training time
    end_time = time.time()
    total_time = end_time - start_time

    # Print training duration
    minutes, seconds = divmod(total_time, 60)
    print(f"\nTraining completed in {int(minutes)} minutes and {int(seconds)} seconds.")

    # Plot training progress
    plot_training_progress(scores, mean_scores)

    return scores, total_time


if __name__ == "__main__":
    # Ask the user to enable visualization
    visualize_input = input("Enable visualization? (y/n): ").strip().lower()
    visualize = visualize_input == "y"

    # Run the training process
    scores, total_time = train(visualize)

    # Prepare and save training parameters and results
    parameters = {
        "Episodes": 1000,
        "Learning Rate": 0.001,
        "Epsilon Start": 1.0,
        "Epsilon Min": 0.1,
        "Epsilon Decay": 0.995,
    }
    results = {
        "Total Time (s)": round(total_time, 2),
        "Max Score": max(scores),
        "Average Score": round(sum(scores) / len(scores), 2),
    }
    save_run_results(parameters, results)

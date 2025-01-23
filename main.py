from game import SnakeGameAI
from trainer import SnakeAITrainer
from utils import save_model, load_model, plot_training_progress, calculate_mean, save_run_results
import torch
import random
import time


def train(visualize, epsilon=None):
    """
    Executes the main training loop for the Snake AI.

    Args:
        visualize (bool): Whether to enable game visualization during training.
        epsilon (float): Starting exploration rate. Use None to load from model or initialize dynamically.
    """
    # Initialize game and trainer
    game = SnakeGameAI(visualize=visualize)
    trainer = SnakeAITrainer(state_size=15, action_size=3)

    # Load model, optimizer, and epsilon
    trainer.model, trainer.optimizer, loaded_epsilon = load_model(trainer.model, trainer.optimizer, "snake_model.pth")

    # Use loaded epsilon if not explicitly provided
    if epsilon is None:
        epsilon = loaded_epsilon

    # Exploration parameters
    epsilon_min = 0.1
    epsilon_decay = 0.997

    EPISODES = 1000
    scores = []
    mean_scores = []
    total_losses = []

    # Start timing the training process
    start_time = time.time()

    for episode in range(EPISODES):
        state = game.reset()
        done = False
        total_reward = 0

        # Normalize initial state
        state = torch.tensor(state, dtype=torch.float32)
        state = (state - state.mean()) / (state.std() + 1e-5)

        # Update epsilon for this episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        while not done:
            # Choose action: exploration vs. exploitation
            if random.random() < epsilon:
                action = random.randint(0, 2)  # Exploration
            else:
                # Set model to evaluation mode to bypass batch norm issues
                trainer.model.eval()
                with torch.no_grad():
                    action = torch.argmax(trainer.model(state.unsqueeze(0))).item()
                trainer.model.train()

            # Perform action and retrieve next state
            next_state, reward, done, score = game.step(
                [1 if i == action else 0 for i in range(3)]
            )

            # Normalize next state
            next_state = torch.tensor(next_state, dtype=torch.float32)
            next_state = (next_state - next_state.mean()) / (next_state.std() + 1e-5)

            # Store transition and train
            trainer.store_transition(state.tolist(), action, reward, next_state.tolist(), done)
            trainer.train_step()

            # Accumulate rewards and update current state
            state = next_state
            total_reward += reward

            if visualize:
                game.render()
                game.clock.tick(10)

        # Track performance metrics
        scores.append(score)
        mean_scores = calculate_mean(scores)

        print(f"Episode {episode + 1}/{EPISODES}, Score: {score}, Reward: {round(total_reward, 2)}, Epsilon: {epsilon:.3f}")

        # Save model every 50 episodes
        if (episode + 1) % 50 == 0:
            save_model(trainer.model, trainer.optimizer, epsilon, "snake_model.pth")

    # End timing the training process
    end_time = time.time()
    total_time = end_time - start_time

    # Log training metrics
    save_run_results(
        parameters={
            "Episodes": EPISODES,
            "Learning Rate": trainer.lr,
            "Epsilon Start": epsilon,
            "Epsilon Min": epsilon_min,
            "Epsilon Decay": epsilon_decay,
        },
        results={
            "Total Time (s)": round(total_time, 2),
            "Max Score": max(scores),
            "Average Score": round(sum(scores) / len(scores), 2),
        },
    )

    return scores, total_time, mean_scores, epsilon


if __name__ == "__main__":
    # Ask the user to enable visualization
    visualize_input = input("Enable visualization? (y/n): ").strip().lower()
    visualize = visualize_input == "y"

    # Run the training process
    scores, total_time, mean_scores, epsilon = train(visualize)

    # Prepare and save training parameters and results
    parameters = {
        "Episodes": 1000,
        "Learning Rate": 0.001,
        "Epsilon Start": 1.0,
        "Epsilon Min": 0.1,
        "Epsilon Decay": 0.997,
        "Final Epsilon": epsilon,
    }
    results = {
        "Total Time (s)": round(total_time, 2),
        "Max Score": max(scores),
        "Average Score": round(sum(scores) / len(scores), 2),
    }
    save_run_results(parameters, results)

    # Print total training time
    minutes, seconds = divmod(total_time, 60)
    print(f"\nTraining completed in {int(minutes)} minutes and {int(seconds)} seconds.")

from game import SnakeGameAI
from trainer import SnakeAITrainer
from utils import save_model, load_model, plot_training_progress, calculate_mean, save_run_results
import torch
import random
import time

# Main training loop
def train(visualize, epsilon=None):
    """
    Executes the main training loop for the Snake AI.

    Args:
        visualize (bool): Whether to enable game visualization during training.
        epsilon (float): Starting exploration rate. Use None to load from model or initialize dynamically.
    """
    # Initialize game and trainer
    game = SnakeGameAI(visualize=visualize)
    trainer = SnakeAITrainer(state_size=11, action_size=3)

    # Load model, optimizer, and epsilon
    trainer.model, trainer.optimizer, loaded_epsilon = load_model(trainer.model, trainer.optimizer, "snake_model.pth")

    # Use loaded epsilon if not explicitly provided
    if epsilon is None:
        epsilon = loaded_epsilon

    # Exploration parameters
    epsilon_min = 0.1
    epsilon_decay = 0.999

    EPISODES = 1000
    scores = []
    mean_scores = []

    # Start timing the training process
    start_time = time.time()

    for episode in range(EPISODES):
        state = game.reset()
        done = False
        total_reward = 0

        # Update epsilon for this episode
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        while not done:
            if random.random() < epsilon:  # Exploration
                action = random.randint(0, 2)
            else:  # Exploitation
                action = torch.argmax(
                    trainer.model(torch.tensor(state, dtype=torch.float32))
                ).item()

            next_state, reward, done, score = game.step(
                [1 if i == action else 0 for i in range(3)]
            )
            trainer.store_transition(state, action, reward, next_state, done)
            trainer.train_step()
            state = next_state
            total_reward += reward

            if visualize:
                game.render()
                game.clock.tick(10)

        scores.append(score)
        mean_scores = calculate_mean(scores)

        print(f"Episode {episode + 1}/{EPISODES}, Score: {score}, Reward: {round(total_reward, 2)}")

        if (episode + 1) % 50 == 0:
            save_model(trainer.model, trainer.optimizer, epsilon, "snake_model.pth")

            # Dynamic epsilon adjustment based on recent performance
            recent_mean_score = sum(scores[-50:]) / min(len(scores), 50)
            ############################ Mess with recent_mean_score necessary ####################
            # if recent_mean_score < 5:  
            #     epsilon = min(1.0, epsilon * 1.1)  # Increase exploration
            #     print(f"Low recent performance detected. Increasing epsilon to {epsilon:.3f}")
            #######################################################################################

        # End timing the training process
        end_time = time.time()
        total_time = end_time - start_time

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
        "Epsilon Decay": 0.999,
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

import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime

def save_model(model, optimizer, epsilon, file_path="snake_model.pth"):
    """
    Save the model and optimizer state to a file.

    Args:
        model (torch.nn.Module): The neural network model to save.
        optimizer (torch.optim.Optimizer): The optimizer associated with the model.
        file_path (str): The file path where the model will be saved.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
    }, file_path)
    print(f"Model saved to {file_path}")

def load_model(model, optimizer, file_path="snake_model.pth"):
    """
    Load the model and optimizer state from a file.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.
        file_path (str): The file path from where the model will be loaded.

    Returns:
        tuple: The model and optimizer with loaded states.
    """
    if os.path.exists(file_path):
        try:
            checkpoint = torch.load(file_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epsilon = checkpoint.get('epsilon', 1.0)  # Default to 1.0 if not saved
            print(f"Model and optimizer loaded from {file_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"No model file found at {file_path}. Starting fresh.")
    return model, optimizer, epsilon

def plot_training_progress(scores, mean_scores):
    """
    Plot the training progress including scores and mean scores.

    Args:
        scores (list): List of scores for each episode.
        mean_scores (list): List of mean scores calculated over episodes.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label="Scores")
    plt.plot(mean_scores, label="Mean Scores")
    plt.title("Training Progress")
    plt.xlabel("Episodes")
    plt.ylabel("Scores")
    plt.legend()
    plt.show()

def calculate_mean(scores, window=50):
    """
    Calculate the mean scores over a rolling window.

    Args:
        scores (list): List of scores from episodes.
        window (int): The size of the rolling window for averaging.

    Returns:
        list: List of mean scores.
    """
    return [sum(scores[max(0, i-window+1):i+1]) / (i - max(0, i-window+1) + 1) for i in range(len(scores))]

def save_run_results(parameters, results, file_path="training_results.txt"):
    """
    Save training parameters and results to a file with a timestamp.

    Args:
        parameters (dict): Dictionary of parameters used for training.
        results (dict): Dictionary of results from the training session.
        file_path (str): The file path where the results will be saved.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_content = f"\n=== Training Run at {timestamp} ===\n"
    log_content += "Parameters:\n"
    for key, value in parameters.items():
        log_content += f"  {key}: {value}\n"
    log_content += "Results:\n"
    for key, value in results.items():
        log_content += f"  {key}: {value}\n"
    log_content += "===========================\n"

    with open(file_path, "a") as file:
        file.write(log_content)

    print(f"Results saved to {file_path}")

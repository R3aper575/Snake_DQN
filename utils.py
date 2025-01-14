import torch
import os


def save_model(model, optimizer, file_path="snake_model.pth"):
    """
    Saves the model's state dictionary to a file.

    Args:
        model (torch.nn.Module): The model to save.
        file_path (str): Path to save the model.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print(f"Model and optimizer saved to {file_path}")


def load_model(model, optimizer, file_path="snake_model.pth"):
    """
    Loads the model's state dictionary from a file.

    Args:
        model (torch.nn.Module): The model to load into.
        file_path (str): Path to load the model from.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    if os.path.exists(file_path):
        
        try:
            checkpoint = torch.load(file_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model and optimizer loaded from {file_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"No model file found at {file_path}. Starting fresh.")
    return model, optimizer


def plot_training_progress(scores, mean_scores):
    """
    Plots the training progress (score per episode and average score).

    Args:
        scores (list): List of scores for each episode.
        mean_scores (list): List of average scores over time.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(scores, label="Score per Episode")
    plt.plot(mean_scores, label="Average Score", linestyle="--")
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.title("Training Progress")
    plt.legend()
    plt.grid()
    plt.show()


def calculate_mean(scores, window=10):
    """
    Calculates the rolling mean of scores over a specified window.

    Args:
        scores (list): List of scores.
        window (int): Rolling window size.

    Returns:
        list: Rolling mean scores.
    """
    mean_scores = []
    for i in range(len(scores)):
        mean = sum(scores[max(0, i - window + 1):i + 1]) / min(window, i + 1)
        mean_scores.append(mean)
    return mean_scores

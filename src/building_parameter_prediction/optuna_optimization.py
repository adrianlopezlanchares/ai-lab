import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.models import MLP
from src.utils.train_functions import train, evaluate
from src.utils.data_processing import create_train_and_test_datasets

building_parameters = pd.read_csv(
    "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/building_parameters/building_parameters.csv"
)
building_parameters.set_index("building_id", inplace=True)
processed_resstock = pd.read_csv(
    "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/resstock/resstock_processed.csv"
)
processed_resstock.set_index("bldg_id", inplace=True)

train_dataset, test_dataset = create_train_and_test_datasets(
    features=processed_resstock,
    labels=building_parameters,
    test_size=0.2,
)


def objective(trial: optuna.Trial) -> float:
    """
    Trains and evaluates a model using hyperparameters suggested by Optuna.

    Args:
        trial (optuna.Trial): The trial object that contains the hyperparameters to be optimized.

    Returns:
        float: The objective value to minimize (e.g., validation loss).
    """
    # Define optimization hyperparameters
    batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 10, 100, step=10)
    num_hidden_layers = trial.suggest_int("hidden_layers", 1, 3)
    hidden_layers = [
        trial.suggest_int(f"hidden_layer_{i}_size", 16, 128, step=16)
        for i in range(num_hidden_layers)
    ]

    # Define set hyperparameters
    input_size = len(processed_resstock.columns)
    output_size = len(building_parameters.columns)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    # Initialize model, loss function, and optimizer
    model = MLP(input_size, hidden_layers, output_size).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    # Evaluate the model
    average_loss = evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    return average_loss


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna/parameter_prediction_study.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=50)

    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)

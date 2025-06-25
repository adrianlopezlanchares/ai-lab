import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader


from typing import List, Tuple

from data_processing import get_cols
from dataset import Dataset
from models import LinearRegressionModel
from train_functions import train


def process_weather_data(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the weather data to extract relevant features.

    Args:
        weather (pd.DataFrame): The raw weather data.

    Returns:
        pd.DataFrame: Processed weather data with relevant features.
    """
    weather.rename(columns={"date_time": "timestamp"}, inplace=True)
    weather.set_index("timestamp", inplace=True)
    weather.index = pd.to_datetime(weather.index)

    weather = weather[["Dry Bulb Temperature [°C]", "Direct Normal Radiation [W/m2]"]]
    weather.rename(columns={"Dry Bulb Temperature [°C]": "temperature"}, inplace=True)

    new_row = pd.DataFrame(
        {"temperature": weather.iloc[0]["temperature"]},
        index=[pd.to_datetime("2018-01-01 00:00:00")],
    )
    weather = pd.concat([new_row, weather])

    return weather


def get_consumption_timeseries(
    resstock: pd.DataFrame, building_data: pd.DataFrame, building_id: str
) -> pd.DataFrame:
    """Gets the consumption timeseries for a specific building.

    Args:
        resstock (pd.DataFrame): Resstock data containing building information.
        building_data (pd.DataFrame): DataFrame containing building-specific data.
        building_id (str): The ID of the building for which to retrieve the timeseries.

    Returns:
        pd.DataFrame: A DataFrame containing the consumption timeseries for the specified building.
    """
    col_to_use = get_cols(resstock, building_id)
    consumption_timeseries = building_data[["timestamp", col_to_use]]

    consumption_timeseries["timestamp"] = pd.to_datetime(
        consumption_timeseries["timestamp"]
    )
    consumption_timeseries.set_index("timestamp", inplace=True)
    consumption_timeseries = consumption_timeseries.resample("H").mean()
    consumption_timeseries = consumption_timeseries[col_to_use]

    consumption_timeseries = pd.DataFrame(consumption_timeseries)
    consumption_timeseries.rename(columns={col_to_use: "consumption"}, inplace=True)

    return consumption_timeseries


def get_train_and_test_datasets(
    building_data: pd.DataFrame,
    resstock: pd.DataFrame,
    weather: pd.DataFrame,
    bldg_id: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gets the training and testing data for the MLP model.

    Args:
        building_data (pd.DataFrame): DataFrame containing building-specific data.
        resstock (pd.DataFrame): Resstock data containing building information.
        weather (pd.DataFrame): DataFrame containing weather data.
        bldg_id (str): The ID of the building for which to retrieve the data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the training data, training labels,
        testing data, and testing labels.
    """
    temp_timeseries = building_data[
        ["timestamp", "out.zone_mean_air_temp.conditioned_space.c"]
    ].copy()

    # Rename columns for clarity
    temp_timeseries.columns = ["timestamp", "indoor_temp"]

    consumption_timeseries = get_consumption_timeseries(
        resstock, building_data, bldg_id
    )

    exog_variables = exog_variables = pd.merge(
        consumption_timeseries,
        weather,
        left_index=True,
        right_index=True,
    )

    train_size = 0.8

    endog_train = temp_timeseries[: int(train_size * len(temp_timeseries))]
    endog_test = temp_timeseries[int(train_size * len(temp_timeseries)) :]

    exog_train = exog_variables[: int(train_size * len(exog_variables))]
    exog_test = exog_variables[int(train_size * len(exog_variables)) :]

    endog_train.set_index("timestamp", inplace=True)
    endog_test.set_index("timestamp", inplace=True)

    train_data = pd.merge(endog_train, exog_train, left_index=True, right_index=True)
    test_data = pd.merge(endog_test, exog_test, left_index=True, right_index=True)

    train_labels = train_data["indoor_temp"].values
    test_labels = test_data["indoor_temp"].values

    # Shift setpoint so that each column has the value of the previous hour
    train_data["indoor_temp"] = train_data["indoor_temp"].shift(1)
    test_data["indoor_temp"] = test_data["indoor_temp"].shift(1)

    # Create new lag_temperature column with lag 1
    train_data["lag_temperature"] = train_data["temperature"].shift(1)
    test_data["lag_temperature"] = test_data["temperature"].shift(1)

    train_data.ffill(inplace=True)
    test_data.ffill(inplace=True)
    train_data.bfill(inplace=True)
    test_data.bfill(inplace=True)

    return train_data.values, train_labels, test_data.values, test_labels


def main():

    resstock = pd.read_parquet(
        "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/resstock.parquet"
    )

    weather = pd.read_csv(
        "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/G2500170_2018.csv",
    )

    weather = process_weather_data(weather)

    i = 0
    parameters = {}
    building_data_path = "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/building_data"
    for file in os.listdir(building_data_path):
        print(f"Processing file {i}/{len(os.listdir(building_data_path))}")
        i += 1

        if file.endswith(".parquet"):
            bldg_id = int(file.split("/")[-1].split(".")[0].split("-")[0])
            building_data = pd.read_parquet(os.path.join(building_data_path, file))

            train_data, train_labels, test_data, test_labels = (
                get_train_and_test_datasets(building_data, resstock, weather, bldg_id)
            )

            train_dataset = Dataset(train_data, train_labels)
            test_dataset = Dataset(test_data, test_labels)

            batch_size = 32

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            train_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

            model = LinearRegressionModel(
                input_size=train_data.shape[1], output_size=1, bias=False
            )

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

            num_epochs = 100
            train(model, train_loader, criterion, optimizer, num_epochs)

            # Get parameters
            building_params = []
            for param in model.parameters():
                for num in param:
                    try:
                        for p in num:
                            building_params.append(p.item())
                    except:
                        building_params.append(num.item())

            parameters[bldg_id] = building_params

    # Save parameters to a CSV file
    # Shape of parameters: {building_id: [param1, param2, ...]}
    parameters_df = pd.DataFrame.from_dict(parameters, orient="index")
    parameters_df.reset_index(inplace=True)
    parameters_df.rename(columns={"index": "building_id"}, inplace=True)
    parameters_df.to_csv(
        "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/building_parameters.csv",
        index=True,
    )


if __name__ == "__main__":
    main()

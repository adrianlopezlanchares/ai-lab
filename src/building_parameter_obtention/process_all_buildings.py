import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from typing import List, Tuple

from src.utils.data_processing import (
    get_cols,
    process_weather_data,
    get_consumption_timeseries,
    get_train_and_test_datasets,
)
from src.utils.dataset import Dataset
from src.utils.models import LinearRegressionModel
from src.utils.train_functions import train

import warnings

warnings.filterwarnings("ignore")


def main():

    resstock = pd.read_parquet(
        "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/resstock/resstock.parquet"
    )

    weather = pd.read_csv(
        "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/weather/G2500170_2018.csv",
    )

    weather = process_weather_data(weather)

    i = 0
    start = 4196
    parameters = {}
    building_data_path = "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/building_data"
    for file in os.listdir(building_data_path):
        try:
            if i < start:
                i += 1
                continue
            i += 1
            print(
                f"Processing file {i}/{len(os.listdir(building_data_path))}           ",
                end="\r",
            )

            if file.endswith(".parquet"):
                bldg_id = int(file.split("/")[-1].split(".")[0].split("-")[0])
                building_data = pd.read_parquet(os.path.join(building_data_path, file))

                train_data, train_labels, _, _ = get_train_and_test_datasets(
                    building_data, resstock, weather, bldg_id
                )

                train_dataset = Dataset(train_data, train_labels)

                batch_size = 32

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )

                model = LinearRegressionModel(
                    input_size=train_data.shape[1], output_size=1, bias=False
                )

                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

                num_epochs = 100
                train(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    num_epochs,
                    device=torch.device("cpu"),
                )

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

        except KeyboardInterrupt:
            print("\n")
            print(f"Saving progress... bldg_id: {bldg_id}, iteration: {i}")
            parameters_df = pd.DataFrame.from_dict(parameters, orient="index")
            parameters_df.reset_index(inplace=True)
            parameters_df.rename(columns={"index": "building_id"}, inplace=True)
            # Drop column named Unnamed: 0
            if "Unnamed: 0" in parameters_df.columns:
                parameters_df.drop(columns=["Unnamed: 0"], inplace=True)
            parameters_df.columns = [
                "building_id",
                "indoor_temp_param",
                "consumption_param",
                "ambient_temp_param",
                "direct_solar_radiation_param",
                "ambient_temp_lag_param",
            ]
            parameters_df.to_csv(
                f"/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/building_parameters/building_parameters_{start}_{i-1}.csv",
                index=True,
            )
            # stop program
            sys.exit(0)

    print("Finished.")
    parameters_df = pd.DataFrame.from_dict(parameters, orient="index")
    parameters_df.reset_index(inplace=True)
    parameters_df.rename(columns={"index": "building_id"}, inplace=True)
    # Drop column named Unnamed: 0
    if "Unnamed: 0" in parameters_df.columns:
        parameters_df.drop(columns=["Unnamed: 0"], inplace=True)
    parameters_df.columns = [
        "building_id",
        "indoor_temp_param",
        "consumption_param",
        "ambient_temp_param",
        "direct_solar_radiation_param",
        "ambient_temp_lag_param",
    ]
    parameters_df.to_csv(
        "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/building_parameters_building_parameters.csv",
        index=True,
    )


if __name__ == "__main__":
    main()

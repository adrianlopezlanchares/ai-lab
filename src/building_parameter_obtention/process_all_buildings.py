import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader


from typing import List, Tuple

from data_processing import (
    get_cols,
    process_weather_data,
    get_consumption_timeseries,
    get_train_and_test_datasets,
)
from dataset import Dataset
from models import LinearRegressionModel
from train_functions import train

import warnings

warnings.filterwarnings("ignore")


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
        print(
            f"Processing file {i}/{len(os.listdir(building_data_path))}           ",
            end="\r",
        )
        i += 1

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

    parameters_df = pd.DataFrame.from_dict(parameters, orient="index")
    parameters_df.reset_index(inplace=True)
    parameters_df.rename(columns={"index": "building_id"}, inplace=True)
    parameters_df.to_csv(
        "/Users/cocoloco/Library/Mobile Documents/com~apple~CloudDocs/Documents/ICAI/4o/AI Lab/data/building_parameters.csv",
        index=True,
    )


if __name__ == "__main__":
    main()

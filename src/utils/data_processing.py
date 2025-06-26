import pandas as pd
import os
import re
import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import Dataset

from dataset import Dataset

COLUMNS_PROCESSED = [
    "in.bedrooms_processed",  # int
    "in.duct_leakage_processed",
    "in.duct_insulation_processed",  # Separate into leakage / insulation: X% / (Uniinsulated/R-n)
    "in.duct_location_processed",  # Places of the house
    "in.geometry_floor_area_processed",  # 1000-1499, ...
    "in.geometry_stories_processed",  # int
    "in.geometry_wall_type_processed",  # Materials
    "in.ground_thermal_conductivity_processed",  # float
    "in.hvac_has_ducts_processed",  # Yes / No
    "in.insulation_ceiling_processed",  # R-n / Uninsulated
    "in.insulation_floor_processed",  # R-n / Uninsulated
    "in.insulation_foundation_wall_processed",  # Wall R-n, Exterior
    "in.insulation_roof_processed",  # Finished / Unfinished, R-n
    "in.insulation_wall_processed",  # Material, Uninsulated / R-n
    "in.occupants_processed",  # int
    "in.orientation_processed",  # North, Northwest...
    "in.roof_material_processed",  # Material
    "in.sqft_processed",  # int
    "in.windows_processed",  # Single/Double/Triple, Low-E/Clear, Metal/Non-metal, Air/Exterior Clear Storm, (L/M-Gain)
    "in.window_areas_processed",
    "in.vintage_processed",
]


def create_train_and_test_datasets(
    features: pd.DataFrame, labels: pd.DataFrame, test_size: float = 0.2
) -> Tuple[Dataset, Dataset]:
    """Creates a dataset for training and testing.

    Args:
        features (pd.DataFrame): DataFrame containing the input features.
        labels (pd.DataFrame): DataFrame containing the target labels.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and testing datasets.
    """
    features = features.to_numpy()
    labels = labels.to_numpy()

    train_features = features[: int(len(features) * (1 - test_size))]
    train_labels = labels[: int(len(labels) * (1 - test_size))]

    test_features = features[int(len(features) * (1 - test_size)) :]
    test_labels = labels[int(len(labels) * (1 - test_size)) :]

    train_dataset = Dataset(train_features, train_labels)
    test_dataset = Dataset(test_features, test_labels)

    return train_dataset, test_dataset


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


def train_test_split(
    windows: Tuple[np.ndarray, np.ndarray], test_size: float = 0.2, model: str = "mlp"
) -> Tuple[Dataset, Dataset]:
    """Split the dataset into training and testing sets.

    Args:
        windows (Tuple[np.ndarray, np.ndarray]): A tuple containing the features and labels.
            The first element is a 2D array of features, and the second element is a 1D array of labels.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        model (str, optional): Determines the type of dataset to return. Defaults to "mlp".

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and testing datasets.
    """
    X, y = windows

    # Calculate the number of test samples
    num_test_samples = int(len(X) * test_size)

    # Split the data into training and testing sets
    X_train, X_test = X[:-num_test_samples], X[-num_test_samples:]
    y_train, y_test = y[:-num_test_samples], y[-num_test_samples:]

    if model == "mlp":
        return Dataset((X_train, y_train)), Dataset((X_test, y_test))
    else:
        raise ValueError(f"Model {model} is not supported")


def get_cols(building_data: pd.DataFrame, building_id: int) -> str:
    """Get the output column name for the given building ID.

    Args:
        building_data (pd.DataFrame): The DataFrame containing building data.
        building_id (int): The ID of the building.

    Raises:
        ValueError: If the building ID is not found in the DataFrame.

    Returns:
        str: The output column name for the building's heating fuel type.
    """
    try:
        building = building_data.loc[building_id]
    except:
        raise ValueError(f"Building ID {building_id} not found in the dataset")

    match building["in.heating_fuel"]:
        case "Natural Gas":
            return "out.natural_gas.heating.energy_consumption"
        case "Electricity":
            return "out.electricity.heating.energy_consumption"
        case "Fuel Oil":
            return "out.fuel_oil.heating.energy_consumption"
        case "Propane":
            return "out.propane.heating.energy_consumption"
        case _:
            return "out.electricity.heating.energy_consumption"


def load_and_process_building(file_path: str, consumption_col: str) -> pd.DataFrame:
    """
    Load and process the building data from a parquet file.

    Args:
        file_path (str): The path to the parquet file containing building data.
        consumption_col (str): The column name for the energy consumption data.

    Returns:
        pd.DataFrame: A DataFrame containing the processed building data.
    """
    df = pd.read_parquet(file_path)

    temp_timeseries = df[
        [
            "timestamp",
            "out.zone_mean_air_temp.conditioned_space.c",
            consumption_col,
        ]
    ].copy()

    temp_timeseries.columns = ["timestamp", "indoor_temp", "consumption"]

    return temp_timeseries


def load_and_process_all_buildings(resstock: pd.DataFrame, tanda: int) -> pd.DataFrame:
    """
    Loads and processes all files in the 'data/buildings' directory.

    Args:
        resstock (pd.DataFrame): The ResStock dataset containing building information.
        tanda (int): An integer indicating which subset of buildings to process.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data from all buildings.
    """
    all_buildings_data = pd.DataFrame()

    path = "/Users/adrian/Documents/ICAI/4o/AI Lab/data/buildings"

    lower_limit = 0
    upper_limit = 0

    match tanda:
        case 0:
            lower_limit = 0
            upper_limit = 1000
        case 1:
            lower_limit = 1000
            upper_limit = 2000
        case 2:
            lower_limit = 2000
            upper_limit = 3000
        case 3:
            lower_limit = 3000
            upper_limit = 4000
        case 4:
            lower_limit = 4000
            upper_limit = 5000
        case 5:
            lower_limit = 5000
            upper_limit = 6000
        case 6:
            lower_limit = 6000
            upper_limit = 7000
        case 7:
            lower_limit = 7000
            upper_limit = 8000
        case 8:
            lower_limit = 8000
            upper_limit = 9000
        case 9:
            lower_limit = 9000
            upper_limit = 10000
        case 10:
            lower_limit = 10000
            upper_limit = 11000
        case 11:
            lower_limit = 11000
            upper_limit = len(os.listdir(path))

    i = 0
    for file in os.listdir(path):

        if i < lower_limit or i >= upper_limit:
            i += 1
            continue

        i += 1

        print(
            f"Processing file {i}/{upper_limit}                  ",
            end="\r",
        )
        if file.endswith(".parquet"):
            bldg_id = int(file.split("/")[-1].split(".")[0].split("-")[0])
            consumption_col = get_cols(resstock, bldg_id)
            building_data = load_and_process_building(
                os.path.join(path, file), consumption_col
            )
            all_buildings_data = pd.concat(
                [all_buildings_data, building_data], ignore_index=False
            )

    return all_buildings_data


def merge_building_data() -> pd.DataFrame:
    """
    Merge all building data from the 'data/processed_buildings' directory into a single DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the merged building data.
    """
    path = "/Users/adrian/Documents/ICAI/4o/AI Lab/data/processed_buildings"
    all_buildings_data = pd.DataFrame()

    i = 0
    for file in os.listdir(path):
        i += 1
        print(
            f"Processing file {i}/{len(os.listdir(path))}                  ",
            end="\r",
        )
        if file.endswith(".csv"):
            building_data = pd.read_csv(os.path.join(path, file))
            all_buildings_data = pd.concat(
                [all_buildings_data, building_data], ignore_index=True
            )

    return all_buildings_data


def create_windows(
    timeseries: pd.DataFrame, resstock: pd.DataFrame, window_size: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(window_size, len(timeseries)):
        print(f"{i}/{len(timeseries)}               ", end="\r")

        bldg_id = timeseries.iloc[i]["bldg_id"]
        if timeseries.iloc[i - window_size]["bldg_id"] != bldg_id:
            continue

        static_features = resstock.loc[bldg_id].values

        ts_window = timeseries[i - window_size : i][
            ["indoor_temp", "consumption"]
        ].values

        target = timeseries.iloc[i]["indoor_temp"]
        features = np.concatenate([ts_window.flatten(), static_features])

        X.append(features)
        y.append(target)

    return np.array(X), np.array(y)


def process_resstock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the ResStock dataset by applying various transformations to specific columns.
    Each processed column is suffixed with '_processed'.
    Columns called "heating_targets" and "cooling_targets" are added to the DataFrame,
    which are a copy of "in.heating_setpoint" and "in.cooling_setpoint" respectively.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = _process_bedrooms(df)
    df = _process_duct_leakage_and_insulation(df)
    df = _process_duct_location(df)
    df = _process_geometry_floor_area(df)
    df = _process_stories(df)
    df = _process_wall_type(df)
    df = _process_ground_thermal_conductivity(df)
    df = _process_has_ducts(df)
    df = _process_ceiling_insulation(df)
    df = _process_floor_insulation(df)
    df = _process_foundation_wall_insulation(df)
    df = _process_roof_insulation(df)
    df = _process_wall_insulation(df)
    df = _process_occupants(df)
    df = _process_orientation(df)
    df = _process_roof_material(df)
    df = _process_sqft(df)
    df = _process_windows(df)
    df = _process_window_areas(df)
    df = _process_vintage(df)
    df = _process_heating_targets(df)
    df = _process_cooling_targets(df)

    df = df[COLUMNS_PROCESSED]

    return df


def _process_bedrooms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.bedrooms' column in the ResStock dataset.
    Turns it into 'in.bedrooms_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.bedrooms_processed' column.
    """

    def extract_bedrooms(text):
        return int(text)

    df["in.bedrooms_processed"] = df["in.bedrooms"].apply(extract_bedrooms)

    return df


def _process_duct_leakage_and_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.duct_leakage_and_insulation' column in the ResStock dataset.
    Turns it into two separate columns: 'in.duct_leakage_processed' and
    'in.duct_insulation_processed'.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.duct_leakage_processed'
                      and 'in.duct_insulation_processed' columns.
    """
    pattern_leakage = r"(\d+)%"
    pattern_insulation = r"R-(\d+)"

    def extract_leak_percentage(text):
        if text == "None":
            return 0

        leakage_text, _ = text.split(",")

        leakage_match = re.findall(pattern_leakage, leakage_text)
        if leakage_match:
            return int(leakage_match[0])
        return 0

    def extract_insulation_number(text):
        if text == "None":
            return 0

        _, insulation_text = text.split(",")
        insulation_match = re.findall(pattern_insulation, insulation_text)
        if insulation_match:
            return int(insulation_match[0])
        return 0

    df["in.duct_leakage_processed"] = df["in.duct_leakage_and_insulation"].apply(
        extract_leak_percentage
    )
    df["in.duct_insulation_processed"] = df["in.duct_leakage_and_insulation"].apply(
        extract_insulation_number
    )

    return df


def _process_duct_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.duct_location' column in the ResStock dataset.
    Turns it into 'in.duct_location_processed' column with numeric values.

    Uses number encoding, NOT one-hot encoding.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.duct_location_processed' column.
    """

    def extract_duct_location(text):
        if text == "None":
            return 0
        if text == "Unheated Basement":
            return 1
        if text == "Heated Basement":
            return 2
        if text == "Living Space":
            return 3
        if text == "Crawlspace":
            return 4
        if text == "Attic":
            return 5
        if text == "Garage":
            return 6

    df["in.duct_location_processed"] = df["in.duct_location"].apply(
        extract_duct_location
    )

    return df


def _process_geometry_floor_area(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.geometry_floor_area' column in the ResStock dataset.
    Turns it into 'in.geometry_floor_area_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.geometry_floor_area_processed' column.
    """
    pattern = r"(\d+)-(\d+)"

    def extract_floor_area(text):
        if text == "4000+":
            return 4000
        match = re.findall(pattern, text)
        if match:
            return int(int(match[0][0]) + int(match[0][1]) // 2)
        return 0

    df["in.geometry_floor_area_processed"] = df["in.geometry_floor_area"].apply(
        extract_floor_area
    )

    return df


def _process_stories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.stories' column in the ResStock dataset.
    Turns it into 'in.stories_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.stories_processed' column.
    """

    def extract_stories(text):
        return int(text)

    df["in.geometry_stories_processed"] = df["in.geometry_stories"].apply(
        extract_stories
    )

    return df


def _process_wall_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.geometry_wall_type' column in the ResStock dataset.
    Turns it into 'in.geometry_wall_type_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.wall_type_processed' column.
    """

    def extract_wall_type(text):
        if text == "None":
            return 0
        if text == "Wood Frame":
            return 1
        if text == "Brick":
            return 2
        if text == "Steel Frame":
            return 3
        if text == "Concrete":
            return 4

    df["in.geometry_wall_type_processed"] = df["in.geometry_wall_type"].apply(
        extract_wall_type
    )

    return df


def _process_ground_thermal_conductivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.ground_thermal_conductivity' column in the ResStock dataset.
    Turns it into 'in.ground_thermal_conductivity_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.ground_thermal_conductivity_processed' column.
    """

    def extract_ground_thermal_conductivity(text):
        return float(text)

    df["in.ground_thermal_conductivity_processed"] = df[
        "in.ground_thermal_conductivity"
    ].apply(extract_ground_thermal_conductivity)

    return df


def _process_has_ducts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.hvac_has_ducts' column in the ResStock dataset.
    Turns it into 'in.hvac_has_ducts_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.has_ducts_processed' column.
    """

    def extract_has_ducts(text):
        if text == "Yes":
            return 1
        if text == "No":
            return 0

    df["in.hvac_has_ducts_processed"] = df["in.hvac_has_ducts"].apply(extract_has_ducts)

    return df


def _process_ceiling_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_ceiling' column in the ResStock dataset.
    Turns it into 'in.insulation_ceiling_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_ceiling_processed' column.
    """

    def extract_ceiling_insulation(text):
        if text == "None":
            return 0
        if text == "Uninsulated":
            return 1
        if text == "R-7":
            return 2
        if text == "R-13":
            return 3
        if text == "R-19":
            return 4
        if text == "R-30":
            return 5
        if text == "R-38":
            return 6
        if text == "R-49":
            return 7

    df["in.insulation_ceiling_processed"] = df["in.insulation_ceiling"].apply(
        extract_ceiling_insulation
    )

    return df


def _process_floor_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_floor' column in the ResStock dataset.
    Turns it into 'in.insulation_floor_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_floor_processed' column.
    """

    def extract_floor_insulation(text):
        if text == "None":
            return 0
        if text == "Uninsulated":
            return 1
        if text == "Ceiling R-13":
            return 2
        if text == "Ceiling R-19":
            return 3
        if text == "Ceiling R-30":
            return 4

    df["in.insulation_floor_processed"] = df["in.insulation_floor"].apply(
        extract_floor_insulation
    )

    return df


def _process_foundation_wall_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_foundation_wall' column in the ResStock dataset.
    Turns it into 'in.insulation_foundation_wall_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_foundation_wall_processed' column.
    """

    def extract_foundation_wall_insulation(text):
        if text == "None":
            return 0
        if text == "Uninsulated":
            return 1

        split_text = text.split(",")
        if split_text[0] == "Wall R-5":
            return 2
        if split_text[0] == "Wall R-10":
            return 3
        if split_text[0] == "Wall R-15":
            return 4

    df["in.insulation_foundation_wall_processed"] = df[
        "in.insulation_foundation_wall"
    ].apply(extract_foundation_wall_insulation)

    return df


def _process_roof_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_roof' column in the ResStock dataset.
    Turns it into 'in.insulation_roof_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_roof_processed' column.
    """

    def extract_roof_insulation(text):
        split_text = text.split(",")

        if split_text[0] == "Unfinished":
            return 0
        if split_text[1] == " Uninsulated":
            return 1
        if split_text[1] == " R-7":
            return 2
        if split_text[1] == " R-13":
            return 3
        if split_text[1] == " R-19":
            return 4
        if split_text[1] == " R-30":
            return 5
        if split_text[1] == " R-38":
            return 6
        if split_text[1] == " R-49":
            return 7

    df["in.insulation_roof_processed"] = df["in.insulation_roof"].apply(
        extract_roof_insulation
    )

    return df


def _process_wall_insulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.insulation_wall' column in the ResStock dataset.
    Turns it into 'in.insulation_wall_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.insulation_wall_processed' column.
    """

    def extract_wall_insulation(text):
        text_split = text.split(",")

        if text_split[-1] == " Uninsulated":
            return 0
        if text_split[-1] == " R-7":
            return 1
        if text_split[-1] == " R-11":
            return 2
        if text_split[-1] == " R-15":
            return 3
        if text_split[-1] == " R-19":
            return 4

    df["in.insulation_wall_processed"] = df["in.insulation_wall"].apply(
        extract_wall_insulation
    )

    return df


def _process_occupants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.occupants' column in the ResStock dataset.
    Turns it into 'in.occupants_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.occupants_processed' column.
    """

    def extract_occupants(text):
        if text == "10+":
            return 10
        return int(text)

    df["in.occupants_processed"] = df["in.occupants"].apply(extract_occupants)

    return df


def _process_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.orientation' column in the ResStock dataset.
    Turns it into 'in.orientation_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.orientation_processed' column.
    """

    def extract_orientation(text):
        # Options:
        # North        2176
        # South        2157
        # West         1949
        # East         1919
        # Southeast     893
        # Northwest     883
        # Northeast     881
        # Southwest     849

        if text == "North":
            return 0
        if text == "Northeast":
            return 1
        if text == "East":
            return 2
        if text == "Southeast":
            return 3
        if text == "South":
            return 4
        if text == "Southwest":
            return 5
        if text == "West":
            return 6
        if text == "Northwest":
            return 7

    df["in.orientation_processed"] = df["in.orientation"].apply(extract_orientation)

    return df


def _process_roof_material(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.roof_material' column in the ResStock dataset.
    Turns it into 'in.roof_material_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.roof_material_processed' column.
    """

    def extract_roof_material(text):
        # Options:
        #         Asphalt Shingles, Medium    5620
        # Composition Shingles        4199
        # Wood Shingles                705
        # Metal, Dark                  473
        # Tile, Clay or Ceramic        426
        # Slate                        272
        # Tile, Concrete                12

        if text == "Asphalt Shingles, Medium":
            return 0
        if text == "Composition Shingles":
            return 1
        if text == "Wood Shingles":
            return 2
        if text == "Metal, Dark":
            return 3
        if text == "Tile, Clay or Ceramic":
            return 4
        if text == "Slate":
            return 5
        if text == "Tile, Concrete":
            return 6

    df["in.roof_material_processed"] = df["in.roof_material"].apply(
        extract_roof_material
    )

    return df


def _process_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.sqft' column in the ResStock dataset.
    Turns it into 'in.sqft_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.sqft_processed' column.
    """

    def extract_sqft(text):
        return int(text)

    df["in.sqft_processed"] = df["in.sqft"].apply(extract_sqft)

    return df


def _process_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.windows' column in the ResStock dataset.
    Turns it into 'in.windows_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.windows_processed' column.
    """

    def extract_windows(text):
        split_text = text.split(",")

        if split_text[0] == "Single":
            return 1
        if split_text[0] == "Double":
            return 2
        if split_text[0] == "Triple":
            return 3

    df["in.windows_processed"] = df["in.windows"].apply(extract_windows)

    return df


def _process_window_areas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.window_areas' column in the ResStock dataset.
    Turns it into 'in.window_areas_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.window_areas_processed' column.
    """

    def extract_window_areas(text):
        # Extract only the first number that appears after the first F
        match = re.search(r"F(\d+)", text)
        if match:
            return int(match.group(1))
        return 0

    df["in.window_areas_processed"] = df["in.window_areas"].apply(extract_window_areas)

    return df


def _process_vintage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.vintage' column in the ResStock dataset.
    Turns it into 'in.vintage_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.vintage_processed' column.
    """

    def extract_vintage(text):
        if text == "<1940":
            return 1930

        text = text[:-1]

        return int(text)

    df["in.vintage_processed"] = df["in.vintage"].apply(extract_vintage)

    return df


def _process_heating_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.heating_setpoint' column in the ResStock dataset.
    Turns it into 'in.heating_setpoint_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.heating_setpoint_processed' column.
    """

    def extract_heating_targets(text):
        text = text[:-1]
        return int(text)

    df["heating_targets"] = df["in.heating_setpoint"].apply(extract_heating_targets)

    return df


def _process_cooling_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'in.cooling_setpoint' column in the ResStock dataset.
    Turns it into 'in.cooling_setpoint_processed' column with numeric values.

    Args:
        df (pd.DataFrame): The ResStock dataset.

    Returns:
        pd.DataFrame: The DataFrame with the processed 'in.cooling_setpoint_processed' column.
    """

    def extract_cooling_targets(text):
        text = text[:-1]
        return int(text)

    df["cooling_targets"] = df["in.cooling_setpoint"].apply(extract_cooling_targets)

    return df

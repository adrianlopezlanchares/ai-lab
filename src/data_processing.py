import pandas as pd
import os


def get_cols(building_data: pd.DataFrame, building_id: int) -> str:

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
        ["timestamp", "out.zone_mean_air_temp.conditioned_space.c", consumption_col]
    ].copy()

    temp_timeseries.columns = ["timestamp", "indoor_temp", "consumption"]

    return temp_timeseries


def load_and_process_all_buildings(resstock: pd.DataFrame) -> pd.DataFrame:
    """
    Loads and processes all files in the 'data/buildings' directory.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data from all buildings.
    """
    all_buildings_data = pd.DataFrame()

    path = "/Users/adrian/Documents/ICAI/4o/AI Lab/data/buildings"

    i = 0
    for file in os.listdir(path):
        i += 1
        print(
            f"Processing file {i}/{len(os.listdir(path))}                  ",
            end="\r",
        )
        if file.endswith(".parquet"):
            bldg_id = int(file.split("/")[-1].split(".")[0].split("-")[0])
            consumption_col = get_cols(resstock, bldg_id)
            building_data = load_and_process_building(
                os.path.join(path, file), consumption_col
            )
            all_buildings_data = pd.concat(
                [all_buildings_data, building_data], ignore_index=True
            )

    return all_buildings_data

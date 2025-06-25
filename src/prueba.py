import pandas as pd
import pickle

from data_processing import process_resstock_data, create_windows


if __name__ == "__main__":
    resstock = pd.read_parquet(
        "./data/MA_baseline_metadata_and_annual_results.parquet", engine="fastparquet"
    )
    buildings0 = pd.read_csv("./data/processed_buildings/buildings_tanda0.csv")

    columns_processed = [
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

    resstock_processed = process_resstock_data(resstock)
    windows0 = create_windows(buildings0, resstock_processed, window_size=8)

    # Save to pickle file
    with open(
        "./data/training_windows/windows0.pkl",
        "wb",
    ) as f:
        pickle.dump(windows0, f)

    print("Windows created and saved successfully.")

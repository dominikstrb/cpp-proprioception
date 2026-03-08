import os
from pathlib import Path
import pandas as pd
import numpy as np


def load_data(pos, data_dir="data/MLMC_Data/"):  # Update if needed
    csv_filename = f"mabel_con_xy_{pos[0]}_{pos[1]}_individualised.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    # --- Step 3: Load CSV ---
    df = pd.read_csv(csv_path)
    print(f"DataFrame loaded from {csv_filename} with shape:", df.shape)

    return df


def preprocess_data(df, lag=1):
    # --- Step 1: Baseline subtract per Trajectory_ID ---
    df = df.groupby("Trajectory_ID", group_keys=False).apply(subtract_baseline)

    # --- Step 2: Filter by Time_Block (10s–30s) ---
    # Identify the start time when right hand velocity exceeds 0 for each trajectory
    start_times = df[abs(df["Left_YVel"]) > 0.3].groupby("Trajectory_ID")["Time"].min()

    # Map the start times back to the main DataFrame
    df["Start_Time"] = df["Trajectory_ID"].map(start_times)

    # Calculate time relative to the detected movement onset
    df["Time_Block"] = df["Time"] - df["Start_Time"]
    df = df[(df["Time_Block"] >= 0) & (df["Time_Block"] < 10)]

    # --- Step 3: Flip and convert to cm ---
    df["Left_HandX"] *= -1  # Flip X axis for left hand
    for col in ["Right_HandX", "Left_HandX", "Right_HandY", "Left_HandY"]:
        df[col] *= 100  # Convert to cm

    downsampled_blocks = (
        df.groupby("Trajectory_ID", group_keys=False)
        .apply(lambda b: downsample_uniform_sample(b, factor=55))
        .reset_index(drop=True)
    )

    # --- Step 4: Truncate all blocks to match minimum sample length ---
    block_counts = downsampled_blocks["Trajectory_ID"].value_counts().sort_index()
    min_count = block_counts.min()

    # --- Step 5: Stack arrays for all axes (optional if needed later) ---
    stacked = {}
    for col in ["Right_HandX", "Left_HandX", "Right_HandY", "Left_HandY"]:
        stacked[col] = []

    for _, block_df in downsampled_blocks.groupby("Trajectory_ID"):
        block_df = block_df.iloc[:min_count]
        for col in stacked:
            stacked[col].append(block_df[col].to_numpy())

    for col in stacked:
        stacked[col] = np.stack(stacked[col])  # (blocks, time)

    # --- Step 6: Build final stacked array (blocks, time, 2) for selected axis ---
    # For X dimension
    block_arrays_x = []
    for block_num, block_df in downsampled_blocks.groupby("Trajectory_ID"):
        block_df = block_df.iloc[:min_count]
        block_arrays_x.append(block_df[["Right_HandX", "Left_HandX"]].to_numpy())

    x_array = np.stack(block_arrays_x, axis=0)  # shape: (blocks, time, 2)
    print("Final shape of X array:", x_array.shape)

    # For Y dimension
    block_arrays_y = []
    for block_num, block_df in downsampled_blocks.groupby("Trajectory_ID"):
        block_df = block_df.iloc[:min_count]
        block_arrays_y.append(block_df[["Right_HandY", "Left_HandY"]].to_numpy())

    y_array = np.stack(block_arrays_y, axis=0)  # shape: (blocks, time, 2)
    print("Final shape of Y array:", y_array.shape)

    if lag > 0:
        x_array = np.stack([x_array[:, :-lag, 0], x_array[:, lag:, 1]], axis=-1)
        y_array = np.stack([y_array[:, :-lag, 0], y_array[:, lag:, 1]], axis=-1)

    xy_array = np.concatenate([x_array, y_array], axis=-1)
    print("Final shape of XY array:", xy_array.shape)

    return xy_array


def subtract_baseline(group):
    group = group.copy()
    for col in ["Right_HandX", "Left_HandX", "Right_HandY", "Left_HandY"]:
        baseline = group[col].iloc[:5000].mean()
        group[col] -= baseline
    return group


def downsample_uniform_sample(block_df, factor=55):
    # Get the trajectory ID
    traj_id = block_df["Trajectory_ID"].iloc[0]

    # Take every 'factor'-th row
    downsampled_df = block_df.iloc[::factor].copy().reset_index(drop=True)
    downsampled_df["Trajectory_ID"] = traj_id  # Ensure ID is retained correctly

    return downsampled_df


def load_multisensory_data(base_path="data"):
    data_path = Path(base_path) / "multisensory/df_all_phases_all_participants.csv"
    df = pd.read_csv(data_path)
    print(f"Multisensory DataFrame loaded from {data_path} with shape:", df.shape)
    return df


def preprocess_multisensory_data(df, participant, condition, vis_noise, phase="pre_cal"):
    df_sub = df[
        (df["participant"] == participant)
        & (df["type"] == condition)
        & (df["vis_noise"] == vis_noise)
        & (df["phase"] == phase)
    ]

    data = []
    for trial_num, df_trial in df_sub.groupby("trial_number"):
        data.append(np.array([df_trial["cursory_pos"], df_trial["lefty_pos"]]).T)

    lens = [d.shape[0] for d in data]
    min_len = min(lens)
    data = [d[:min_len] for d in data]

    return np.stack(data)

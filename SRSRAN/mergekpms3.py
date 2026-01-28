import pickle
import pandas as pd

# Resampling function with interpolation
def resample_to_one_per_second(df, time_col='_time', ue_col='ue_id', freq='1S'):
    if time_col not in df.columns or ue_col not in df.columns:
        print(f"Columns {time_col} or {ue_col} are missing in the DataFrame.")
        return pd.DataFrame()

    # Convert the time column to datetime format and drop duplicate timestamps within each ue_id
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.drop_duplicates(subset=[time_col, ue_col])

    # Separate numeric columns for resampling
    numeric_df = df.select_dtypes(include='number').copy()
    numeric_df[time_col] = df[time_col].values
    numeric_df[ue_col] = df[ue_col].values

    # Resample each group based on ue_id and interpolate only numeric columns
    resampled_dfs = []
    for ue_id, group in numeric_df.groupby(ue_col):
        group = group.set_index(time_col).resample(freq).mean()  # Resample by time column
        # Interpolate only if there are numeric columns
        if not group.select_dtypes(include='number').empty:
            group = group.interpolate(method='linear')  # Fill NaNs by interpolation
        group[ue_col] = ue_id  # Add ue_id back after resampling
        resampled_dfs.append(group.reset_index())

    numeric_resampled = pd.concat(resampled_dfs, ignore_index=True)

    # Concatenate non-numeric columns, excluding `ue_id` and `_time`
    non_numeric_df = df.select_dtypes(exclude='number').drop(columns=[ue_col, time_col], errors='ignore').reset_index(drop=True)
    non_numeric_resampled = non_numeric_df.reindex(numeric_resampled.index).fillna(method='ffill').reset_index(drop=True)

    # Concatenate numeric and non-numeric data
    final_resampled_df = pd.concat([numeric_resampled, non_numeric_resampled], axis=1)

    return final_resampled_df

# Load the datasets
with open('df_iperf_test.pkl', 'rb') as file:
    iperf = pickle.load(file)
with open('df_kpms_test.pkl', 'rb') as file:
    kpms = pickle.load(file)
with open('df_latency_test_combined.pkl', 'rb') as file:
    latency = pickle.load(file)
snr = pd.read_csv('pucch_data_new.csv')

# Convert and clean timestamps with consistent rounding
iperf['_time'] = pd.to_datetime(iperf['_time'].astype(str).str.split('_').str[0], errors='coerce').dt.floor('S')
kpms['_time'] = pd.to_datetime(kpms['_time'].astype(str), errors='coerce').dt.floor('S')
latency['_time'] = pd.to_datetime(latency['_time'].astype(str), errors='coerce').dt.floor('S')
snr['timestamp'] = pd.to_datetime(snr['timestamp'].astype(str), errors='coerce').dt.floor('S')

# Rename columns for consistency
iperf.rename(columns={'ue_nr': 'ue_id'}, inplace=True)
kpms.rename(columns={'ue_nr': 'ue_id'}, inplace=True)
snr.rename(columns={'timestamp': '_time', 'rnti': 'ue_id'}, inplace=True)

# Apply the resampling function to each DataFrame
iperf_resampled = resample_to_one_per_second(iperf)
kpms_resampled = resample_to_one_per_second(kpms)
latency_resampled = resample_to_one_per_second(latency)
snr_resampled = resample_to_one_per_second(snr)

# Filter out empty DataFrames if any columns were missing
dataframes = [df for df in [iperf_resampled, kpms_resampled, latency_resampled, snr_resampled] if not df.empty]

# Ensure consistent data types for '_time' and 'ue_id'
for df in dataframes:
    df['_time'] = pd.to_datetime(df['_time'], errors='coerce')  # Ensure '_time' is datetime
    df['ue_id'] = df['ue_id'].astype(str)  # Convert 'ue_id' to string

# Remove duplicate columns to prevent conflicts
for i, df in enumerate(dataframes):
    dataframes[i] = df.loc[:, ~df.columns.duplicated()]

# Merge the resampled DataFrames on '_time' and 'ue_id' columns
merged_df = dataframes[0]
for df in dataframes[1:]:
    merged_df = pd.merge(merged_df, df, on=['_time', 'ue_id'], how='outer')

# Sort and display the merged DataFrame
merged_df = merged_df.sort_values(by=['_time', 'ue_id']).reset_index(drop=True)
print(merged_df.head())

# Threshold for dropping rows with fewer than the required number of non-NaN values
threshold = len(merged_df.columns) // 2  # Adjust as needed

# Drop rows with fewer than the threshold non-NaN values
cleaned_df = merged_df.dropna(thresh=threshold)

# Save the cleaned DataFrame to a file if needed, e.g., CSV
cleaned_df.to_csv("merged_data.csv", index=False)

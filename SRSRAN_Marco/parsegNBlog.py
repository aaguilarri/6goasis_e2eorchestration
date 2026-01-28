import re
import pandas as pd
from datetime import datetime, timedelta

def search_pucch_data(file_path):
    """
    Searches for lines containing 'PUCCH' and extracts relevant data between a PUCCH line and the next timestamp.

    Args:
    - file_path (str): Path to the log file.

    Returns:
    - DataFrame with columns 'timestamp', 'rnti', 'epre', 'rsrp', 'sinr', 'cfo', 'prb1', 'prb2', 'slot', 'bwp1', 'bwp2'
    """
    pucch_data = []

    # Regular expression patterns to match the needed fields
    timestamp_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}'
    rnti_pattern = r'rnti=0x(\d{4})'
    epre_pattern = r'epre=\+?(-?\d+\.\d+)dB'
    rsrp_pattern = r'rsrp=\+?(-?\d+\.\d+)dB'
    sinr_pattern = r'sinr=\+?(-?\d+\.\d+)dB'
    cfo_pattern = r'cfo=\+?(-?\d+\.\d+)Hz'
    prb1_pattern = r'prb1=(\d+)'
    prb2_pattern = r'prb2=(\d+|na)'
    slot_pattern = r'slot=(\d+\.\d+)'  # Modified to capture decimal slot values
    bwp_pattern = r'bwp=\[(\d+),\s*(\d+)\)'

    with open(file_path, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if 'PUCCH' in line:
                # Extract timestamp from the current PUCCH line
                timestamp_match = re.match(timestamp_pattern, line)
                if not timestamp_match:
                    i += 1
                    continue
                # Parse the timestamp and add two hours
                original_timestamp = datetime.strptime(timestamp_match.group(0), "%Y-%m-%dT%H:%M:%S.%f")
                adjusted_timestamp = original_timestamp #+ timedelta(hours=2)
                timestamp = adjusted_timestamp.isoformat()

                # Initialize variables for the fields
                rnti, epre, rsrp, sinr, cfo, prb1, prb2, slot, bwp1, bwp2 = None, None, None, None, None, None, None, None, None, None

                # Process subsequent lines until a new timestamp is found
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()

                    # Stop if the next line starts with a new timestamp (end of current PUCCH block)
                    if re.match(timestamp_pattern, next_line):
                        break

                    # Extract fields from the line
                    rnti_match = re.search(rnti_pattern, next_line)
                    epre_match = re.search(epre_pattern, next_line)
                    rsrp_match = re.search(rsrp_pattern, next_line)
                    sinr_match = re.search(sinr_pattern, next_line)
                    cfo_match = re.search(cfo_pattern, next_line)
                    prb1_match = re.search(prb1_pattern, next_line)
                    prb2_match = re.search(prb2_pattern, next_line)
                    slot_match = re.search(slot_pattern, next_line)
                    bwp_match = re.search(bwp_pattern, next_line)

                    # Update variables if matches are found
                    if rnti_match:
                        rnti_value = int(rnti_match.group(1), 16)
                        rnti = 1 if rnti_value == 0x4601 else 3 if rnti_value == 0x4602 else 2
                    if epre_match:
                        epre = float(epre_match.group(1))
                    if rsrp_match:
                        rsrp = float(rsrp_match.group(1))
                    if sinr_match:
                        sinr = float(sinr_match.group(1))
                    if cfo_match:
                        cfo = float(cfo_match.group(1))
                    if prb1_match:
                        prb1 = int(prb1_match.group(1))
                    if prb2_match and prb2_match.group(1) != 'na':
                        prb2 = int(prb2_match.group(1))
                    if slot_match:
                        slot = float(slot_match.group(1))  # Directly capture the decimal slot value
                    if bwp_match:
                        bwp1 = int(bwp_match.group(1))
                        bwp2 = int(bwp_match.group(2))

                    i += 1

                # Append the extracted data to the list
                pucch_data.append({
                    'timestamp': timestamp,
                    'rnti': rnti,
                    'epre': epre,
                    'rsrp': rsrp,
                    'sinr': sinr,
                    'cfo': cfo,
                    'prb1': prb1,
                    'prb2': prb2,
                    'slot': slot,
                    'bwp1': bwp1,
                    'bwp2': bwp2
                })

            # Move to the next line
            i += 1

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(pucch_data, columns=['timestamp', 'rnti', 'epre', 'rsrp', 'sinr', 'cfo', 'prb1', 'prb2', 'slot', 'bwp1', 'bwp2'])
    return df

# Example usage
file_path = "gnb.log"  # Update this to the path of your log file
pucch_df = search_pucch_data(file_path)
print(pucch_df.head())
# Save the DataFrame to a CSV file
csv_file_path = "pucch_data_new.csv"
pucch_df.to_csv(csv_file_path, index=False)

print(f"Data has been saved to {csv_file_path}")

import pandas as pd
import numpy as np

def convert_to_milliseconds(time_str: str) -> int:
    parts = time_str.split(":")
    if len(parts) != 3:
        raise ValueError("Formato non valido. Deve essere 'minuti:secondi:millisecondi'")
    
    minutes = int(parts[0])
    seconds = int(parts[1])
    milliseconds = int(parts[2])

    total_milliseconds = (minutes * 60 * 1000) + (seconds * 1000) + milliseconds
    return total_milliseconds

def fix_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Convert in milliseconds
    df["current_time"] = df["current_time_str"].apply(convert_to_milliseconds)
    
    # Delete negative speed values
    df = df[df["speed"] >= 0]

    cols_to_drop = [
    'wheel_slip_front_left', 'wheel_slip_front_right',
    'wheel_slip_rear_left', 'wheel_slip_rear_right',
    'current_time_str'
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    return df


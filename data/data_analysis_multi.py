import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def parse_filename(filename):
    # Exemple: vehicle_telemetry_mattia_abu_0G_S.csv
    base = os.path.basename(filename)
    parts = base.replace('.csv', '').split('_')
    driver = parts[2]
    track = parts[3]
    temp = parts[4]
    return driver, track, temp

def load_telemetry_data(folder_pattern="*.csv"):
    all_files = glob.glob(folder_pattern)
    rows = []

    for f in all_files:
        driver, track, temp = parse_filename(f)
        df = pd.read_csv(f)
        df['driver'] = driver
        df['track'] = track
        df['temp'] = temp
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def analyze_data(df):
    # Mean slip for driver and temperature
    slip_by_temp_track = df.groupby(['track', 'temp'])[['wheel_slip_front_left', 'wheel_slip_front_right',
                                                        'wheel_slip_rear_left', 'wheel_slip_rear_right']].mean()
    slip_by_temp_track = slip_by_temp_track.mean(axis=1).reset_index(name='mean_slip_0')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=slip_by_temp_track, x='track', y='mean_slip_0', hue='temp', palette="viridis")
    plt.title("Average Slip by Circuit and Temperature (All Drivers)")
    plt.xlabel("Circuit")
    plt.ylabel("Average Slip")
    plt.legend(title="Temperature")
    plt.show()

    # Mean grip for driver and temperature
    slip_by_temp = df.groupby(['temp'])[['wheel_slip_front_left', 'wheel_slip_front_right',
                                        'wheel_slip_rear_left', 'wheel_slip_rear_right']].mean()
    slip_by_temp = slip_by_temp.mean(axis=1).reset_index(name='mean_slip')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=slip_by_temp, x='temp', y='mean_slip', hue='temp', palette="coolwarm", dodge=False, legend=False)
    plt.title("Average Slip by Temperature (All Circuits and Drivers)")
    plt.xlabel("Temperature")
    plt.ylabel("Average Slip")
    plt.show()
    
    # Result distribution by driver and temperature
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='result', hue='temp')
    plt.title("Distribution of labels (result) by temperature")
    plt.show()

    # Mena slip vs temperature cs track
    grouped_slip = df.groupby(['track', 'temp', 'driver'])[['wheel_slip_front_left', 'wheel_slip_front_right',
                                                            'wheel_slip_rear_left', 'wheel_slip_rear_right']].mean()
    grouped_slip = grouped_slip.mean(axis=1).reset_index(name='mean_slip')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped_slip, x='track', y='mean_slip', hue='temp')
    plt.title("Average Slip by Track and Temperature")
    plt.show()

    # Corner mean speed by driver and track
    df['is_curve'] = df['result'] != "neutral"  # Assuming 'neutral' means straight
    curve_speed = df[df['is_curve']].groupby(['track', 'driver'])['speed'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=curve_speed, x='track', y='speed', hue='driver')
    plt.title("Average speed in corners by track and driver")
    plt.show()

    # Mean grip behavior by driver and track
    grip_behavior = df.groupby(['track', 'driver'])[['wheel_slip_front_left', 'wheel_slip_front_right',
                                                     'wheel_slip_rear_left', 'wheel_slip_rear_right']].mean()
    grip_behavior = grip_behavior.mean(axis=1).reset_index(name='mean_grip')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grip_behavior, x='track', y='mean_grip', hue='driver')
    plt.title("Average grip behavior by track and driver")
    plt.show()

    # Variation in G-force by driver and track
    g_force_diff = df.groupby(['track', 'driver'])['g_force_y'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=g_force_diff, x='track', y='g_force_y', hue='driver')
    plt.title("Difference in lateral G-forces by track and driver")
    plt.show()

    # Boxplots for speed and G-force by driver and track
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='track', y='speed', hue='driver')
    plt.title("Distribution of speed by track and driver")
    plt.xlabel("Track")
    plt.ylabel("Speed (km/h)")
    plt.legend(title="Driver")
    plt.show()

    # Boxplot for G-force
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='track', y='g_force_y', hue='driver')
    plt.title("Distribution of lateral G-forces by track and driver")
    plt.xlabel("Track")
    plt.ylabel("Lateral G-forces")
    plt.legend(title="Driver")
    plt.show()

    # Boxplot for average slip by temperature and driver
    # Limiting the maximum value of slip to 3 for better visualization
    plt.figure(figsize=(12, 6))
    df["mean_slip_1"] = df[['wheel_slip_front_left', 'wheel_slip_front_right', 'wheel_slip_rear_left', 'wheel_slip_rear_right']].mean(axis=1)
    df["mean_slip_1"] = np.where(df["mean_slip_1"] > 3, 3, df["mean_slip_1"])
    sns.boxplot(data=df, x='temp', y='mean_slip_1', hue='driver')
    plt.title("Distribution of average slip by temperature and driver")
    plt.xlabel("Temperature")
    plt.ylabel("Average slip")
    plt.legend(title="Driver")
    plt.show()

    # Campionamento casuale del 10% per ogni combinazione di driver, track e temp
    sampled_df = df.groupby(['driver', 'track', 'temp'], group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))

    # Seleziona solo le colonne numeriche
    numeric_cols = ['speed', 'g_force_x', 'g_force_y', 'g_force_z', 
                    'wheel_slip_front_left', 'wheel_slip_front_right', 
                    'wheel_slip_rear_left', 'wheel_slip_rear_right', 
                    'yaw_rate']

    # Data Normalization
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(sampled_df[numeric_cols])
    
    # PCA Reduction
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(normalized_data)
    sampled_df['PCA_X'] = pca_result[:, 0]
    sampled_df['PCA_Y'] = pca_result[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sampled_df, x='PCA_X', y='PCA_Y', hue='driver', style='track', alpha=0.7)
    plt.title("PCA of telemetry data (reduced to 2 components)")
    plt.xlabel("PCA X")
    plt.ylabel("PCA Y")
    plt.legend(title="Driver")
    plt.show()

def main():
    df = load_telemetry_data("dataset/vehicle_telemetry_*.csv")
    if df.empty:
        print("Nessun file trovato. Controlla il pattern o la cartella.")
        return
    analyze_data(df)

if __name__ == "__main__":
    main()
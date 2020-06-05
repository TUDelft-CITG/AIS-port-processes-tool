import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.ticker as mtick
pd.options.mode.chained_assignment = None


# Plot service times
def plot_service_times(data, title_name):
    # Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(data['service_time[hr]'], bins=50)
    plt.title(title_name)
    plt.xlabel('Time [hr]')
    plt.ylabel('Number of vessel tracks')
    plt.show()


# Plot inter arrival times, based on port entry sequence
def plot_inter_arrival_times(data, title_name):
    # Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(data['inter_arrival_time_port[hr]'], bins=50)
    plt.title(title_name)
    plt.xlabel('Time [hr]')
    plt.ylabel('Number of vessel tracks')
    plt.show()


# Plot waiting times
def plot_waiting_times(data, title_name, title_name_adjusted):
    # Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(data['waiting_time[hr]'], bins=50)
    plt.title(title_name)
    plt.xlabel('Time [hr]')
    plt.ylabel('Number of vessel tracks')
    plt.show()

    # Adjusted waiting time (only show waiting time > 10 min)
    adjusted_waiting_time = data.loc[data['waiting_time[hr]'] > (10 / 60)]  # 10 min
    plt.figure(figsize=(12, 6))
    plt.hist(adjusted_waiting_time['waiting_time[hr]'], bins=50)
    plt.title(title_name_adjusted)
    plt.xlabel('Time [hr]')
    plt.ylabel('Number of vessel tracks')
    plt.show()


# Test handle
if __name__ == '__main__':
    # Load data frame with attached columns, sorted by port entry time
    location = 'ct_BEST'
    df = pd.read_csv('Data-frames/New_df_p_' + location + '.csv')

    # Plot service times
    plot_service_times(df, location + ': Service times')

    # Plot inter arrival times (based on port entry)
    plot_inter_arrival_times(df, location + ': Inter arrival times based on port entry')

    # Plot waiting times
    plot_waiting_times(df, location + ': Waiting times', location + ': Adjusted waiting times')


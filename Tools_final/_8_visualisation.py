import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.ticker as mtick
import seaborn as sns
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
    adjusted_waiting_time = data.loc[data['waiting_time[hr]'] > (100 / 60)]  # 10 min
    plt.figure(figsize=(12, 6))
    plt.hist(adjusted_waiting_time['waiting_time[hr]'], bins=50)
    plt.title(title_name_adjusted)
    plt.xlabel('Time [hr]')
    plt.ylabel('Number of vessel tracks')
    plt.show()


# Plot boxplot of WT/ST ratio
def plot_wt_st_ratio(data, title_name):
    plt.figure()
    sns.boxplot(data['waiting/service_time[%]'], orient="v", )
    plt.title(title_name)
    plt.show()


# Plot waiting times across service times
def plot_wt_st(data, title_name, percentage1, percentage2, percentage3):
    plt.figure()
    plt.plot(data['service_time[hr]'], data['waiting_time[hr]'], 'o')
    plt.title(title_name)
    plt.ylabel('Waiting time [hr]')
    plt.xlabel('Service time [hr]')

    # Add line
    ST = np.linspace(0, df['service_time[hr]'].max(), 100)
    WT1 = ST * (percentage1/100)
    plt.plot(ST, WT1, label=percentage1)
    WT2 = ST * (percentage2/100)
    plt.plot(ST, WT2, label=percentage2)
    WT3 = ST * (percentage3/100)
    plt.plot(ST, WT3, label=percentage3)
    plt.legend(title='%')
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

    # Plot WT/ST ratio
    plot_wt_st_ratio(df, location + ': WT/ST ratio')

    # Plot waiting times across service times
    plot_wt_st(df, location + ': Service times vs waiting times', 10, 20, 30)

    # Return averages
    print('Average service time: ', np.round(df['inter_arrival_time_port[hr]'].mean(), 2), 'hr')
    print('Average waiting time: ', np.round(df['waiting_time[hr]'].mean(), 2), 'hr')
    print('Average WT/ST ratio: ', np.round(df['waiting/service_time[%]'].mean(), 2), '%')


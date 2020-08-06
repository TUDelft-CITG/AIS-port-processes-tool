""" Step 8. Visualisations of study parameters
 Input: Data frame with study parameters and occupancy's
 Actions: Visualise study parameters
 Output: Multiple visualisations of study parameters
 """

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
def plot_waiting_times(data, title_name, title_name_adjusted, adjust_WT):
    # Histogram
    plt.figure(figsize=(12, 6))
    plt.hist(data['waiting_time[hr]'], bins=50)
    plt.title(title_name)
    plt.xlabel('Time [hr]')
    plt.ylabel('Number of vessel tracks')
    plt.show()
    #
    # # Adjusted waiting time (only show waiting time > limit:adjust_WT)
    # adjusted_waiting_time = data.loc[data['waiting_time[hr]'] > adjust_WT]
    # plt.figure(figsize=(12, 6))
    # plt.hist(adjusted_waiting_time['waiting_time[hr]'], bins=50)
    # plt.title(title_name_adjusted)
    # plt.xlabel('Time [hr]')
    # plt.ylabel('Number of vessel tracks')
    # plt.show()

#
# # Split service time into different groups
# def serv_time_group(df):
#     df['service_time_group'] = '0'
#     for row in df.itertuples():
#         if df.at[row.Index, 'service_time[hr]'] < 15:
#             df.at[row.Index, 'service_time_group'] = '0-15'
#         elif (df.at[row.Index, 'service_time[hr]'] >= 15) and (df.at[row.Index, 'service_time[hr]'] < 30):
#             df.at[row.Index,'service_time_group'] = '15-30'
#         elif (df.at[row.Index, 'service_time[hr]'] >= 30) and (df.at[row.Index, 'service_time[hr]'] < 45):
#             df.at[row.Index,'service_time_group'] = '30-45'
#         elif (df.at[row.Index, 'service_time[hr]'] >= 45) and (df.at[row.Index, 'service_time[hr]'] < 60):
#             df.at[row.Index, 'service_time_group'] = '45-60'
#         elif (df.at[row.Index, 'service_time[hr]'] >= 60) and (df.at[row.Index, 'service_time[hr]'] < 75):
#             df.at[row.Index, 'service_time_group'] = '60-75'
#         elif df.at[row.Index, 'service_time[hr]'] >= 75:
#             df.at[row.Index, 'service_time_group'] = '75-..'
#     return df['service_time_group']
#
#
# # Plot boxplot of WT/ST ratio
# def plot_wt_st_ratio(data, title_name, remove):
#     plt.figure()
#     # Single boxplot
#     # sns.boxplot(data['waiting/service_time[%]'], orient="v", )
#     data['service_time_group'] = serv_time_group(data)
#     sns.boxplot(x="service_time_group", y='waiting_time[hr]', data=data)
#     plt.xlabel('Service time [hr]')
#     plt.ylabel('Waiting time [hr]')
#     # To remove outliers
#     if remove > 0:
#         plt.ylim(0, 100)
#     plt.title(title_name)
#     plt.show()

#
# # Plot waiting times across service times
# def plot_wt_st(data, title_name, percentage1, percentage2, percentage3):
#     plt.figure()
#     plt.plot(data['service_time[hr]'], data['waiting_time[hr]'], 'o')
#     plt.title(title_name)
#     plt.ylabel('Waiting time [hr]')
#     plt.xlabel('Service time [hr]')
#
#     # Add line
#     ST = np.linspace(0, data['service_time[hr]'].max(), 100)
#     WT1 = ST * (percentage1/100)
#     plt.plot(ST, WT1, label=percentage1)
#     WT2 = ST * (percentage2/100)
#     plt.plot(ST, WT2, label=percentage2)
#     WT3 = ST * (percentage3/100)
#     plt.plot(ST, WT3, label=percentage3)
#     plt.legend(title='%')
#     plt.show()
#
#
# def jointplot_wt_st(df):
#     x = df['service_time[hr]']
#     y = df['waiting_time[hr]']
#     hexplot = sns.jointplot(x, y, kind="hex")
#     plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # shrink fig so cbar is visible
#     # Make new ax object for the cbar
#     cbar_ax = hexplot.fig.add_axes([.85, .25, .05, .4])  # x, y, width, height
#     plt.colorbar(cax=cbar_ax, label='Number of vessel tracks')
#     plt.show()


# Test handle
if __name__ == '__main__':
    # Load data frame with attached columns, sorted by port entry time
    location = 'ct_rdam_euromax'
    df = pd.read_csv('Data-frames/Results_phase_2/' + location + '/Df_stats_' + location + '.csv')

    # # Plot service times
    # plot_service_times(df, location + ': Service times')
    #
    # # Plot inter arrival times (based on port entry)
    # plot_inter_arrival_times(df, location + ': Inter arrival times based on port entry')
    #
    # # Plot waiting times
    # adjust_WT = 1 # number of first x hours to delete from visualisation of waiting times
    # plot_waiting_times(df, location + ': Waiting times', location + ': Adjusted waiting times', adjust_WT)
    #
    # Plot WT/ST ratio
    plot_wt_st_ratio(df, location + ': WT/ST ratio', 1)

    # # Plot waiting times across service times
    # plot_wt_st(df, location + ': Service times vs waiting times', 10, 20, 30)

    # # Plot joint plot
    # jointplot_wt_st(df, 1)

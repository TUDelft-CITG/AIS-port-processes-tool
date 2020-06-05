import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.ticker as mtick
pd.options.mode.chained_assignment = None


# Count number of vessels present with input = certain timestamp
def vessels_present_pertime(data, t):
    data['terminal_entry_time'] = pd.to_datetime(data.terminal_entry_time, format='%Y-%m-%d %H:%M')
    data['terminal_exit_time'] = pd.to_datetime(data.terminal_exit_time, format='%Y-%m-%d %H:%M')
    list_vessels_present = 0
    for row in data.itertuples():
        if row.terminal_entry_time < t < row.terminal_exit_time:
            list_vessels_present += 1
    return (t, list_vessels_present)


# Determine berth occupancy per terminal
def berth_occupancy(data, number_of_berths, operating_hours, visualise):
    # Split timestamps
    # Define starting and end point
    t_start = data.terminal_entry_time.min()
    t_end = data.terminal_exit_time.max()
    # Make a list of all timestamps between begin and end (per hour)
    time = pd.date_range(t_start, t_end, freq='H')
    timestamps = [str(x) for x in time]
    timestamps = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M')

    # Make new data frame with timestamps and number of vessels present at a certain time
    list_vessels = []
    for i in timestamps:
        list_vessels.append(vessels_present_pertime(data, i)[1])
    data_time = pd.DataFrame({"timestamp": timestamps, 'vessels': 0})
    x = np.linspace(0, len(timestamps) - 1, len(timestamps))
    for r in x:
        data_time['vessels'][int(r)] = list_vessels[int(r)]

    # Add column with percentage occupied, compared to total number of berths
    data_time['occupancy[%]'] = 0
    for row in data_time.itertuples():
        data_time.at[row.Index, 'occupancy[%]'] = (int(row.vessels) * 100 / number_of_berths)

    df_occupancy = data_time

    # Average berth occupancy for entire terminal
    average_occupancy = df_occupancy['occupancy[%]'].mean()
    # Average berth occupancy compared to operating hours
    average_occupancy_op_hours = average_occupancy * (365 * 24.) / operating_hours

    print('The average occupancy of the terminal is', np.round(average_occupancy, 2), '%, relative to the total '
                                                                                         'operating time the average'
                                                                                         ' occupancy is',
          np.round(average_occupancy_op_hours, 2), '%')

    if visualise > 0:
        plt.figure()
        plt.xlabel('Time [per hour]')
        plt.ylabel('Occupancy at terminal [%]')
        plt.title('Terminal occupancy over time')
        plt.plot(df_occupancy.timestamp, df_occupancy['occupancy[%]'])

        x = np.linspace(1, number_of_berths, number_of_berths)
        for i in x:
            berth = [(100/number_of_berths)*i, (100/number_of_berths)*i]
            plt.plot([df_occupancy.timestamp.min(), df_occupancy.timestamp.max()], berth, linestyle='--', label=i)
        plt.legend(title='Berth')
        plt.show()

    return df_occupancy


# Count length of vessels present with input = certain timestamp
def length_present_pertime(data, t):
    data['terminal_entry_time'] = pd.to_datetime(data.terminal_entry_time, format='%Y-%m-%d %H:%M')
    data['terminal_exit_time'] = pd.to_datetime(data.terminal_exit_time, format='%Y-%m-%d %H:%M')
    list_length_present = 0
    for row in data.itertuples():
        if row.terminal_entry_time < t < row.terminal_exit_time:
            list_length_present += int(row.loa)
    return (t, list_length_present)


# Determine length occupancy per terminal
def length_occupancy(data, total_length, operating_hours, visualise):
    # Split timestamps
    # Define starting and end point
    t_start = data.terminal_entry_time.min()
    t_end = data.terminal_exit_time.max()
    # Make a list of all timestamps between begin and end (per hour)
    time = pd.date_range(t_start, t_end, freq='H')
    timestamps = [str(x) for x in time]
    timestamps = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M')

    # Make new data frame with timestamps and number of vessels present at a certain time
    list_lengths = []
    for i in timestamps:
        list_lengths.append(length_present_pertime(data, i)[1])
    data_time = pd.DataFrame({"timestamp": timestamps, 'length_present[m]': 0})
    x = np.linspace(0, len(timestamps) - 1, len(timestamps))
    for r in x:
        data_time['length_present[m]'][int(r)] = list_lengths[int(r)]

    # Add column with percentage length occupied, compared to total length available
    data_time['occupancy_length[%]'] = 0
    for row in data_time.itertuples():
        data_time.at[row.Index, 'occupancy_length[%]'] = ((int(data_time.at[row.Index, 'length_present[m]'])) * 100
                                                          / total_length)

    df_occupancy = data_time

    # Average berth occupancy for entire terminal
    average_occupancy = df_occupancy['occupancy_length[%]'].mean()
    # Average berth occupancy compared to operating hours
    average_occupancy_op_hours = average_occupancy * (365 * 24.) / operating_hours

    print('The average length occupancy of the terminal is', np.round(average_occupancy, 2), '%, relative to the total '
                                                                                         'operating time the average'
                                                                                         ' length occupancy is',
          np.round(average_occupancy_op_hours, 2), '%')

    if visualise > 0:
        plt.figure()
        plt.xlabel('Time [per hour]')
        plt.ylabel('Length occupancy [%]')
        plt.title('Length occupancy over time')
        plt.plot(df_occupancy.timestamp, df_occupancy['occupancy_length[%]'])
        plt.plot([df_occupancy.timestamp.min(), df_occupancy.timestamp.max()], [100, 100], linestyle='--',
                 label=total_length)
        plt.legend(title='Length available [m]')
        plt.show()

    return data_time


# Run all occupancy steps:
def run_all_occupancy(df, number_of_berths, operating_hours, visualise_berth_oc, length_term, visualise_length_oc):
    # If number of berths known: berth occupancy per terminal
    if number_of_berths > 0:
        df_berth_occupancy = berth_occupancy(df, number_of_berths, operating_hours, visualise_berth_oc)
    else:
        df_berth_occupancy = 0
        print('The input for number of berths was not specified, thus the berth occupancy can not be defined')

    # If length of terminal can be fully occupied, determine length occupancy:
    if length_term > 0:
        df_length_occupancy = length_occupancy(df, length_term, operating_hours, visualise_length_oc)
    else:
        df_length_occupancy = 0
        print('The length of terminal was not specified, thus the legnth occupancy can not be defined')

    # Determine factor: waiting time in terms of service time
    df['waiting/service_time[%]'] = df['waiting_time[hr]'] * 100 / df['service_time[hr]']
    print('The average of all factors (waiting times/service times) is',
          np.round(df['waiting/service_time[%]'].mean(), 2),
          '%, and the average waiting time as a fraction of the average service time is',
          np.round(df['waiting_time[hr]'].mean() * 100 / df['service_time[hr]'].mean(), 2), '%')

    return df_berth_occupancy, df_length_occupancy, df


# Test handle
if __name__ == '__main__':
    # Load data frame with attached columns, sorted by port entry time
    location = 'ct_BEST'
    df = pd.read_csv('Data-frames/New_df_p_' + location + '.csv')

    """ ....... INPUTS ......... """
    # Number of berths: (1,2,3... number, or if unknown: 0)
    number_of_berths = 0  # Input
    # Operating hours per year:
    operating_hours = 355 * 24  # Input
    # Visualise berth occupancy over time (1 = yes)
    visualise_berth_oc = 1  # Input
    # Total length terminal [m] (if unknown: 0)
    length_term = 1490  # Input
    # Visualise length occupancy over time (1 = yes)
    visualise_length_oc = 1  # Input

    # Run all occupancy steps
    df_berth_occupancy, df_length_occupancy, df_factor_occupancy = run_all_occupancy(df, number_of_berths,
                                                                                     operating_hours,
                                                                                     visualise_berth_oc, length_term,
                                                                                     visualise_length_oc)

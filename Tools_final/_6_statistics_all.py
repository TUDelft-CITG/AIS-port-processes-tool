""" Step 6. Run all statistical steps on the processed AIS data
 Input: Processed data [csv file] (entry and exit timestamps for the port area, anchorage area, terminal area)
 Actions: Attach service times, sort by port entry, inter arrival times, sort by port entry (relative to  first moment
 in time), visualise study parameters, split into different vessel classes and fit multiple distributions
Output: Data frame with study parameters, and multiple visualisations (fitted distributions)
 """

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from _1_data_gathering import drop_and_report
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.ticker as mtick
pd.options.mode.chained_assignment = None
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt
from fitter import Fitter
from decimal import Decimal


# Remove all vessel tracks with time in polygon < 30 min:
def remove_little_messages_tot_second(data):
    drop_list = list()
    for row in data.itertuples():
        if data.at[row.Index, 'service_time[hr]'] < 0.5:
            drop_list.append(row.Index)
            continue
    data = drop_and_report(data, drop_list, 'Remove obvious non-berthing tracks (from new data frame)')
    return data


# Add service and waiting times
def service_times(df):
    df['port_entry_time'] = pd.to_datetime(df.port_entry_time, format='%Y-%m-%d %H:%M')
    df['port_exit_time'] = pd.to_datetime(df.port_exit_time, format='%Y-%m-%d %H:%M')
    df['terminal_entry_time'] = pd.to_datetime(df.terminal_entry_time, format='%Y-%m-%d %H:%M')
    df['terminal_exit_time'] = pd.to_datetime(df.terminal_exit_time, format='%Y-%m-%d %H:%M')
    df['anchorage_entry_time'] = pd.to_datetime(df.anchorage_entry_time, format='%Y-%m-%d %H:%M')
    df['anchorage_exit_time'] = pd.to_datetime(df.anchorage_exit_time, format='%Y-%m-%d %H:%M')

    # Add service times
    df['service_time[hr]'] = (df.terminal_exit_time - df.terminal_entry_time).astype('timedelta64[s]') / 3600.

    # Remove obvious non - berthing tracks that re-entered in new data frame (service time < 30 min)
    df = remove_little_messages_tot_second(df)

    # Replace all NaN by zero
    df = df.fillna(0)

    return df


# Add inter arrival time, based on sorting by time entering port
def sort_by_port_entry(df):
    """ Based on port time arrivals"""
    df_p = df.copy()
    df_p = df_p.drop(columns=['Unnamed: 0', 'term_track_number', 'port_track_number'])
    df_p = df_p.sort_values(by=['port_entry_time'])
    df_p = df_p.reset_index(drop=True)

    # Add inter arrival time (based on port entries)
    df_p['inter_arrival_time_port[hr]'] = 0.0
    for row in df_p.itertuples():
        if row.Index != 0:
            df_p.at[row.Index, 'inter_arrival_time_port[hr]'] = (row.port_entry_time -
                                                                 df_p.at[row.Index - 1, 'port_entry_time']
                                                                 ).total_seconds() / 3600.
    return df_p


# Add inter arrival time, based on sorting by time entering port, relative to t0
def sort_by_port_entry_rel(df):
    """ Based on port time arrivals"""
    df_p = df.copy()
    # Normalise timestamps (first entry port = 0)
    t0 = df_p.port_entry_time.min()

    df_p.port_entry_time = (df_p.port_entry_time - t0).astype('timedelta64[s]') / 3600.
    df_p.port_exit_time = (df_p.port_exit_time - t0).astype('timedelta64[s]') / 3600.
    df_p.terminal_entry_time = (df_p.terminal_entry_time - t0).astype('timedelta64[s]') / 3600.
    df_p.terminal_exit_time = (df_p.terminal_exit_time - t0).astype('timedelta64[s]') / 3600.
    df_p.anchorage_entry_time = (df_p.anchorage_entry_time - t0).astype('timedelta64[s]') / 3600.
    df_p.anchorage_exit_time = (df_p.anchorage_exit_time - t0).astype('timedelta64[s]') / 3600.

    for row in df_p.itertuples():
        if row.anchorage_exit_time < 0:
            df_p.at[row.Index, 'anchorage_entry_time'] = 0
            df_p.at[row.Index, 'anchorage_exit_time'] = 0

    return df_p


# Add vessel class for container terminals
def add_vessel_class_ct(df):
    df['vessel_class'] = 'undefined'
    for row in df.itertuples():
        if row.type == 'Container' or row.type == 'Inland Waterways Dry Cargo / Passenger':
            if row.loa < 145:  # Small feeder
                df.at[row.Index, 'vessel_class'] = 'c_1'
            elif (row.loa >= 145) and (row.loa < 185):  # Regional feeder
                df.at[row.Index, 'vessel_class'] = 'c_2'
            elif (row.loa >= 185) and (row.loa < 223):  # Feeder max + Panamax
                df.at[row.Index, 'vessel_class'] = 'c_3'
            elif (row.loa >= 223) and (row.loa < 366):  # New Panamax
                df.at[row.Index, 'vessel_class'] = 'c_4'
            elif row.loa >= 366:  # Post New Panamax
                df.at[row.Index, 'vessel_class'] = 'c_5'
        elif row.type == 'General Cargo':
            if row.DWT < 5000:
                df.at[row.Index, 'vessel_class'] = 'gc_1'
            elif (row.DWT >= 5000) and (row.DWT < 10000):
                df.at[row.Index, 'vessel_class'] = 'gc_2'
            elif (row.DWT >= 10000) and (row.DWT < 15000):
                df.at[row.Index, 'vessel_class'] = 'gc_3'
            elif (row.DWT >= 15000) and (row.DWT < 20000):
                df.at[row.Index, 'vessel_class'] = 'gc_4'
            elif (row.DWT >= 20000) and (row.DWT < 30000):
                df.at[row.Index, 'vessel_class'] = 'gc_5'
            elif row.DWT >= 30000:
                df.at[row.Index, 'vessel_class'] = 'gc_6'
    return df['vessel_class']


# Add vessel class for dry bulk terminals
def add_vessel_class_db(df):
    df['vessel_class'] = 'undefined'
    for row in df.itertuples():
        if row.type == 'General Cargo':
            if row.DWT < 5000:
                df.at[row.Index, 'vessel_class'] = 'gc_1'
            elif (row.DWT >= 5000) and (row.DWT < 10000):
                df.at[row.Index, 'vessel_class'] = 'gc_2'
            elif (row.DWT >= 10000) and (row.DWT < 15000):
                df.at[row.Index, 'vessel_class'] = 'gc_3'
            elif (row.DWT >= 15000) and (row.DWT < 20000):
                df.at[row.Index, 'vessel_class'] = 'gc_4'
            elif (row.DWT >= 20000) and (row.DWT < 30000):
                df.at[row.Index, 'vessel_class'] = 'gc_5'
            elif row.DWT >= 30000:
                df.at[row.Index, 'vessel_class'] = 'gc_6'
        elif row.type == 'Bulk Dry' or row.type == 'Other Bulk Dry' or \
                row.type == 'Inland Waterways Dry Cargo / Passenger':
            if row.DWT < 10000:  # Small handy
                df.at[row.Index, 'vessel_class'] = 'db_1'
            elif (row.DWT >= 10000) and (row.DWT < 65000):  # Handy + Handymax + Supramax
                df.at[row.Index, 'vessel_class'] = 'db_2'
            elif (row.DWT >= 65000) and (row.DWT < 85000):  # Panamax
                df.at[row.Index, 'vessel_class'] = 'db_3'
            elif (row.DWT >= 85000) and (row.DWT < 200000):  # (Mini) Capesize
                df.at[row.Index, 'vessel_class'] = 'db_4'
            elif row.DWT >= 200000:  # VLBC or VLOC
                df.at[row.Index, 'vessel_class'] = 'db_5'
    return df['vessel_class']


# Add vessel class for liquid bulk terminals
def add_vessel_class_lb(df):
    df['vessel_class'] = 'undefined'
    for row in df.itertuples():
        if row.type == 'Gas tankers' or row.type == 'Oil' or row.type == 'Chemical' or \
                row.type == 'Inland Waterways Tanker':
            if row.loa < 250:
                df.at[row.Index, 'vessel_class'] = 'lng_1'
            elif (row.loa >= 250) and (row.loa < 275):
                df.at[row.Index, 'vessel_class'] = 'lng_2'
            elif (row.loa >= 275) and (row.loa < 300):
                df.at[row.Index, 'vessel_class'] = 'lng_3'
            elif row.loa >= 300:
                df.at[row.Index, 'vessel_class'] = 'lng_4'
    return df['vessel_class']


# Add vessel class dependent on terminal type
def add_vessel_class(df, terminal_type):
    df['vessel_class'] = 'undefined'
    if terminal_type == 'container':
        df['vessel_class'] = add_vessel_class_ct(df)
    elif terminal_type == 'dry_bulk':
        df['vessel_class'] = add_vessel_class_db(df)
    elif terminal_type == 'liquid_bulk':
        df['vessel_class'] = add_vessel_class_lb(df)
    else:
        print('No correct terminal type given')
    return df['vessel_class']


# Plot vessel arrival/ vessel class (4 ways)
def vessel_arrivals_per_class_type(df):
    # Plot vessel arrivals per vessel class per type
    fig, ax = plt.subplots()
    df.groupby(['vessel_class', 'type']).count()['terminal_entry_time'].unstack().plot.bar(ax=ax, stacked=True)
    plt.xticks(rotation=0)
    plt.legend(title='Vessel type')
    plt.xlabel('Vessel class')
    plt.ylabel('Number of arrivals per vessel class')
    plt.title('Vessel type per vessel class')


# Keep only certain vessel class
def keep_certain_vessel_class(df, vessel_class):
    data = df.copy()
    drop_list = list()
    for row in data.itertuples():
        if row.vessel_class != vessel_class:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, "Keep only certain vessel class")
    return data


# Split data into vessel class data frames
def split_data_frame_class(df, terminal_type):
    if terminal_type == 'container':
        df_1 = keep_certain_vessel_class(df, 'c_1')
        df_2 = keep_certain_vessel_class(df, 'c_2')
        df_3 = keep_certain_vessel_class(df, 'c_3')
        df_4 = keep_certain_vessel_class(df, 'c_4')
        df_5 = keep_certain_vessel_class(df, 'c_5')
        return df_1, df_2, df_3, df_4, df_5
    elif terminal_type == 'dry_bulk':
        df_1 = keep_certain_vessel_class(df, 'db_1')
        df_2 = keep_certain_vessel_class(df, 'db_2')
        df_3 = keep_certain_vessel_class(df, 'db_3')
        df_4 = keep_certain_vessel_class(df, 'db_4')
        df_5 = keep_certain_vessel_class(df, 'db_5')
        return df_1, df_2, df_3, df_4, df_5
    elif terminal_type == 'liquid_bulk':
        df_1 = keep_certain_vessel_class(df, 'lng_1')
        df_2 = keep_certain_vessel_class(df, 'lng_2')
        df_3 = keep_certain_vessel_class(df, 'lng_3')
        df_4 = keep_certain_vessel_class(df, 'lng_4')
        return df_1, df_2, df_3, df_4
    else:
        print('No correct terminal type given')


# Print all descriptive results for every different data set (per classification)
def results_all(df_tot, terminal_type):
    if terminal_type == 'container':
        class_list = [df_tot, df_1, df_2, df_3, df_4, df_5]
        list_class = ['total', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5']
    elif terminal_type == 'dry_bulk':
        class_list = [df_tot, df_1, df_2, df_3, df_4, df_5]
        list_class = ['total', 'db_1', 'db_2', 'db_3', 'db_4', 'db_5']
    elif terminal_type == 'liquid_bulk':
        class_list = [df_tot, df_1, df_2, df_3, df_4]
        list_class = ['total', 'lng_1', 'lng_2', 'lng_3', 'lng_4']
    else:
        print('No correct terminal type was given')

    # Create data frame
    list_count = []
    list_mean = []
    list_std = []
    list_min = []
    list_25 = []
    list_50 = []
    list_75 = []
    list_max = []

    for i in class_list:
        list_count.append(i['inter_arrival_time_port[hr]'].describe()['count'])
        list_mean.append(i['inter_arrival_time_port[hr]'].describe()['mean'])
        list_std.append(i['inter_arrival_time_port[hr]'].describe()['std'])
        list_min.append(i['inter_arrival_time_port[hr]'].describe()['min'])
        list_25.append(i['inter_arrival_time_port[hr]'].describe()['25%'])
        list_50.append(i['inter_arrival_time_port[hr]'].describe()['50%'])
        list_75.append(i['inter_arrival_time_port[hr]'].describe()['75%'])
        list_max.append(i['inter_arrival_time_port[hr]'].describe()['max'])

    df_describe_iat = pd.DataFrame()
    df_describe_iat['vessel_class'] = list_class
    df_describe_iat['count'] = list_count
    df_describe_iat['mean'] = list_mean
    df_describe_iat['std'] = list_std
    df_describe_iat['min'] = list_min
    df_describe_iat['25%'] = list_25
    df_describe_iat['50%'] = list_50
    df_describe_iat['75%'] = list_75
    df_describe_iat['max'] = list_max

    # Create data frame
    list_count = []
    list_mean = []
    list_std = []
    list_min = []
    list_25 = []
    list_50 = []
    list_75 = []
    list_max = []

    for i in class_list:
        list_count.append(i['service_time[hr]'].describe()['count'])
        list_mean.append(i['service_time[hr]'].describe()['mean'])
        list_std.append(i['service_time[hr]'].describe()['std'])
        list_min.append(i['service_time[hr]'].describe()['min'])
        list_25.append(i['service_time[hr]'].describe()['25%'])
        list_50.append(i['service_time[hr]'].describe()['50%'])
        list_75.append(i['service_time[hr]'].describe()['75%'])
        list_max.append(i['service_time[hr]'].describe()['max'])

    df_describe_st = pd.DataFrame()
    df_describe_st['vessel_class'] = list_class
    df_describe_st['count'] = list_count
    df_describe_st['mean'] = list_mean
    df_describe_st['std'] = list_std
    df_describe_st['min'] = list_min
    df_describe_st['25%'] = list_25
    df_describe_st['50%'] = list_50
    df_describe_st['75%'] = list_75
    df_describe_st['max'] = list_max

    return df_describe_iat, df_describe_st


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
def berth_occupancy(data, number_of_berths, operating_hours):
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


# Count length of vessels present with input = certain timestamp, for adjusted lengths
def length_present_pertime_adjusted(data, t):
    data['terminal_entry_time'] = pd.to_datetime(data.terminal_entry_time, format='%Y-%m-%d %H:%M')
    data['terminal_exit_time'] = pd.to_datetime(data.terminal_exit_time, format='%Y-%m-%d %H:%M')
    list_length_present = 0
    for row in data.itertuples():
        if row.terminal_entry_time < t < row.terminal_exit_time:
            list_length_present += (int(row.loa) + 15)
    return (t, list_length_present)


# Determine length occupancy per terminal
def length_occupancy(data, total_length, operating_hours):
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
    data_time = pd.DataFrame({"timestamp": timestamps, 'length_present[m]': 0, 'length_present_adjust[m]': 0})
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

    # Adjusted length occupancy (to adjust for length between ships)
    # Make new data frame with timestamps and number of vessels present at a certain time
    list_lengths_adjust = []
    for i in timestamps:
        list_lengths_adjust.append((length_present_pertime_adjusted(data, i)[1])+15)
    x = np.linspace(0, len(timestamps) - 1, len(timestamps))
    for r in x:
        df_occupancy['length_present_adjust[m]'][int(r)] = list_lengths_adjust[int(r)]

    # Add column with percentage length occupied, compared to total length available (for adjusted lengths)
    df_occupancy['occupancy_length_adjust[%]'] = 0
    for row in data_time.itertuples():
        df_occupancy.at[row.Index, 'occupancy_length_adjust[%]'] = ((int(df_occupancy.at[row.Index,
                                                                                         'length_present_adjust[m]']))
                                                                    * 100 / total_length)

    # Adjusted average length occupancy
    adjust_avg_length_occup = (df_occupancy['occupancy_length_adjust[%]'].mean())
    adjust_avg_length_occup_op_hours = adjust_avg_length_occup * (365 * 24.) / operating_hours
    print('The adjusted average length occupancy of the terminal is', np.round(adjust_avg_length_occup, 2),
          '%, relative to the total operating time the average length occupancy is',
          np.round(adjust_avg_length_occup_op_hours, 2), '%')

    return df_occupancy


# Run all occupancy steps
def run_all_occupancy(df, number_of_berths, operating_hours, length_term):
    # If number of berths known: berth occupancy per terminal
    if number_of_berths > 0:
        df_berth_occupancy = berth_occupancy(df, number_of_berths, operating_hours)
    else:
        df_berth_occupancy = 0
        print('The input for number of berths was not specified, thus the berth occupancy can not be defined')

    # If length of terminal can be fully occupied, determine length occupancy:
    if length_term > 0:
        df_length_occupancy = length_occupancy(df, length_term, operating_hours)
    else:
        df_length_occupancy = 0
        print('The length of terminal was not specified, thus the length occupancy can not be defined')

    return df_berth_occupancy, df_length_occupancy, df


# Plot occupancy (berth or length)
def plot_occupancy(df_length_occupancy, df_berth_occupancy, total_length, number_of_berths):
    # Length occupancy
    if total_length > 0:
        plt.figure()
        plt.xlabel('Time [per hour]')
        plt.ylabel('Length occupancy [%]')
        plt.title('Adjusted length occupancy over time')
        plt.plot(df_length_occupancy.timestamp, df_length_occupancy['occupancy_length_adjust[%]'])
        plt.plot([df_length_occupancy.timestamp.min(), df_length_occupancy.timestamp.max()], [100, 100], linestyle='--',
                 label='Maximum length available')
        plt.plot([df_length_occupancy.timestamp.min(), df_length_occupancy.timestamp.max()],
                 [df_length_occupancy['occupancy_length_adjust[%]'].mean(),
                  df_length_occupancy['occupancy_length_adjust[%]'].mean()], linestyle='--',
                 label='Average occupancy')
        plt.legend()
        plt.show()

    # Berth occupancy
    if number_of_berths > 0:
        plt.figure()
        plt.xlabel('Time [per hour]')
        plt.ylabel('Occupancy at terminal [%]')
        plt.title('Terminal occupancy over time')
        plt.plot(df_berth_occupancy.timestamp, df_berth_occupancy['occupancy[%]'])

        x = np.linspace(1, number_of_berths, number_of_berths)
        for i in x:
            berth = [(100 / number_of_berths) * i, (100 / number_of_berths) * i]
            plt.plot([df_berth_occupancy.timestamp.min(), df_berth_occupancy.timestamp.max()], berth, linestyle='--',
                     label=i)
        plt.legend(title='Berth')
        plt.show()


# Plot service times
def plot_service_times(data, title_name):
    # Histogram
    plt.figure()
    plt.hist(data['service_time[hr]'], bins=50)
    plt.title(title_name)
    plt.xlabel('Time [hr]')
    plt.ylabel('Number of vessel tracks')
    plt.show()


# Plot inter arrival times, based on port entry sequence
def plot_inter_arrival_times(data, title_name):
    # Histogram
    plt.figure()
    plt.hist(data['inter_arrival_time_port[hr]'], bins=50)
    plt.title(title_name)
    plt.xlabel('Time [hr]')
    plt.ylabel('Number of vessel tracks')
    plt.show()


# Plot CDF for inter arrival and service times per vessel class
def plot_distr_iat_st_class(df, bin_number, terminal_type, df_iat, df_st):
    # Split data set into different classes
    if terminal_type == 'container':
        df_1, df_2, df_3, df_4, df_5 = split_data_frame_class(df, 'container')
        list_class = ['total', 'c_1', 'c_2', 'c_3', 'c_4', 'c_5']
    elif terminal_type == 'dry_bulk':
        df_1, df_2, df_3, df_4, df_5 = split_data_frame_class(df, 'dry_bulk')
        list_class = ['total', 'db_1', 'db_2', 'db_3', 'db_4', 'db_5']
    elif terminal_type == 'liquid_bulk':
        df_1, df_2, df_3, df_4 = split_data_frame_class(df, 'liquid_bulk')
        list_class = ['total', 'lng_1', 'lng_2', 'lng_3', 'lng_4']
    else:
        print('No correct terminal type was given')

    # Plot CDF IAT
    hx_df, hy_df = np.histogram(df['inter_arrival_time_port[hr]'], bins=bin_number, density=True)
    hx_df_1, hy_df_1 = np.histogram(df_1['inter_arrival_time_port[hr]'], bins=bin_number, density=True)
    hx_df_2, hy_df_2 = np.histogram(df_2['inter_arrival_time_port[hr]'], bins=bin_number, density=True)
    hx_df_3, hy_df_3 = np.histogram(df_3['inter_arrival_time_port[hr]'], bins=bin_number, density=True)
    hx_df_4, hy_df_4 = np.histogram(df_4['inter_arrival_time_port[hr]'], bins=bin_number, density=True)
    if terminal_type != 'liquid_bulk':
        hx_df_5, hy_df_5 = np.histogram(df_5['inter_arrival_time_port[hr]'], bins=bin_number, density=True)

    plt.figure()
    plt.step(hy_df[1:], (np.cumsum(hx_df) * (hy_df[1] - hy_df[0])), 'k-',
             label=(list_class[0] + ' / N = ' + str(df['inter_arrival_time_port[hr]'].count())))
    plt.step(hy_df_1[1:], (np.cumsum(hx_df_1) * (hy_df_1[1] - hy_df_1[0])),
             label=(list_class[1] + ' / N = ' + str(df_1['inter_arrival_time_port[hr]'].count())))
    plt.step(hy_df_2[1:], (np.cumsum(hx_df_2) * (hy_df_2[1] - hy_df_2[0])),
             label=(list_class[2] + ' / N = ' + str(df_2['inter_arrival_time_port[hr]'].count())))
    plt.step(hy_df_3[1:], (np.cumsum(hx_df_3) * (hy_df_3[1] - hy_df_3[0])),
             label=(list_class[3] + ' / N = ' + str(df_3['inter_arrival_time_port[hr]'].count())))
    plt.step(hy_df_4[1:], (np.cumsum(hx_df_4) * (hy_df_4[1] - hy_df_4[0])),
             label=(list_class[4] + ' / N = ' + str(df_4['inter_arrival_time_port[hr]'].count())))
    if terminal_type != 'liquid_bulk':
        plt.step(hy_df_5[1:], (np.cumsum(hx_df_5) * (hy_df_5[1] - hy_df_5[0])),
             label=(list_class[5] + ' / N = ' + str(df_5['inter_arrival_time_port[hr]'].count())))
    plt.legend(title='Vessel class / number of arrivals')
    plt.ylabel('Cdf')
    plt.xlabel('Inter arrival times [hr]')
    plt.title(location + ': inter arrival times per class (CDF)')
    plt.xlim(0, df['inter_arrival_time_port[hr]'].max())
    plt.ylim(0, 1)
    plt.show()

    # Plot CDF ST
    hx_df, hy_df = np.histogram(df['service_time[hr]'], bins=bin_number, density=True)
    hx_df_1, hy_df_1 = np.histogram(df_1['service_time[hr]'], bins=bin_number, density=True)
    hx_df_2, hy_df_2 = np.histogram(df_2['service_time[hr]'], bins=bin_number, density=True)
    hx_df_3, hy_df_3 = np.histogram(df_3['service_time[hr]'], bins=bin_number, density=True)
    hx_df_4, hy_df_4 = np.histogram(df_4['service_time[hr]'], bins=bin_number, density=True)
    if terminal_type != 'liquid_bulk':
        hx_df_5, hy_df_5 = np.histogram(df_5['service_time[hr]'], bins=bin_number, density=True)

    plt.figure()
    plt.step(hy_df[1:], (np.cumsum(hx_df) * (hy_df[1] - hy_df[0])), 'k-',
             label=(list_class[0] + ' / N = ' + str(df['service_time[hr]'].count())))
    plt.step(hy_df_1[1:], (np.cumsum(hx_df_1) * (hy_df_1[1] - hy_df_1[0])),
             label=(list_class[1] + ' / N = ' + str(df_1['service_time[hr]'].count())))
    plt.step(hy_df_2[1:], (np.cumsum(hx_df_2) * (hy_df_2[1] - hy_df_2[0])),
             label=(list_class[2] + ' / N = ' + str(df_2['service_time[hr]'].count())))
    plt.step(hy_df_3[1:], (np.cumsum(hx_df_3) * (hy_df_3[1] - hy_df_3[0])),
             label=(list_class[3] + ' / N = ' + str(df_3['service_time[hr]'].count())))
    plt.step(hy_df_4[1:], (np.cumsum(hx_df_4) * (hy_df_4[1] - hy_df_4[0])),
             label=(list_class[4] + ' / N = ' + str(df_4['service_time[hr]'].count())))
    if terminal_type != 'liquid_bulk':
        plt.step(hy_df_5[1:], (np.cumsum(hx_df_5) * (hy_df_5[1] - hy_df_5[0])),
                 label=(list_class[5] + ' / N = ' + str(df_5['service_time[hr]'].count())))
    plt.legend(title='Vessel class / number of arrivals')
    plt.ylabel('Cdf')
    plt.xlim(0, df['service_time[hr]'].max())
    plt.ylim(0, 1)
    plt.xlabel('Service times [hr]')
    plt.title(location + ': service times per class (CDF)')
    plt.show()

    # Plot description of IAT
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Inter arrival time distribution characteristics, per vessel class')

    ax1.plot(df_iat['vessel_class'], df_iat['mean'], label='mean')
    ax1.plot(df_iat['vessel_class'], df_iat['std'], label='std')
    ax1.plot(df_iat['vessel_class'], df_iat['min'], label='min')
    ax1.plot(df_iat['vessel_class'], df_iat['25%'], label='25%')
    ax1.plot(df_iat['vessel_class'], df_iat['50%'], label='50%')
    ax1.plot(df_iat['vessel_class'], df_iat['75%'], label='75%')
    ax1.plot(df_iat['vessel_class'], df_iat['max'], label='max')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Vessel class')
    ax1.set_ylabel('Inter arrival time [hr]')

    ax2.plot(df_iat['vessel_class'], df_iat['mean'], label='mean')
    ax2.plot(df_iat['vessel_class'], df_iat['std'], label='std')
    ax2.plot(df_iat['vessel_class'], df_iat['min'], label='min')
    ax2.plot(df_iat['vessel_class'], df_iat['25%'], label='25%')
    ax2.plot(df_iat['vessel_class'], df_iat['50%'], label='50%')
    ax2.plot(df_iat['vessel_class'], df_iat['75%'], label='75%')
    ax2.set_xlabel('Vessel class')
    plt.show()

    # Plot description of ST
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Service time distribution characteristics, per vessel class')

    ax1.plot(df_st['vessel_class'], df_st['mean'], label='mean')
    ax1.plot(df_st['vessel_class'], df_st['std'], label='std')
    ax1.plot(df_st['vessel_class'], df_st['min'], label='min')
    ax1.plot(df_st['vessel_class'], df_st['25%'], label='25%')
    ax1.plot(df_st['vessel_class'], df_st['50%'], label='50%')
    ax1.plot(df_st['vessel_class'], df_st['75%'], label='75%')
    ax1.plot(df_st['vessel_class'], df_st['max'], label='max')
    ax1.legend(loc='upper left')
    ax1.set_xlabel('Vessel class')
    ax1.set_ylabel('Service time [hr]')

    ax2.plot(df_st['vessel_class'], df_st['mean'], label='mean')
    ax2.plot(df_st['vessel_class'], df_st['std'], label='std')
    ax2.plot(df_st['vessel_class'], df_st['min'], label='min')
    ax2.plot(df_st['vessel_class'], df_st['25%'], label='25%')
    ax2.plot(df_st['vessel_class'], df_st['50%'], label='50%')
    ax2.plot(df_st['vessel_class'], df_st['75%'], label='75%')
    ax2.set_xlabel('Vessel class')
    plt.show()


# Fit distributions on inter arrival time distribution + plot, per data frame
def iat_distributions(df, location):
    if df.shape[0] > 0:
        data = df['inter_arrival_time_port[hr]']
        bin_number = 100
        y, x = np.histogram(data, bins=bin_number, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        x_0 = np.linspace(0, data.max(), 100)
        distribution = []
        location_par = []
        scale_par = []
        shape_par = []
        D = []
        p = []
        chi_p = []
        chi = []

        histo, bin_edges = np.histogram(data, bins=bin_number, density=False)
        observed_values = histo
        n = len(data)

        # Exponential (same as Gamma with a = 1 and Weibull with c =1)
        distribution.append('Exponential')
        exp_loc, exp_scale = scipy.stats.distributions.expon.fit(data)
        location_par.append((np.around(exp_loc, 4)))
        scale_par.append((np.around(exp_scale, 4)))
        shape_par.append((np.around(1, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.expon.cdf, args=scipy.stats.expon.fit(data))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.expon.cdf, args=scipy.stats.expon.fit(data))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.expon.cdf(bin_edges, exp_loc, exp_scale))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Gamma
        distribution.append('Gamma')
        a_gam, loc_gam, scale_gam = scipy.stats.distributions.gamma.fit(data)
        location_par.append((np.around(loc_gam, 4)))
        scale_par.append((np.around(scale_gam, 4)))
        shape_par.append((np.around(a_gam, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.gamma.cdf(bin_edges, a_gam, loc_gam, scale_gam))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Erlang-2 (same as Gamma with a = 2)
        distribution.append('Erlang-2')
        a_gam2, loc_gam2, scale_gam2 = scipy.stats.distributions.gamma.fit(data, fa=2)
        location_par.append((np.around(loc_gam2, 4)))
        scale_par.append((np.around(scale_gam2, 4)))
        shape_par.append((np.around(a_gam2, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=2))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=2))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.gamma.cdf(bin_edges, a_gam2, loc_gam2, scale_gam2))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Weibull distribution
        distribution.append('Weibull')
        c_weib, loc_weib, scale_weib = scipy.stats.distributions.weibull_min.fit(data)
        location_par.append((np.around(loc_weib, 4)))
        scale_par.append((np.around(scale_weib, 4)))
        shape_par.append((np.around(c_weib, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.weibull_min.cdf,
                                                     args=scipy.stats.weibull_min.fit(data))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.weibull_min.cdf,
                                                     args=scipy.stats.weibull_min.fit(data))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.weibull_min.cdf(bin_edges, c_weib, loc_weib, scale_weib))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Create table with results
        df_results_iat = pd.DataFrame(distribution, columns=['Distribution_type'])
        df_results_iat['Location_parameter'] = location_par
        df_results_iat['Scale_parameter'] = scale_par
        df_results_iat['Shape_parameter'] = shape_par
        df_results_iat['ks_D'] = D
        df_results_iat['ks_p'] = p
        df_results_iat['ks_p_lim'] = 'undefined'
        df_results_iat['chi_p'] = chi_p
        df_results_iat['chi'] = chi
        df_results_iat['chi_p_lim'] = 'undefined'
        for row in df_results_iat.itertuples():
            if float(row.ks_p) > 0.05:
                df_results_iat.at[row.Index, 'ks_p_lim'] = 'Yes'
            else:
                df_results_iat.at[row.Index, 'ks_p_lim'] = 'No'
        for row in df_results_iat.itertuples():
            if float(row.chi_p) > 0.05:
                df_results_iat.at[row.Index, 'chi_p_lim'] = 'Yes'
            else:
                df_results_iat.at[row.Index, 'chi_p_lim'] = 'No'

        # Plot distributions
        # data cdf
        hx, hy = np.histogram(data, bins=bin_number, density=True)
        dx = hy[1] - hy[0]
        F1 = np.cumsum(hx) * dx
        plt.figure()
        plt.step(hy[1:], F1, 'k-', label='Data')
        # fitted cdf
        plt.plot(x_0, scipy.stats.distributions.expon.cdf(x_0, exp_loc, exp_scale), 'r-', label='Exponential')
        plt.plot(x_0, scipy.stats.distributions.gamma.cdf(x_0, a_gam, loc_gam, scale_gam), 'g-', label='Gamma')
        plt.plot(x_0, scipy.stats.distributions.gamma.cdf(x_0, a_gam2, loc_gam2, scale_gam2), 'b-', label='Erlang-2')
        plt.plot(x_0, scipy.stats.distributions.weibull_min.cdf(x_0, c_weib, loc_weib, scale_weib), 'y-', label='Weibull')
        plt.legend()
        plt.ylabel('Cdf')
        plt.xlim(0, data.max())
        plt.ylim(0, 1)
        plt.xlabel('Inter arrival times [hr]')
        plt.title(location + ': distribution fitting for inter arrival times')
        plt.show()

    else:
        df_results_iat = pd.DataFrame()

    return df_results_iat


# Fit distributions on inter arrival time distribution + plot, per data frame
def st_distributions(df, location):
    if df.shape[0] > 0:
        data = df['service_time[hr]']
        bin_number = 100
        x_0 = np.linspace(0, data.max(), 100)
        y, x = np.histogram(data, bins=bin_number, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        x_0 = np.linspace(0, data.max(), 100)
        distribution = []
        location_par = []
        scale_par = []
        shape_par = []
        shape_2_par= []
        D = []
        p = []
        chi_p = []
        chi = []

        histo, bin_edges = np.histogram(data, bins=bin_number, density=False)
        observed_values = histo
        n = len(data)

        # Exponential (same as Gamma with a = 1 and Weibull with c =1)
        distribution.append('Exponential')
        exp_loc, exp_scale = scipy.stats.distributions.expon.fit(data)
        location_par.append((np.around(exp_loc, 4)))
        scale_par.append((np.around(exp_scale, 4)))
        shape_par.append((np.around(1, 4)))
        shape_2_par.append((np.around(np.nan, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.expon.cdf, args=scipy.stats.expon.fit(data))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.expon.cdf, args=scipy.stats.expon.fit(data))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.expon.cdf(bin_edges, exp_loc, exp_scale))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Gamma
        distribution.append('Gamma')
        a_gam, loc_gam, scale_gam = scipy.stats.distributions.gamma.fit(data)
        location_par.append((np.around(loc_gam, 4)))
        scale_par.append((np.around(scale_gam, 4)))
        shape_par.append((np.around(a_gam, 4)))
        shape_2_par.append((np.around(np.nan, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.gamma.cdf(bin_edges, a_gam, loc_gam, scale_gam))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Erlang-2 (same as Gamma with a = 2)
        distribution.append('Erlang-2')
        a_gam2, loc_gam2, scale_gam2 = scipy.stats.distributions.gamma.fit(data, fa=2)
        location_par.append((np.around(loc_gam2, 4)))
        scale_par.append((np.around(scale_gam2, 4)))
        shape_par.append((np.around(a_gam2, 4)))
        shape_2_par.append((np.around(np.nan, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=2))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=2))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.gamma.cdf(bin_edges, a_gam2, loc_gam2, scale_gam2))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Erlang-3 (same as Gamma with a = 3)
        distribution.append('Erlang-3')
        a_gam3, loc_gam3, scale_gam3 = scipy.stats.distributions.gamma.fit(data, fa=3)
        location_par.append((np.around(loc_gam3, 4)))
        scale_par.append((np.around(scale_gam3, 4)))
        shape_par.append((np.around(a_gam3, 4)))
        shape_2_par.append((np.around(np.nan, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=3))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=3))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.gamma.cdf(bin_edges, a_gam3, loc_gam3, scale_gam3))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Erlang-4 (same as Gamma with a = 4)
        distribution.append('Erlang-4')
        a_gam4, loc_gam4, scale_gam4 = scipy.stats.distributions.gamma.fit(data, fa=4)
        location_par.append((np.around(loc_gam4, 4)))
        scale_par.append((np.around(scale_gam4, 4)))
        shape_par.append((np.around(a_gam4, 4)))
        shape_2_par.append((np.around(np.nan, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=4))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=4))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.gamma.cdf(bin_edges, a_gam4, loc_gam4, scale_gam4))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Erlang-5 (same as Gamma with a = 5)
        distribution.append('Erlang-5')
        a_gam5, loc_gam5, scale_gam5 = scipy.stats.distributions.gamma.fit(data, fa=5)
        location_par.append((np.around(loc_gam5, 4)))
        scale_par.append((np.around(scale_gam5, 4)))
        shape_par.append((np.around(a_gam5, 4)))
        shape_2_par.append((np.around(np.nan, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=5))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.gamma.cdf,
                                                     args=scipy.stats.gamma.fit(data, fa=5))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.gamma.cdf(bin_edges, a_gam5, loc_gam5, scale_gam5))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Normal
        distribution.append('Normal')
        loc_norm, scale_norm = scipy.stats.distributions.norm.fit(data)
        location_par.append((np.around(loc_norm, 4)))
        scale_par.append((np.around(scale_norm, 4)))
        shape_par.append((np.around(np.nan, 4)))
        shape_2_par.append((np.around(np.nan, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.norm.cdf, args=scipy.stats.norm.fit(data))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.norm.cdf, args=scipy.stats.norm.fit(data))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.norm.cdf(bin_edges, loc_norm, scale_norm))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Beta
        distribution.append('Beta')
        a_beta, b_beta, loc_beta, scale_beta = scipy.stats.distributions.beta.fit(data)
        location_par.append((np.around(loc_beta, 4)))
        scale_par.append((np.around(scale_beta, 4)))
        shape_par.append((np.around(a_beta, 4)))
        shape_2_par.append((np.around(b_beta, 4)))
        D.append((np.around((scipy.stats.kstest(data, scipy.stats.beta.cdf, args=scipy.stats.beta.fit(data))[0]), 4)))
        p.append((np.around((scipy.stats.kstest(data, scipy.stats.beta.cdf, args=scipy.stats.beta.fit(data))[1]), 4)))
        expected_values = n * np.diff(scipy.stats.beta.cdf(bin_edges, a_beta, b_beta, loc_beta, scale_beta))
        chi_p.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[1]), 4)))
        chi.append((np.around((scipy.stats.chisquare(observed_values, expected_values)[0]), 4)))

        # Create table with results
        df_results_st = pd.DataFrame(distribution, columns=['Distribution_type'])
        df_results_st['Location_parameter'] = location_par
        df_results_st['Scale_parameter'] = scale_par
        df_results_st['Shape_parameter'] = shape_par
        df_results_st['Shape_parameter(b)'] = shape_2_par
        df_results_st['ks_D'] = D
        df_results_st['ks_p'] = p
        df_results_st['ks_p_lim'] = 'undefined'
        df_results_st['chi_p'] = chi_p
        df_results_st['chi'] = chi
        df_results_st['chi_p_lim'] = 'undefined'
        for row in df_results_st.itertuples():
            if float(row.ks_p) > 0.05:
                df_results_st.at[row.Index, 'ks_p_lim'] = 'Yes'
            else:
                df_results_st.at[row.Index, 'ks_p_lim'] = 'No'
        for row in df_results_st.itertuples():
            if float(row.chi_p) > 0.05:
                df_results_st.at[row.Index, 'chi_p_lim'] = 'Yes'
            else:
                df_results_st.at[row.Index, 'chi_p_lim'] = 'No'

        # Plot distributions
        # data cdf
        hx, hy= np.histogram(data, bins=bin_number, density=True)
        dx = hy[1] - hy[0]
        F1 = np.cumsum(hx) * dx
        plt.figure()
        plt.step(hy[1:], F1, 'k-', label='Data')
        # fitted cdf
        plt.plot(x_0, scipy.stats.distributions.expon.cdf(x_0, exp_loc, exp_scale), 'r-', label='Exponential')
        plt.plot(x_0, scipy.stats.distributions.gamma.cdf(x_0, a_gam, loc_gam, scale_gam), 'g-', label='Gamma')
        plt.plot(x_0, scipy.stats.distributions.gamma.cdf(x_0, a_gam2, loc_gam2, scale_gam2), 'b-', label='Erlang-2')
        plt.plot(x_0, scipy.stats.distributions.gamma.cdf(x_0, a_gam3, loc_gam3, scale_gam3), 'm-', label='Erlang-3')
        plt.plot(x_0, scipy.stats.distributions.gamma.cdf(x_0, a_gam4, loc_gam4, scale_gam4), 'y-', label='Erlang-4')
        plt.plot(x_0, scipy.stats.distributions.gamma.cdf(x_0, a_gam5, loc_gam5, scale_gam5), 'orange', label='Erlang-5')
        plt.plot(x_0, scipy.stats.distributions.norm.cdf(x_0,  loc_norm, scale_norm), 'c-', label='Normal')
        plt.plot(x_0, scipy.stats.distributions.beta.cdf(x_0,a_beta, b_beta, loc_beta, scale_beta), 'pink', label='Beta')
        plt.legend()
        plt.ylabel('Cdf')
        plt.xlabel('Service times [hr]')
        plt.xlim(0, data.max())
        plt.ylim(0, 1)
        plt.title(location + ': distribution fitting for service times')
        plt.show()

    else:
        df_results_st = pd.DataFrame()

    return df_results_st


# Fit all distributions for all terminal types
def fit_all_iat_distr(df, location, terminal_type):
    if terminal_type == 'container' or terminal_type == 'dry_bulk':
        df_tot = iat_distributions(df, (location + ' All classes'))
        df_iat_1 = iat_distributions(df_1, (location + ' Class 1'))
        df_iat_2 = iat_distributions(df_2, (location + ' Class 2'))
        df_iat_3 = iat_distributions(df_3, (location + ' Class 3'))
        df_iat_4 = iat_distributions(df_4, (location + ' Class 4'))
        df_iat_5 = iat_distributions(df_5, (location + ' Class 5'))
        return df_tot, df_iat_1, df_iat_2, df_iat_3, df_iat_4, df_iat_5
    elif terminal_type == 'liquid_bulk':
        df_tot = iat_distributions(df, (location + ' All classes'))
        df_iat_1 = iat_distributions(df_1, (location + ' Class 1'))
        df_iat_2 = iat_distributions(df_2, (location + ' Class 2'))
        df_iat_3 = iat_distributions(df_3, (location + ' Class 3'))
        df_iat_4 = iat_distributions(df_4, (location + ' Class 4'))
        return df_tot, df_iat_1, df_iat_2, df_iat_3, df_iat_4


# Fit all distributions for all terminal types
def fit_all_st_distr(df, location, terminal_type):
    if terminal_type == 'container' or terminal_type == 'dry_bulk':
        df_tot = st_distributions(df, (location + ' All classes'))
        df_st_1 = st_distributions(df_1, (location + ' Class 1'))
        df_st_2 = st_distributions(df_2, (location + ' Class 2'))
        df_st_3 = st_distributions(df_3, (location + ' Class 3'))
        df_st_4 = st_distributions(df_4, (location + ' Class 4'))
        df_st_5 = st_distributions(df_5, (location + ' Class 5'))
        return df_tot, df_st_1, df_st_2, df_st_3, df_st_4, df_st_5
    elif terminal_type == 'liquid_bulk':
        df_tot = st_distributions(df, (location + ' All classes'))
        df_st_1 = st_distributions(df_1, (location + ' Class 1'))
        df_st_2 = st_distributions(df_2, (location + ' Class 2'))
        df_st_3 = st_distributions(df_3, (location + ' Class 3'))
        df_st_4 = st_distributions(df_4, (location + ' Class 4'))
        return df_tot, df_st_1, df_st_2, df_st_3, df_st_4


# Test handle
if __name__ == '__main__':
    """ ....... INPUTS ......... """
    # Terminal location
    location = 'ct_rdam_apm2'
    # Choose terminal type (Options: 'container', 'dry_bulk', 'liquid_bulk')
    terminal_type = 'container'
    # Number of berths: (1,2,3... number, or if unknown: 0)
    number_of_berths = 0
    # Operating hours per year:
    operating_hours = 365 * 24
    # Total length terminal [m] (if unknown: 0)
    length_term = 1500

    # Load data
    df = pd.read_csv('Data-frames/Results_phase_3/' + location + '/Final_df_' + location + '.csv')

    # Add service times
    df = service_times(df)

    # Based on port entry arrival time, return inter arrival time
    df_p = sort_by_port_entry(df)

    # Based on port entry arrival time, return all timestamps relative to t0
    df_p_rel = sort_by_port_entry_rel(df_p)

    # Add vessel class categorizations dependent on terminal type
    df_p['vessel_class'] = add_vessel_class(df_p, terminal_type)

    # Split data set into class categorizations
    if terminal_type == 'container':
        df_1, df_2, df_3, df_4, df_5 = split_data_frame_class(df_p, 'container')
    if terminal_type == 'dry_bulk':
        df_1, df_2, df_3, df_4, df_5 = split_data_frame_class(df_p, 'dry_bulk')
    if terminal_type == 'liquid_bulk':
        df_1, df_2, df_3, df_4 = split_data_frame_class(df_p, 'liquid_bulk')

    # Fit all distributions for inter arrival times
    if terminal_type == 'container':
        df_tot_iat, df_1_iat, df_2_iat, df_3_iat, df_4_iat, df_5_iat = fit_all_iat_distr(df_p, location, 'container')
    if terminal_type == 'dry_bulk':
        df_tot_iat, df_1_iat, df_2_iat, df_3_iat, df_4_iat, df_5_iat = fit_all_iat_distr(df_p, location, 'dry_bulk')
    if terminal_type == 'liquid_bulk':
        df_tot_iat, df_1_iat, df_2_iat, df_3_iat, df_4_iat = fit_all_iat_distr(df_p, location, 'liquid_bulk')

    # Fit all distribution for service times
    if terminal_type == 'container':
        df_tot_st, df_1_st, df_2_st, df_3_st, df_4_st, df_5_st = fit_all_st_distr(df_p, location, 'container')
    if terminal_type == 'dry_bulk':
        df_tot_st, df_1_st, df_2_st, df_3_st, df_4_st, df_5_st = fit_all_st_distr(df_p, location, 'dry_bulk')
    if terminal_type == 'liquid_bulk':
        df_tot_st, df_1_st, df_2_st, df_3_st, df_4_st = fit_all_st_distr(df_p, location, 'liquid_bulk')

    """ ...... RESULTS ......  """
    # Number of vessel arrivals per certain vessel class
    print('Number of vessel arrivals per certain vessel class:', df_p.groupby(['type']).count()['terminal_entry_time'])

    # All descriptive results
    df_describe_iat, df_describe_st = results_all(df_p, terminal_type)

    # Run all occupancy steps
    df_berth_occupancy, df_length_occupancy, df_all = run_all_occupancy(df, number_of_berths, operating_hours,
                                                                        length_term)

    """ ....... VISUALS ........ """
    # Plot class arrivals (per type/class)
    vessel_arrivals_per_class_type(df_p)

    # Plot service times and inter arrival times (based on port entry)
    plot_service_times(df_p, location + ': Service times')
    plot_inter_arrival_times(df_p, location + ': Inter arrival times based on port entry')

    # Plot distributions IAT and ST per vessel class
    plot_distr_iat_st_class(df_p, 50, terminal_type, df_describe_iat, df_describe_st)

    # Plot occupancy
    plot_occupancy(df_length_occupancy, df_berth_occupancy, length_term, number_of_berths)
    #
    """ .... EXPORT DATA FRAMES ......"""
    df_p.to_csv('Data-frames/Results_phase_3/' + location + '/Df_stats_' + location + '.csv')
    df_describe_iat.to_csv('Data-frames/Results_phase_3/' + location + '/Df_describe_iat' + location + '.csv')
    df_describe_st.to_csv('Data-frames/Results_phase_3/' + location + '/Df_describe_st' + location + '.csv')

    # Only if length occupancy is available
    if length_term > 0:
        df_length_occupancy.to_csv('Data-frames/Results_phase_3/' + location + '/Df_length_occup_' + location + '.csv')
    # Only if terminal occupancy is available
    if number_of_berths > 0:
        df_berth_occupancy.to_csv('Data-frames/Results_phase_3/' + location + '/Df_berth_occup_' + location + '.csv')

    # Fitted distributions
    df_tot_iat.to_csv('Data-frames/Results_phase_3/' + location + '/Df_iat_tot' + location + '.csv')
    df_1_iat.to_csv('Data-frames/Results_phase_3/' + location + '/Df_iat_1' + location + '.csv')
    df_2_iat.to_csv('Data-frames/Results_phase_3/' + location + '/Df_iat_2' + location + '.csv')
    df_3_iat.to_csv('Data-frames/Results_phase_3/' + location + '/Df_iat_3' + location + '.csv')
    df_4_iat.to_csv('Data-frames/Results_phase_3/' + location + '/Df_iat_4' + location + '.csv')
    df_tot_st.to_csv('Data-frames/Results_phase_3/' + location + '/Df_st_tot' + location + '.csv')
    df_1_st.to_csv('Data-frames/Results_phase_3/' + location + '/Df_st_1' + location + '.csv')
    df_2_st.to_csv('Data-frames/Results_phase_3/' + location + '/Df_st_2' + location + '.csv')
    df_3_st.to_csv('Data-frames/Results_phase_3/' + location + '/Df_st_3' + location + '.csv')
    df_4_st.to_csv('Data-frames/Results_phase_3/' + location + '/Df_st_4' + location + '.csv')

    if terminal_type == 'container' or terminal_type == 'dry_bulk':
        df_5_iat.to_csv('Data-frames/Results_phase_3/' + location + '/Df_iat_5' + location + '.csv')
        df_5_st.to_csv('Data-frames/Results_phase_3/' + location + '/Df_st_5' + location + '.csv')




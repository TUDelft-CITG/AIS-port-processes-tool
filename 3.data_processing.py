import pandas as pd
import geopy.distance
import math
from data_cleaning import drop_and_report


# Determine sog 
def add_sog_cog_to_data_new(data):
    # Cutoff time after which the vessel is presumed to have left and returned. This will lease the speed and course to
    # be calculated as if it is the first/last value in a new series
    cut_off_time = '2H'

    # Add columns to data
    data['sog_ms'] = 0.0
    for row in data.itertuples():

        # First row of data set
        if row.Index == 0 \
                and data.at[row.Index + 1, 'mmsi'] == row.mmsi \
                and (data.at[row.Index + 1, 'timestamp'] - row.timestamp) < pd.Timedelta(cut_off_time):
            next_row = data.iloc[row.Index + 1]
            sog = add_sog_cog_first_row(row, next_row)

        # First row of new MMSI set
        elif row.Index != 0 and data.at[row.Index - 1, 'mmsi'] != row.mmsi and (row.Index != len(data) - 1) \
                and data.at[row.Index + 1, 'mmsi'] == row.mmsi \
                and (data.at[row.Index + 1, 'timestamp'] - row.timestamp) < pd.Timedelta(cut_off_time):
            next_row = data.iloc[row.Index + 1]
            sog = add_sog_cog_first_row(row, next_row)

        # First row of MMSI set after X number of hours
        elif row.Index != 0 and data.at[row.Index - 1, 'mmsi'] == row.mmsi and (row.Index != len(data) - 1) \
                and data.at[row.Index + 1, 'mmsi'] == row.mmsi \
                and (row.timestamp - data.at[row.Index - 1, 'timestamp']) > pd.Timedelta(cut_off_time):
            next_row = data.iloc[row.Index + 1]
            sog = add_sog_cog_first_row(row, next_row)

        # Last row of data set
        elif row.Index == len(data) - 1 \
                and data.at[row.Index - 1, 'mmsi'] == row.mmsi \
                and (row.timestamp - data.at[row.Index - 1, 'timestamp']) < pd.Timedelta(cut_off_time):
            prev_row = data.iloc[row.Index - 1]
            sog = add_sog_cog_last_row(prev_row, row)

        # Last row of MMSI set
        elif row.Index != (len(data) - 1) and row.Index != 0 and data.at[row.Index - 1, 'mmsi'] == row.mmsi \
                and data.at[row.Index + 1, 'mmsi'] != row.mmsi \
                and (row.timestamp - data.at[row.Index - 1, 'timestamp']) < pd.Timedelta(cut_off_time):
            prev_row = data.iloc[row.Index - 1]
            sog = add_sog_cog_last_row(prev_row, row)

        # Middle rows
        elif row.Index != 0 and row.Index != (len(data) - 1) and data.at[row.Index - 1, 'mmsi'] == row.mmsi \
                and data.at[row.Index + 1, 'mmsi'] == row.mmsi \
                and (row.timestamp - data.at[row.Index - 1, 'timestamp']) < pd.Timedelta(cut_off_time) \
                and (data.at[row.Index + 1, 'timestamp'] - row.timestamp) < pd.Timedelta(cut_off_time):
            prev_row = data.iloc[row.Index - 1]
            next_row = data.iloc[row.Index + 1]
            sog = add_sog_cog_middle_row(prev_row, row, next_row)

        else:
            sog = add_sog_cog_single_row()

        data.at[row.Index, 'sog_ms'] = sog

    return data


def add_sog_cog_first_row(row, next_row):
    #SOG
    new_distance = geopy.distance.distance((row.lat, row.lon), (next_row['lat'], next_row['lon'])).m
    new_time = (next_row.timestamp - row.timestamp).total_seconds()
    sog = new_distance / new_time

    return sog


def add_sog_cog_last_row(prev_row, row):
    #SOG
    old_distance = geopy.distance.distance((prev_row['lat'], prev_row['lon']), (row.lat, row.lon)).m
    old_time = (row.timestamp - prev_row.timestamp).total_seconds()
    sog = old_distance / old_time

    return sog


def add_sog_cog_middle_row(prev_row, row, next_row):
    # SOG
    old_distance = geopy.distance.distance((prev_row['lat'], prev_row['lon']), (row.lat, row.lon)).m
    old_time = (row.timestamp - prev_row.timestamp).total_seconds()
    old_speed = old_distance / old_time
    new_distance = geopy.distance.distance((row.lat, row.lon), (next_row['lat'], next_row['lon'])).m
    new_time = (next_row.timestamp - row.timestamp).total_seconds()
    new_speed = new_distance / new_time
    sog = (old_speed + new_speed) / 2

    return sog


def add_sog_cog_single_row():
    sog = 0

    return sog



# Check speed values
def low_high_speed_check(data):
    # Initialize variables for low- and high-speed check
    max_speed = 25  # m/s
    min_speed = 0  # m/s

    drop_list = list()
    for row in data.itertuples():
        if row.sog_ms > max_speed or row.sog_ms < min_speed:
            drop_list.append(row.Index)

    data = drop_and_report(data, drop_list, 'Speed outlier')

    return data



# Run entire preprocessing steps
def process_data_all(data):
    if len(data) != 0:
        data = add_sog_cog_to_data_new(data)
        data = low_high_speed_check(data)
    else:
        print('No data was returned')

    return data


# Test handle
if __name__ == '__main__':
    import time
    from data_cleaning import clean_data_all

    data_read = pd.read_csv('SH-raw_incl.vesseltypes_01012020-01022020.csv')
    data_input = clean_data_all(data_read)

    start_time = time.time()
    data_1 = add_sog_cog_to_data_new(data_input)
    time_data_1 = time.time()
    print('Time for add_sog_cog_to_data_new:', time_data_1 - start_time, 'seconds')

    data_2 = low_high_speed_check(data_1)
    time_data_2 = time.time()
    print('Time for low_high_speed_check:', time_data_2 - time_data_1, 'seconds')

    # Determine the number of speed values that are undefined (equal to 0 m/s)
    speed = data_2.pivot_table(index=['sog_ms'], aggfunc='size')
    print('The percentage of values with an undefined speed =',
          round(speed[0] * 100 / data_2.shape[0], 3), '%')

    print('The total time for data_processing.py:', time.time() - start_time, 'seconds')

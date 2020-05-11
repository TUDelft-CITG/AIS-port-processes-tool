import pandas as pd
import geopy.distance
import math
from data_cleaning import drop_and_report


# Determine sog and cog
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


def calculate_compass_bearing(point_1, point_2):
    """ Calculates the compass bearing of a line between two points. This bearing ranges from 0 to 359.99 with 0 as
    north rotating clockwise.
    The formulae used is the following:
        θ = atan2(sin(Δlat).cos(lon2), cos(lon1).sin(lon2) − sin(lon1).cos(lon2).cos(Δlat))
    This is based on trigonometry in a curved space as opposed to the regular flat space theory often used. This adds to
    accuracy around the poles.
    :param point_1:     tuple containing (longitude, latitude) of a first AIS coordinate
    :param point_2:     tuple containing (longitude, latitude) of a second AIS coordinate
    :return:            the compass bearing of the line from point_1 to point_2
    """
    if (type(point_1) != tuple) or (type(point_2) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lon1 = math.radians(point_1[1])
    lon2 = math.radians(point_2[1])

    diffLat = math.radians(point_2[0] - point_1[0])

    x = math.sin(diffLat) * math.cos(lon2)
    y = math.cos(lon1) * math.sin(lon2) - (math.sin(lon1) * math.cos(lon2) * math.cos(diffLat))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to +180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


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


# Check course and location outliers
# Not necessary using small polygons (due to little movement = high likelihood of large variance of course)
def heading_check(data):
    """Checks heading difference between consecutive points and removes points which have very large differences.
    This check considers three consecutive points, and calculates the heading between them. The heading between point 1
    and point 2 is heading_1_2 and the heading between point 2 and point 3 is heading_2_3. If the absolute difference
    between these two headings is greater than 175 or smaller than 185 degrees, remove point 2. This would indicate a
    location outlier and it can catch certain data points which the location outlier check misses.
    Designed to work in tandem with location_outlier_check.
    :param data:    pandas Dataframe containing partially cleaned data
    :return:        pandas Dataframe containing further cleaned data with location outliers removed   """
    drop_list = list()
    cut_off_time = '2H'
    for row in data.itertuples():
        if row.Index == len(data) - 1:
            pass
        elif data.at[row.Index + 1, 'mmsi'] != row.mmsi:
            pass
        elif (data.at[row.Index + 1, 'timestamp'] - row.timestamp) < pd.Timedelta(cut_off_time):
            pass
        else:
            cog = row.cog
            next_cog = data.at[row.Index + 1, 'cog']
            if (abs(next_cog - cog) > 175) and (abs(next_cog - cog) < 185):
                drop_list.append(row.Index)

    data = drop_and_report(data, drop_list, 'Heading')

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

    # Determine the number of course values that are undefined (equal to 0 degrees)
    course = data_2.pivot_table(index=['cog'], aggfunc='size')
    print('The percentage of values with an undefined course =',
          round(course[0] * 100 / data_2.shape[0], 3), '%')

    print('The total time for data_preprocessing.py:', time.time() - start_time, 'seconds')

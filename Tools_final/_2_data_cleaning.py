""" Step 2. Data cleaning
 Input: Terminal data frame with vessel track labels
 Actions: Cleaning steps (rounding parameters, removing duplicates, deleting false inputs, location outlier check)
 Output: Cleaned terminal data frame
 """

import pandas as pd
import geopy.distance
from _1_data_gathering import drop_and_report


# Rounding input parameters
def clean_data_rounding(data):
    # Round coordinates to 6 decimals and times to seconds
    data.lat = data.lat.round(6)
    data.lon = data.lon.round(6)
    data.timestamp = data.timestamp.dt.round('1s')
    data.mmsi = data.mmsi.round(0)
    return data


# Removing duplicate rows
def clean_duplicates(data):
    """Further cleans the AIS data by removing duplicate points based on timestamp and location.
    If a point has the same mmsi and timestamp combination or the same location as the previous point it is removed.
    :param data:    pandas Dataframe containing partially cleaned data
    :return:        pandas Dataframe containing further cleaned data with rounded values and removed duplicates.
    """
    drop_list = list()
    for row in data.itertuples():
        # If the coordinates of the current data point exactly match the coordinates of the previous data point,
        # exclude the current data point, as that means it's double.
        # If the timestamp of the current row matches that of the previous row it is a duplicate.
        if (row.Index != 0) and ((row.lat == data.at[row.Index - 1, 'lat'] and row.lon ==
                                  data.at[row.Index - 1, 'lon']) or
                                 (row.timestamp == data.at[row.Index - 1, 'timestamp'] and
                                    row.mmsi == data.at[row.Index - 1, 'mmsi'])):
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Duplicate rows clean')
    return data


# Removing fault MMSI numbers and fault lat & lon numbers
def delete_faulty_inputs(data):
    """ Loops through data passed as a pandas Dataframe and deletes rows from data if:
        -   mmsi is not the expected length (9)
        -   latitude is outside range [-90, 90]
        -   longitude is outside range [-180, 180]
    :param data:    pandas Dataframe of partially cleaned data from the database
    :return:        pandas Dataframe of partially cleaned data with invalid rows removed.
    """
    drop_list = list()
    for row in data.itertuples():
        # If mmsi is not of expected length, exclude data point
        if len(str(row.mmsi)) != 9:
            drop_list.append(row.Index)
        # If coordinates are invalid, exclude data point
        elif (float(row.lat) < -90) or (float(row.lat) > 90):
            drop_list.append(row.Index)
        elif (float(row.lon) < -180) or (float(row.lon) > 180):
            drop_list.append(row.Index)

    data = drop_and_report(data, drop_list, 'Delete faulty inputs')
    return data


# Removing location outliers
def location_outlier_check(data):
    """Filters data points that are outliers in terms of coordinates.
    Works in the following way. Three points are considered, consecutive in time, say point 1, point 2 and point 3.
    If the distance between point 1 and point 3 is greater than 30 meters, find the average coordinate between the
    two points. This minimum distance ensures that this check is not done when the vessel is making a sharp turn.
    If the distance between point 2 (the point in between point 1 and 3) and this mid-point that we calculated is
    greater than the distance between point 1 and point 3, then discard point 2.
    :param data:    pandas Dataframe containing partially cleaned data
    :return:        pandas Dataframe containing further cleaned data with location outliers removed
    """
    # Cutoff time after which the vessel is presumed to have left and returned. This will lease the speed and course to
    # be calculated as if it is the first/last value in a new series
    cut_off_time = '2H'
    drop_list = list()
    for row in data.itertuples():
        # always keep the first row because we can't determine its validity
        if row.Index == 0:
            continue
        # always keep the first row of a new vessel series because we can't determine its validity
        elif data.at[row.Index - 1, 'mmsi'] != row.mmsi or \
                row.timestamp - data.at[row.Index - 1, 'timestamp'] > pd.Timedelta(cut_off_time):
            continue
        # always keep the last row because we can't determine its validity
        elif row.Index == len(data) - 1:
            continue
        # always keep the last row of a vessel series because we can't determine its validity
        elif data.at[row.Index + 1, 'mmsi'] != row.mmsi or \
                data.at[row.Index + 1, 'timestamp'] - row.timestamp > pd.Timedelta(cut_off_time):
            continue

        # Determine coordinates of previous and current point
        prev_coord = (data.at[row.Index - 1, 'lat'], data.at[row.Index - 1, 'lon'])
        coord = (row.lat, row.lon)
        next_coord = (data.at[row.Index + 1, 'lat'], data.at[row.Index + 1, 'lon'])
        # Determine coordinates of the midpoint between the previous and next data point and calculate the
        # distance between those two points.
        mid_coord = (((prev_coord[0] + next_coord[0]) / 2), ((prev_coord[1] + next_coord[1]) / 2))
        dist = geopy.distance.distance(prev_coord, next_coord).m

        # If the distance between the previous and next point is at least 30 meters, and the current point
        # is further away from the midpoint than that calculated distance, exclude the current data point.
        # The 30 meter cut off is to prevent tight turns during maneuvering being removed
        if dist > 30:
            if geopy.distance.distance(coord, mid_coord).m > dist:
                drop_list.append(row.Index)

    data = drop_and_report(data, drop_list, 'Location outlier')
    return data


# Run entire cleaning steps
def clean_data_all(data):
    """     Run a full cleaning algorithm on a set of data pulled for the database. Removes points which can not be
    correct or have grossly deviating mmsi/lon etc. Will only run if there is data otherwise it is skipped.
    :param data:    pandas Dataframe of uncleaned data from the database
    :return:        pandas Dataframe of cleaned data from analysis   """
    if len(data) != 0:
        data = clean_data_rounding(data)
        data = clean_duplicates(data)
        data = delete_faulty_inputs(data)
        data = location_outlier_check(data)
    else:
        print('No data was returned')
    return data


# Test handle
if __name__ == '__main__':
    import time
    data_input = pd.read_csv('Raw_data_best_ANCHORAGE.csv')

    start_time = time.time()
    data_1 = clean_data_rounding(data_input)
    time_data_1 = time.time()
    print('Time for clean_data_rounding:', time_data_1 - start_time, 'seconds')

    data_2 = clean_duplicates(data_1)
    time_data_2 = time.time()
    print('Time for clean_duplicates:', time_data_2 - time_data_1, 'seconds')

    data_3 = delete_faulty_inputs(data_2)
    time_data_3 = time.time()
    print('Time for delete_faulty_inputs:', time_data_3 - time_data_2, 'seconds')

    data_4 = location_outlier_check(data_3)
    time_data_4 = time.time()
    print('Time for location_outlier_check:', time_data_4 - time_data_3, 'seconds')

    print('The total time for data_cleaning.py:', time.time() - start_time, 'seconds')
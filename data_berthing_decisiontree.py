from data_cleaning import drop_and_report

# Number each MMSI set of when it is in the terminal , as new column (0 = not a track / not in polygon)
def number_set_mmsi(data):
    track_numbers = 0
    data['track_number'] = 0
    for row in data.itertuples():
        if row.Index != 0 and row.in_small_polygon == 'Yes' and data.at[row.Index - 1, 'in_small_polygon'] == 'No':
            track_numbers += 1
            data.at[row.Index, 'track_number'] = track_numbers
        elif row.Index != 0 and row.in_small_polygon == 'Yes' and data.at[row.Index - 1, 'mmsi'] == row.mmsi:
            data.at[row.Index, 'track_number'] = data.at[row.Index - 1, 'track_number']
    return data['track_number']


# Feature 1:
# Analyze the time since first in polygon (duration spread out)
def add_time_since_first(data):
    data['time_in_polygon'] = 0.0
    for row in data.itertuples():
        if row.Index != 0 and row.in_small_polygon == 'Yes' and data.at[row.Index - 1, 'in_small_polygon'] == 'Yes' \
                and row.mmsi == data.at[row.Index - 1, 'mmsi']:
            data.at[row.Index, 'time_in_polygon'] = \
                (row.timestamp - data.at[row.Index - 1, 'timestamp']).total_seconds() + \
                data.at[row.Index - 1, 'time_in_polygon']
    return data


# Feature 2: average speed in polygon [m/s]
def average_speed_small_polygon(data):
    list_means = []
    x = np.linspace(0, len(data.track_number.unique()) - 1, len(data.track_number.unique()))
    for i in x:
        list_means.append(data.loc[data.track_number == i].sog_ms.mean())

    for row in data.itertuples():
        if row.track_number == 0:
            data.at[row.Index, 'mean_sog_per_track'] = 0.0
        else:
            data.at[row.Index, 'mean_sog_per_track'] = list_means[row.track_number]
    return data['mean_sog_per_track']


# Feature 3: AIS message frequency [seconds]
def avg_timestamp_interval(data):
    data['avg_timestamp_interval'] = 0.0000
    list_freq = []
    x = np.linspace(0, len(data.track_number.unique()) - 1, len(data.track_number.unique()))

    for i in x:
        if i == 0:
            list_freq.append(0)
        elif data.loc[data.track_number == i].timestamp.count() > 1:
            list_freq.append(np.diff(data.loc[data.track_number == i].timestamp).mean() / np.timedelta64(1, 's'))
        else:
            list_freq.append(0)

    for row in data.itertuples():
        if row.track_number == 0:
            data.at[row.Index, 'avg_timestamp_interval'] = 0.0
        else:
            data.at[row.Index, 'avg_timestamp_interval'] = list_freq[row.track_number]

    return data['avg_timestamp_interval']


# Feature 4: total number of messages per track
def messages_tot(data):
    data['messages_tot'] = 0.0
    list_messages = []
    x = np.linspace(0, len(data.track_number.unique()) - 1, len(data.track_number.unique()))

    for i in x:
        list_messages.append(data.loc[data.track_number == i].timestamp.count())

    for row in data.itertuples():
        if row.track_number == 0:
            data.at[row.Index, 'messages_tot'] = 0.0
        else:
            data.at[row.Index, 'messages_tot'] = list_messages[row.track_number]

    return data['messages_tot']


# Feature 5: average distance between two successive points [m]
def avg_distance(data):
    data['distance'] = 0.0
    for row in data.itertuples():
        if row.Index != 0 and row.mmsi == data.at[row.Index - 1, 'mmsi']:
            data.at[row.Index, 'distance'] = geopy.distance.distance((data.at[row.Index - 1, 'lat'],
                                                                      data.at[row.Index - 1, 'lon']),
                                                                     (row.lat, row.lon)).m
    data['distance_avg'] = 0.00000
    list_distances = []
    x = np.linspace(0, len(data.track_number.unique()) - 1, len(data.track_number.unique()))

    for i in x:
        if i == 0:
            list_distances.append(0)
        elif data.loc[data.track_number == i].timestamp.count() > 1:
            list_distances.append(data.loc[data.track_number == i].distance.mean())
        else:
            list_distances.append(0)

    for row in data.itertuples():
        if row.track_number == 0:
            data.at[row.Index, 'distance_avg'] = 0.0
        else:
            data.at[row.Index, 'distance_avg'] = list_distances[row.track_number]

    return data['distance_avg']


# Feature 6: Average speed for 75% of the time
# zero for all single input tracks
def average_speed_smallest_75speed(data):
    list_75speed = []
    x = np.linspace(0, len(data.track_number.unique()) - 1, len(data.track_number.unique()))
    for i in x:
        list_75speed.append((data.loc[data.track_number == i]).iloc[(data.loc[data.track_number == i]).sog_ms.argsort()]
                              [:round(len(data.loc[data.track_number == i])*0.75)].sog_ms.mean())

    for row in data.itertuples():
        if row.track_number == 0:
            data.at[row.Index, 'mean_75_sog_per_track'] = 0.0
        else:
            data.at[row.Index, 'mean_75_sog_per_track'] = list_75speed[row.track_number]

    return data['mean_75_sog_per_track']


# Feature 7: standard deviation speed
def std_speed(data):
    data['std_sogms'] = 0.00000
    list_std = []
    x = np.linspace(0, len(data.track_number.unique()) - 1, len(data.track_number.unique()))
    for i in x:
        if data.loc[data.track_number == i].sog_ms.count() == 1:
            list_std.append(0)
        else:
            list_std.append(data.loc[data.track_number == i].sog_ms.std())

    for row in data.itertuples():
        if row.track_number == 0:
            data.at[row.Index, 'std_sogms'] = 0.0
        else:
            data.at[row.Index, 'std_sogms'] = list_std[row.track_number]

    return data['std_sogms']


# Feature 8: standard deviation distance difference
def std_distance(data):
    data['std_distance'] = 0.00000
    list_std_dist = []
    x = np.linspace(0, len(data.track_number.unique()) - 1, len(data.track_number.unique()))
    for i in x:
        if data.loc[data.track_number == i].distance.count() == 1:
            list_std_dist.append(0)
        else:
            list_std_dist.append(data.loc[data.track_number == i].distance.std())

    for row in data.itertuples():
        if row.track_number == 0:
            data.at[row.Index, 'std_distance'] = 0.0
        else:
            data.at[row.Index, 'std_distance'] = list_std_dist[row.track_number]

    return data['std_distance']


# Feature 9: message frequency [/hr]
def message_frequency(data):
    data['message_frequency'] = 0.0000
    for row in data.itertuples():
        if data.at[row.Index, 'time_in_polygon'] != 0:
            data.at[row.Index, 'message_frequency'] = (data.at[row.Index, 'messages_tot'] /
                                                      data.at[row.Index, 'time_in_polygon'])*3600
    return data['message_frequency']


# Merge data sets:
# 1. Delete vessel track 0
def remove_rows_trackzero(data):
    drop_list = list()
    for row in data.itertuples():
        if row.track_number == 0:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, "Remove 'track 0' ")
    return data


# 2. Keep last row of every vessel track
def keep_last_row(data):
    drop_list = list()
    for row in data.itertuples():
        if row.Index != len(data) - 1 and row.track_number == data.at[row.Index + 1, 'track_number']:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, "Keep last row of track")
    return data


# 3. Add column if berthed or not
def add_berthed_or_not(data, data_seaweb):
    # Replace all infinite values in messages frequency by zero
    # Merge two data frames (outerjoin)
    df_merged = pd.merge(
        left=data[['mmsi', 'timestamp', 'in_small_polygon', 'mean_sog_per_track', 'track_number', 'time_in_polygon',
                   'avg_timestamp_interval', 'messages_tot', 'distance_avg', 'mean_75_sog_per_track', 'std_sogms',
                   'std_distance', 'message_frequency']],
        right=data_seaweb[['Sailed Time', 'MMSI']],
        left_on='mmsi', right_on='MMSI', how='outer'
        )

    # Change to datetime value and different column name
    df_merged['Sailed_Time'] = pd.to_datetime(df_merged['Sailed Time'])
    df_merged.track_number = df_merged.track_number.fillna(0)

    # for every row, check if berthed (compared to Seaweb)
    # row sailed time cant be more than 6 hours different than row timestamp
    df_merged['berthed'] = 0
    # 0 is not berthed, 1 is berthed (compared to Seaweb)
    for row in df_merged.itertuples():
        if abs((row.timestamp - row.Sailed_Time).total_seconds()) < 3600 * 6:
            df_merged.at[row.Index, 'berthed'] = 1

    # Check for every track if it is berthed or not (1 = berthed)
    x = np.linspace(0, len(df_merged.track_number.unique())-1, len(df_merged.track_number.unique()))
    list_track_berthed = []
    for i in x:
        list_track_berthed.append(min(df_merged.loc[df_merged.track_number == i].berthed.sum(), 1))

    df_merged['track_berthed'] = 0.00
    for row in df_merged.itertuples():
        df_merged.at[row.Index, 'track_berthed'] = list_track_berthed[int(row.track_number)]

    return df_merged


# Run all steps
def processing_for_ML(data, data_seaweb):
    if len(data) != 0:
        data['timestamp'] = pd.to_datetime(data.timestamp, format='%Y-%m-%d %H:%M')

        # Add track number for each vessel track
        data['track_number'] = number_set_mmsi(data)

        # Define the features
        # Feature 1: duration in small polygon [s]
        # In the column 'time_in_polygon' the time in the polygon since first entry is visualised
        data['time_in_polygon'] = add_time_since_first(data)

        # Feature 2: add average speed per track [m/s]
        data['mean_sog_per_track'] = average_speed_small_polygon(data)

        # Feature 3: add average time interval [s]
        data['avg_timestamp_interval'] = avg_timestamp_interval(data)

        # Feature 4: total number of messages per track
        data['messages_tot'] = messages_tot(data)

        # Feature 5: average distance between two successive points
        data['distance_avg'] = avg_distance(data)

        # Feature 6: average speed of 75% of smallest speeds
        data['mean_75_sog_per_track'] = average_speed_smallest_75speed(data)

        # Feature 7: standard deviation speed
        data['std_sogms'] = std_speed(data)

        # Feature 8: standard deviation distance difference
        data['std_distance'] = std_distance(data)

        # Feature 9: message frequency [/hr]
        data['message_frequency'] = message_frequency(data)

        # Merge data sets
        # 1. Delete vessel track = 0
        data_AIS_1 = remove_rows_trackzero(data)
        # 2. Keep last row of every vessel track
        data_AIS_2 = keep_last_row(data_AIS_1)
        # 3. Merge data sets
        data_AIS_merged = add_berthed_or_not(data_AIS_2, data_seaweb)
        # 4. Again, delete rows with track = 0 and keep last row of vessel track
        data_AIS_merged_1 = remove_rows_trackzero(data_AIS_merged)
        data_AIS_merged_2 = keep_last_row(data_AIS_merged_1)
        data = data_AIS_merged_2
    else:
        print('No data was returned')

    return data


# Test handle
if __name__ == '__main__':
    from data_cleaning import clean_data_all
    import pandas as pd
    import geopy.distance
    import numpy as np
    from numpy import inf
    from data_berthed_vessels import add_time_since_first

    # Load data sets
    # Cleaned/ preprocessed dataset with column: in small_terminal
    data_AIS = pd.read_csv('Cleaned_dataset-UK_LBT-01072019–01022020_LARGEPOLY.csv')
    seaweb = pd.read_csv('Seaweb-dataset-UK-LBT.csv')

    # Apply processing steps
    data_processed = processing_for_ML(data_AIS, seaweb)

    # Export processed dataframes
    data_processed.to_csv('Data_forDT_UK_LBT-01072019-01022020.csv')







# Extracting unnecessary vessels such as tugs and pilot vessels
# By defining if a vessel is laying 'still' long enough
import pandas as pd
from data_cleaning import drop_and_report
import geopy.distance
import numpy as np
pd.options.mode.chained_assignment = None

""" Input parameters for berthing definitions """

""" Define the distance (between two successive points) limit for which the vessel is defined as lying still"""
distance_limit = 10.0  # m
""" Define the speed (between two successive points) limit for which the vessel is defined as lying still"""
speed_limit = 0.5 # m/s
""" Define the time limit for which the vessel is defined as lying still (seconds)"""
duration_limit = 3600  # 1 hr

"""" ....................... Don't change anything after this line ..................... """

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


# Delete vessel track 0
def remove_rows_trackzero(data):
    drop_list = list()
    for row in data.itertuples():
        if row.track_number == 0:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, "Remove 'track 0' ")
    return data


# Time since first in polygon (duration spread out)
def add_time_since_first(data):
    data['time_in_polygon'] = 0.0
    for row in data.itertuples():
        if row.Index != 0 and row.track_number == data.at[row.Index - 1, 'track_number']:
            data.at[row.Index, 'time_in_polygon'] = \
                (row.timestamp - data.at[row.Index - 1, 'timestamp']).total_seconds() + \
                data.at[row.Index - 1, 'time_in_polygon']
    return data['time_in_polygon']


# Delete time in polygon = 0
def remove_time_zero(data):
    drop_list = list()
    for row in data.itertuples():
        if row.time_in_polygon == 0:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, "Remove time in polygon = 0 ")
    return data


# Define distance between two rows
def distance_between_two_points(data):
    data['distance'] = 0.0
    for row in data.itertuples():
        if row.Index != 0 and row.mmsi == data.at[row.Index - 1, 'mmsi']:
            data.at[row.Index, 'distance'] = geopy.distance.distance((data.at[row.Index - 1, 'lat'],
                                                                      data.at[row.Index - 1, 'lon']),
                                                                     (row.lat, row.lon)).m
    return data['distance']


# Define vessel that lies still (per row)
def vessel_lie_still(data):
    # where 0 = doesn't lie still (false) and 1 = does lie still (true)
    data['lie_still'] = 0.0
    for row in data.itertuples():
        if row.distance < distance_limit and row.sog_ms < speed_limit:
            data.at[row.Index, 'lie_still'] = 1.0
    return data['lie_still']


# Determine total time lying still
def time_lie_still_first_last_total(data):
    data['time_lie_still_begin'] = 0.0
    data['time_lie_still_end'] = 0.0
    data['time_lie_still'] = 0.0

    # Define first moment of vessel lying still
    for row in data.itertuples():
        if row.lie_still == 1 and data.at[row.Index - 1, 'lie_still'] == 0 and \
                row.mmsi == data.at[row.Index - 1, 'mmsi']:
            data.at[row.Index, 'time_lie_still_begin'] = row.time_in_polygon

        elif row.lie_still == 1 and data.at[row.Index - 1, 'lie_still'] == 1 and \
                row.mmsi == data.at[row.Index - 1, 'mmsi']:
            data.at[row.Index, 'time_lie_still_begin'] = data.at[row.Index - 1, 'time_lie_still_begin']

    # Define last moment of vessel lying still
    for row in data.itertuples():
        if row.lie_still == 1 and data.at[row.Index + 1, 'lie_still'] == 0:
            data.at[row.Index, 'time_lie_still_end'] = row.time_in_polygon

    # Define total time of lying still
    for row in data.itertuples():
        if data.at[row.Index, 'time_lie_still_end'] != 0.0:
            data.at[row.Index, 'time_lie_still'] = data.at[row.Index, 'time_lie_still_end'] - \
                                                   data.at[row.Index, 'time_lie_still_begin']

    return data['time_lie_still']


# Add column for berthing vessels
def berthed_at_terminal(data):
    # where 0 = doesn't berth (false) and 1 = does berth (true) !
    data['berthed'] = 0.0
    for row in data.itertuples():
        if row.time_lie_still > duration_limit:
            data.at[row.Index, 'berthed'] = 1.0
    return data['berthed']


# Define a list of vessel tracks that berth
def tracks_berthed_vessels(data):
    berth_list = []
    for row in data.itertuples():
        if data.at[row.Index, 'berthed'] == 1.0:
            berth_list.append(row.track_number)
    berth_list_uniques = list(dict.fromkeys(berth_list))
    return berth_list_uniques


# If one row of a vessel track is berthed, all rows of the vessel track are berthed
def all_rows_berthed(data, berthlist):
    # Define a column, if a row with track number berths = 1
    data['track_berthed'] = 0
    x = np.linspace(0, len(data) - 1, len(data))
    for i in x:
        if data.track_number.isin(berth_list)[i]:
            data.track_berthed[i] = 1.0
    return data['track_berthed']


# Keep last row of every vessel track
def keep_last_row(data):
    drop_list = list()
    for row in data.itertuples():
        if row.Index != len(data) - 1 and row.track_number == data.at[row.Index + 1, 'track_number']:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, "Keep last row of track")
    return data


# Merge data set with seaweb
def add_berthed_or_not(data, data_seaweb):
    # Merge two data frames (outerjoin)
    df_merged = pd.merge(
        left=data[['mmsi', 'timestamp', 'in_small_polygon', 'track_number', 'time_in_polygon', 'track_berthed']],
        right=data_seaweb[['Sailed Time', 'MMSI']],
        left_on='mmsi', right_on='MMSI', how='outer'
        )

    # Change to datetime value and different column name
    df_merged['Sailed_Time'] = pd.to_datetime(df_merged['Sailed Time'])
    df_merged.track_number = df_merged.track_number.fillna(0)

    # for every row, check if berthed (compared to Seaweb)
    # row sailed time cant be more than 6 hours different than row timestamp
    df_merged['berthed_seaweb'] = 0
    # 0 is not berthed, 1 is berthed (compared to Seaweb)
    for row in df_merged.itertuples():
        if abs((row.timestamp - row.Sailed_Time).total_seconds()) < 3600 * 6:
            df_merged.at[row.Index, 'berthed_seaweb'] = 1

    # Check for every track if it is berthed or not (1 = berthed)
    x = np.linspace(0, len(df_merged.track_number.unique())-1, len(df_merged.track_number.unique()))
    list_track_berthed = []
    for i in x:
        list_track_berthed.append(min(df_merged.loc[df_merged.track_number == i].berthed_seaweb.sum(), 1))

    df_merged['track_berthed_seaweb'] = 0.00
    for row in df_merged.itertuples():
        df_merged.at[row.Index, 'track_berthed_seaweb'] = list_track_berthed[int(row.track_number)]

    return df_merged


# Make confusion matrix
def cm(data):
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    for row in data.itertuples():
        if row.track_berthed == row.track_berthed_seaweb and row.track_berthed_seaweb == 1:
            TP += 1
        elif row.track_berthed == row.track_berthed_seaweb and row.track_berthed_seaweb == 0:
            TN += 1
        elif row.track_berthed != row.track_berthed_seaweb and row.track_berthed_seaweb == 1:
            FN += 1
        elif row.track_berthed != row.track_berthed_seaweb and row.track_berthed_seaweb == 0:
            FP += 1

    cm = np.array([[TN,  FP], [FN,  TP]])
    # Plot confusion matrix
    from mlxtend.plotting import plot_confusion_matrix
    import matplotlib.pyplot as plt
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    plt.title('Confusion matrix')
    plt.show()
    print('Total accuracy is', np.round((TN+TP)/(TN+TP+FP+FN),5), '%')
    print('Percentage of correctly predicted berths (compared to total number of actual berths', np.round((TP)/(FN+TP),
                                                                                                          5), '%')
    print('False number berths compared to total berths', np.round(FP/(FN+TP),5))

# Test handle
if __name__ == '__main__':
    # Terminals
    df_SH = pd.read_csv('Cleaned_dataset_SH_15042019-01022020.csv')
    df_seaweb_SH = pd.read_csv('Seaweb_SH-15042019-01022020.csv')
    df_DB_lisbon = pd.read_csv('Cleaned_dataset-Lisbon-DB-01072019–01022020.csv')
    df_seaweb_DB_Lisbon = pd.read_csv('Seaweb-dataset-LisbonDB.csv')
    df_LB_Rdam = pd.read_csv('Cleaned_dataset-RdamLBT-01072019–01022020.csv')
    df_seaweb_rdam1 = pd.read_csv('Seaweb-dataset-Rdam-LBT_b1.csv')
    df_seaweb_rdam2 = pd.read_csv('Seaweb-dataset-Rdam-LBT_b2.csv')

    # Merge data sets
    df = pd.concat([df_SH, df_DB_lisbon, df_LB_Rdam], ignore_index=True)
    df_seaweb = pd.concat([df_seaweb_SH, df_seaweb_DB_Lisbon, df_seaweb_rdam1, df_seaweb_rdam2], ignore_index=True)

    df['timestamp'] = pd.to_datetime(df.timestamp, format='%Y-%m-%d %H:%M')

    # Add track number for each vessel track
    df['track_number'] = number_set_mmsi(df)

    # Delete vessel track = 0
    df_1 = remove_rows_trackzero(df)

    # Add time in polygon, per row for every track
    df_1['time_in_polygon'] = add_time_since_first(df_1)

    # Add distance between successive rows
    df_1['distance'] = distance_between_two_points(df_1)

    # Add if a row lies still or not
    df_1['lie_still'] = vessel_lie_still(df_1)

    # Define total continuous time vessel lying still
    df_1['time_lie_still'] = time_lie_still_first_last_total(df_1)

    # Define if a row is berthed or not
    df_1['berthed'] = berthed_at_terminal(df_1)

    # Define a list with track numbers that have berthed
    berth_list = tracks_berthed_vessels(df_1)

    # Define a column which returns if track is berthed
    df_1['track_berthed'] = all_rows_berthed(df_1, berth_list)

    # Keep last row of every vessel track
    df_2 = keep_last_row(df_1)

    # Merge data sets (track_berthed = own expection, track_berthed_seaweb = berthed based on seaweb)
    df_merged = add_berthed_or_not(df_2, df_seaweb)

    # Remove rows with time in polygon = 0
    df_merged_1 = remove_time_zero(df_merged)

    # Keep last row of every vessel track
    df_merged_2 = keep_last_row(df_merged_1)

    # Drop unnecessary columns
    df_final = df_merged_2.drop(['time_in_polygon', 'Sailed Time', 'Sailed_Time', 'MMSI', 'berthed_seaweb'], axis=1)

    # Compare to find how well function works by cm
    cm(df_final)


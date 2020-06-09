import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
import pickle
from xgboost import XGBClassifier
class MyXGBClassifier(XGBClassifier):
    @property
    def coef_(self):
        return None
pd.options.mode.chained_assignment = None
from _1_data_gathering import adjust_rhdhv_data, sort_data_rows, vessel_categories_CT, vessel_categories_DBT, \
    vessel_categories_LBT, drop_and_report, add_present_polygon_1, add_present_polygon_2, label_vessel_tracks, \
    keep_data_terminal, keep_data_anchorage, figure_google_maps_1, figure_google_maps_2
from _2_data_cleaning import clean_data_all, clean_data_rounding, clean_duplicates, delete_faulty_inputs
from _3_data_enrichment import process_data_all
from _4_data_transformation_1 import processing_for_ML_without_seaweb, number_set_mmsi
from _4_data_transformation_2 import remove_little_messages_tot, keep_tp


def data_gathering(data, terminal_type, anchorage_areas, visualise):
    # Adjust column names (edit function for specific data source, default: RHDHV data base)
    data = adjust_rhdhv_data(data)

    # Sort data rows by MMSI and timestamp (not necessary for data from RHDHV data base)
    data = sort_data_rows(data)

    # Keep only 1 terminal type
    if terminal_type == 'container':
        data = vessel_categories_CT(data)
    elif terminal_type == 'dry_bulk':
        data = vessel_categories_DBT(data)
    elif terminal_type == 'liquid_bulk':
        data = vessel_categories_LBT(data)
    else:
        print('No correct input for terminal type is given')

    # Create the polygons
    poly_term = Polygon([Coord1_term, Coord2_term, Coord3_term, Coord4_term])
    poly_port = Polygon([Coord1_port, Coord2_port, Coord3_port, Coord4_port])
    poly_anch1 = Polygon([Coord1_anch_1, Coord2_anch_1, Coord3_anch_1, Coord4_anch_1])
    poly_anch2 = Polygon([Coord1_anch_2, Coord2_anch_2, Coord3_anch_2, Coord4_anch_2])

    if anchorage_areas == 'one':
        data = add_present_polygon_1(data, poly_term, poly_anch1)
    elif anchorage_areas == 'two':
        data = add_present_polygon_2(data, poly_term, poly_anch1, poly_anch2)
    else:
        print('No correct input for number of anchorage areas is given')

    # Label each vessel track for terminal and anchorage area (0 = not a track/ not in polygon)
    data = label_vessel_tracks(data)

    # Split the data into two data sets: terminal and anchorage area(s)
    data_1 = data.copy()
    data_2 = data.copy()
    data_terminal = keep_data_terminal(data_1)
    data_anchorage = keep_data_anchorage(data_2)

    # Make a list of the latitude and longitude points
    lon_list_term = [Coord1_term[0], Coord2_term[0], Coord3_term[0], Coord4_term[0]]
    lat_list_term = [Coord1_term[1], Coord2_term[1], Coord3_term[1], Coord4_term[1]]

    lon_list_port = [Coord1_port[0], Coord2_port[0], Coord3_port[0], Coord4_port[0]]
    lat_list_port = [Coord1_port[1], Coord2_port[1], Coord3_port[1], Coord4_port[1]]

    lon_list_anch1 = [Coord1_anch_1[0], Coord2_anch_1[0], Coord3_anch_1[0], Coord4_anch_1[0]]
    lat_list_anch1 = [Coord1_anch_1[1], Coord2_anch_1[1], Coord3_anch_1[1], Coord4_anch_1[1]]

    lon_list_anch2 = [Coord1_anch_2[0], Coord2_anch_2[0], Coord3_anch_2[0], Coord4_anch_2[0]]
    lat_list_anch2 = [Coord1_anch_2[1], Coord2_anch_2[1], Coord3_anch_2[1], Coord4_anch_2[1]]

    # Visualise in google maps
    if visualise == 'yes' and anchorage_areas == 'one':
        figure_google_maps_1(data_terminal[0:1000], data_anchorage[0:1000], lon_list_term, lat_list_term,
            lat_list_port, lon_list_port, lon_list_anch1, lat_list_anch1)
    elif visualise == 'yes' and anchorage_areas == 'two':
        figure_google_maps_2(data_terminal[0:1000], data_anchorage[0:1000], lon_list_term, lat_list_term,
            lat_list_port, lon_list_port, lon_list_anch1, lat_list_anch1, lon_list_anch2, lat_list_anch2)
    else:
        print('Visualisation of data is not performed based on input')

    return data_terminal, data_anchorage, data


def keep_berthed_vessel_tracks(data, list_tracks):
    drop_list = list()
    # Define a column, if a row with track number is berthed or not (not berthed = 1)
    data['berthed_vessel_track'] = 1.0
    x = np.linspace(0, len(data) - 1, len(data))
    for i in x:
        if data.track_number.isin(list(list_tracks.track_number))[i]:
            data.berthed_vessel_track[i] = 0.0

    for row in data.itertuples():
        if row.berthed_vessel_track > 0:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, "Keep only berthed vessel tracks from OG data")

    return data


def extracting_vessel_tracks(df_features, classifier, df_terminal):
    # Remove all obvious non berthing vessel tracks (shorter time in polygon than 30 min)
    df_features = remove_little_messages_tot(df_features)

    # Select certain features
    feature_cols = ['avg_timestamp_interval', 'messages_tot', 'loa', 'mean_75_sog_per_track', 'DWT',
                    'message_frequency', 'distance_avg', 'mean_sog_per_track', 'std_sogms', 'std_distance',
                    'time_in_polygon', 'teu_capacity', 'std_location']
    X = df_features[feature_cols]  # Features

    # Fill all NaN with 0
    X = X.fillna(0)

    # Predict if vessel tracks are berthed or not
    # Predict if vessel tracks are berthed or not
    y_pred = classifier.predict(X)

    # Attach predictions to old data set
    y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
    df_all = (pd.concat([df_features, y_pred], axis=1))

    # Keep only the berthed vessel tracks
    df_tracks = keep_tp(df_all)

    # Add vessel track number to OG data
    df_terminal['track_number'] = number_set_mmsi(df_terminal)

    # For a list of all kept vessel tracks, filter the original data set
    df_final_full = keep_berthed_vessel_tracks(df_terminal, df_tracks)

    # For certain features, the featured data with 1 row per vessel track is sufficient
    df_final_per_track = df_tracks

    return df_final_full, df_final_per_track


# Clean port data (less extensive than for terminal cleaning)
def clean_port_data(data):
    data = clean_data_rounding(data)
    data = clean_duplicates(data)
    data = delete_faulty_inputs(data)
    return data


# Keep only data with MMSI numbers that berth at the required terminal
def keep_berthed_MMSI(data, df_terminal):
    # Add column where per row it is defined if a MMSI is also a berthed MMSI or not
    array_false_true = (data.mmsi.isin(list(df_terminal.mmsi)))
    data['false_true_mmsi'] = array_false_true

    # Keep only data frame with berthed MMSI numbers
    data_mmsi = data.loc[data.false_true_mmsi == True]

    return data_mmsi


# Label vessel tracks for entire port, for new MMSI or when same MMSI but message more than 12 hrs earlier
def label_vessel_tracks_port(data):
    data['timestamp'] = pd.to_datetime(data.timestamp, format='%Y-%m-%d %H:%M')
    track_numbers = 0
    data['port_track_number'] = 0
    for row in data.itertuples():
        if row.Index != 0 and ((data.at[row.Index - 1, 'mmsi'] != row.mmsi) or
                               (data.at[row.Index - 1, 'mmsi'] == row.mmsi and
                                (row.timestamp - data.at[row.Index - 1, 'timestamp']).total_seconds() > 12*3600)):
            track_numbers += 1
            data.at[row.Index, 'port_track_number'] = track_numbers
        elif row.Index != 0:
            data.at[row.Index, 'port_track_number'] = data.at[row.Index - 1, 'port_track_number']

    return data['port_track_number']


# Add entry and exit time for terminal polygon, per every port track
def entry_exit_terminal(df_port):
    list_tracks_term = df_port.track_number_term.unique()
    df_tracks_term = pd.DataFrame(list_tracks_term, columns=['term_track_number'])
    df_tracks_term['port_track_number'] = df_port.port_track_number.min()

    return df_tracks_term


# Make new data frame with entry and exit times for port
def new_data_frame(df_port, df_term_berthed_tracks):
    # Add for every port track number, if there is a terminal track number = new Data Frame
    port_tracks = df_port.port_track_number.unique()
    df_new = pd.DataFrame(columns=['term_track_number', 'port_track_number'])
    for i in port_tracks:
        data1 = (entry_exit_terminal(df_port.loc[df_port.port_track_number == i]))
        df_new = df_new.append(data1, ignore_index=True)

    df_new['port_entry_time'] = df_port.timestamp.min()
    df_new['port_exit_time'] = df_port.timestamp.max()
    df_new['terminal_entry_time'] = df_port.timestamp.min()
    df_new['terminal_exit_time'] = df_port.timestamp.max()
    df_new['anchorage_entry_time'] = pd.to_datetime(0, format='%Y-%m-%d %H:%M')
    df_new['anchorage_exit_time'] = pd.to_datetime(0, format='%Y-%m-%d %H:%M')

    drop_list = list()
    for row in df_new.itertuples():
        if len(((df_port.loc[df_port.track_number_term ==
                               row.term_track_number]).loc[df_port.in_terminal == 1]).index) == 0:
            # Delete this row: a port track that doesn't enter the terminal
            drop_list.append(row.Index)

    df_new = drop_and_report(df_new, drop_list, "Remove port tracks that don't enter the terminal polygon")

    for row in df_new.itertuples():
        # Add entry and exit time for port
        df_new.at[row.Index, 'port_entry_time'] = \
            df_port.loc[df_port.port_track_number == row.port_track_number].timestamp.min()
        df_new.at[row.Index, 'port_exit_time'] = \
            df_port.loc[df_port.port_track_number == row.port_track_number].timestamp.max()

        # Add entry and exit time for terminal
        df_new.at[row.Index, 'terminal_entry_time'] = \
            (df_port.loc[df_port.port_track_number == row.port_track_number]).loc[df_port.track_number_term ==
                                                                                  row.term_track_number].timestamp.min()
        df_new.at[row.Index, 'terminal_exit_time'] = \
            (df_port.loc[df_port.port_track_number == row.port_track_number]).loc[df_port.track_number_term ==
                                                                                  row.term_track_number].timestamp.max()

    # Only keep tracks, where the terminal track is one that has actually berthed
    df_new_merge = pd.merge(left=df_new, right=df_term_berthed_tracks, left_on='terminal_exit_time',
                            right_on='timestamp', how='inner')
    df_new = df_new_merge[
        ['term_track_number', 'port_track_number', 'mmsi', 'loa', 'port_entry_time', 'port_exit_time',
         'terminal_entry_time', 'terminal_exit_time', 'anchorage_entry_time', 'anchorage_exit_time']]

    for row in df_new.itertuples():
        # Add entry and exit time for anchorage
        # Only add entry and exit time if the port track goes through the anchorage area
        if df_port.loc[df_port.port_track_number == row.port_track_number].in_anchorage.sum() > 0:

            # If the port track has only 1 terminal track:
            if row.Index == 0 or row.Index == (len(df_new) - 1) or (row.Index != 0 and row.Index != (len(df_new) - 1)
                                                                    and row.term_track_number !=
                                                                    df_new.at[row.Index - 1, 'term_track_number'] and
                                                                    row.term_track_number !=
                                                                    df_new.at[row.Index + 1, 'term_track_number']):
                # First entry = first entry for port track in anchorage area
                # (check that this timestamp, is before current row entering terminal, otherwise = 0)
                df_t0_anch = df_port.loc[df_port.port_track_number == row.port_track_number].loc[df_port.in_anchorage
                                                                                                 == 1].timestamp.min()
                if df_t0_anch < df_new.at[row.Index, 'terminal_entry_time']:
                    df_new.at[row.Index, 'anchorage_entry_time'] = df_t0_anch

                    # Last entry = last entry in anchorage before first entry current terminal
                    df_new.at[row.Index, 'anchorage_exit_time'] = \
                        ((df_port.loc[df_port.port_track_number == row.port_track_number]
                        ).loc[df_port['timestamp'] < df_new.at[row.Index, 'terminal_entry_time']]
                        ).loc[df_port['in_anchorage'] == 1].timestamp.max()

            # If the port track has multiple terminal tracks:
            else:
                # First terminal track of port track
                if row.port_track_number != df_new.at[row.Index - 1, 'port_track_number']:
                    # First entry = first entry of port track in anchorage area
                    # (check that this timestamp, is before current row entering terminal, otherwise = 0)
                    df_t0_anch = df_port.loc[df_port.port_track_number == row.port_track_number].loc[
                        df_port.in_anchorage == 1].timestamp.min()
                    if df_t0_anch < df_new.at[row.Index, 'terminal_entry_time']:
                        df_new.at[row.Index, 'anchorage_entry_time'] = df_t0_anch

                        # Last entry = last entry anchorage before first entry current terminal
                        df_new.at[row.Index, 'anchorage_exit_time'] = \
                            ((df_port.loc[df_port.port_track_number == row.port_track_number]
                            ).loc[df_port['timestamp'] < df_new.at[row.Index, 'terminal_entry_time']]
                            ).loc[df_port['in_anchorage'] == 1].timestamp.max()

                # Last terminal track of port track
                elif row.port_track_number != df_new.at[row.Index + 1, 'port_track_number']:
                    # First entry = first entry anchorage area after previous last entry terminal area
                    # (check that this timestamp, is before current row entering terminal, otherwise use previous)
                    first_entry_after_last_entry_terminal = \
                        ((df_port.loc[df_port.port_track_number == row.port_track_number]
                        ).loc[df_port['timestamp'] > df_new.at[row.Index - 1, 'terminal_exit_time']]
                        ).loc[df_port['in_anchorage'] == 1].timestamp.min()
                    if first_entry_after_last_entry_terminal < row.terminal_entry_time:
                        df_new.at[row.Index, 'anchorage_entry_time'] = first_entry_after_last_entry_terminal
                    else:
                        df_new.at[row.Index, 'anchorage_entry_time'] = df_new.at[row.Index - 1, 'anchorage_entry_time']

                    # Last entry = last entry anchorage before first entry current terminal
                    df_new.at[row.Index, 'anchorage_exit_time'] = \
                        ((df_port.loc[df_port.port_track_number == row.port_track_number]
                        ).loc[df_port['timestamp'] < df_new.at[row.Index, 'terminal_entry_time']]
                        ).loc[df_port['in_anchorage'] == 1].timestamp.max()

                # All 'middle' terminal tracks
                else:
                    # First entry = first entry anchorage area after previous last entry terminal area
                    # (check that this timestamp, is before current row entering terminal, otherwise use previous)
                    first_entry_after_last_entry_terminal = \
                        ((df_port.loc[df_port.port_track_number == row.port_track_number]
                        ).loc[df_port['timestamp'] > df_new.at[row.Index - 1, 'terminal_exit_time']]
                        ).loc[df_port['in_anchorage'] == 1].timestamp.min()
                    if first_entry_after_last_entry_terminal < row.terminal_entry_time:
                        df_new.at[row.Index, 'anchorage_entry_time'] = first_entry_after_last_entry_terminal
                    else:
                        df_new.at[row.Index, 'anchorage_entry_time'] = df_new.at[row.Index - 1, 'anchorage_entry_time']
                    # Last entry = last entry anchorage before first entry current terminal
                    df_new.at[row.Index, 'anchorage_exit_time'] = \
                        ((df_port.loc[df_port.port_track_number == row.port_track_number]
                        ).loc[df_port['timestamp'] < df_new.at[row.Index, 'terminal_entry_time']]
                        ).loc[df_port['in_anchorage'] == 1].timestamp.max()

    return df_new


# RUN ALL
def run_all_steps(df, terminal_type, anchorage_areas, visualise, classifier):
    """" ............... DATA GATHERING ................  """
    df_terminal, df_anchorage, df_port = data_gathering(df, terminal_type, anchorage_areas, visualise)

    """" ............... DATA CLEANING (TERMINAL & PORT)  ................  """
    df_terminal = clean_data_all(df_terminal)
    df_port = clean_port_data(df_port)

    """" ............... DATA PROCESSING (TERMINAL) ................  """
    df_terminal = process_data_all(df_terminal)
    df_terminal_OG = df_terminal.copy()

    """" ............... ADDING FEATURES FOR EXTRACTING VESSEL TRACKS ................  """
    df_features = processing_for_ML_without_seaweb(df_terminal)

    """" ............... EXTRACTING (NOT BERTHED) VESSEL TRACKS ................  """
    df_term_final_full, df_term_final_per_track = extracting_vessel_tracks(df_features, classifier, df_terminal_OG)

    """ ......... FILTER PORT AND ANCHORAGE AREA, KEEP ONLY MMSI NUMBERS THAT BERTH """
    df_port_mmsi = keep_berthed_MMSI(df_port, df_term_final_per_track)
#   df_anch_mmsi = keep_berthed_MMSI(df_anchorage, df_term_final_per_track)

    """ .............. ADD VESSEL TRACK LABELLING FOR ENTIRE PORT ......... """
    df_port_mmsi1 = (df_port_mmsi.reset_index(drop=True)).copy()
    df_port_mmsi1['label_vessel_track'] = label_vessel_tracks_port(df_port_mmsi1)

    """ ............ RETURN NEW DATA FRAME WITH FIRST & LAST TIMESTAMPS ........ """
    df_new = new_data_frame(df_port_mmsi1, df_term_final_per_track)

    return df_new


# Test handle
if __name__ == '__main__':
    # Load raw data
    df = pd.read_csv('Data-frames/Datasets_phase_2/Container_terminals/Barcelona/Raw_data_best.csv')
    location = 'ct_BEST'

    """ For every new location, necessary inputs """
    # Choose vessel category based on terminal type (Options: 'container', 'dry_bulk', 'liquid_bulk')
    terminal_type = 'container'  # Input

    # Input latitude and longitude locations of terminal polygon. Every coordinate is a corner of the polygon,
    #    example coordX = (lon, lat)
    Coord1_term = (4.020561, 51.979217)
    Coord2_term = (4.046034, 51.973346)
    Coord3_term = (4.044223, 51.971183)
    Coord4_term = (4.019244, 51.976948)

    # Input latitude and longitude locations of port polygon.
    Coord1_port = (3.837200, 51.724568)
    Coord2_port = (4.2325, 51.9967)
    Coord3_port = (3.4722, 52.3689)
    Coord4_port = (3.1339, 51.8934)

    # Choose number of anchorage areas: one or two
    anchorage_areas = 'one'  # Input

    # Input latitude and longitude locations of anchorage polygon 1.
    Coord1_anch_1 = (3.4360, 52.1520)
    Coord2_anch_1 = (3.7326, 52.0457)
    Coord3_anch_1 = (3.8892, 52.1436)
    Coord4_anch_1 = (3.5678, 52.2497)

    # If a second polygon for waiting time is necessary:
    # Input latitude and longitude locations of anchorage polygon 2.
    Coord1_anch_2 = (3.3289, 51.9189)
    Coord2_anch_2 = (3.8068, 52.0305)
    Coord3_anch_2 = (3.9056, 51.9019)
    Coord4_anch_2 = (3.4579, 51.8171)

    # Visualise polygons and data in google maps: 'yes' or 'no'
    visualise = 'yes'  # Input

    # Load classifier to predicted berthed vessels tracks
    with open('classifier_pickle', 'rb') as f:
        classifier = pickle.load(f)
    # Run all steps
    df_new = run_all_steps(df, terminal_type, anchorage_areas, visualise, classifier)

    # Export data set
    df_new.to_csv('Data-frames/Final_df_' + location + '.csv')






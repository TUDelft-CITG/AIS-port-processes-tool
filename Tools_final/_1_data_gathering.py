""" Step 1. Data gathering
 Input: Raw data file from RHDHV AIS data base [csv file], coordinates for terminal/port/anchorage area's
 Actions: Sort data, adjust column names, extract certain categories,  extracting data in terminal polygon,
 vessel track labelling for terminal
 Output: visualisation of polygons with data using google maps, terminal data frame with vessel track labels
 """

# Loading the data
# For now: load data set from RHDHV AIS website (AWS server) [csv file]

# Loading and importing packages
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry import Point
import gmplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.ticker as mtick
from datetime import timedelta


def adjust_rhdhv_data(data):
    # Remove unnecessary columns
    data = data.copy()
    # data = data.drop(['displacement_tonnage'], axis=1)
    # If necessary, rename column name {"old_name":"new_name"}
    data.rename(columns={"mmsi": "mmsi", "timestamp": "timestamp", "latitude": "lat", "longitude": "lon",
                       "category_ais_platform": "type", "length_overall": "loa", "dead_weight_tonnage": "DWT",
                         # todo: als type en category op AWS website weer veranderd zijn, terugwisselen!
                       "teu_capacity": "teu_capacity", "breadth": "breadth"}, inplace=True)
    data['timestamp'] = pd.to_datetime(data.timestamp, format='%Y-%m-%d %H:%M')
    return data


# Sort the data set by MMSI number and timestamp (Already done by extracting data from AWS website)
def sort_data_rows(data):
    data.sort_values(by=['mmsi', 'timestamp'])
    return data


# Visualise category's left after category filtering
def category_visual(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    data.groupby(['type'])['mmsi'].nunique().plot.bar(ax=ax)
    plt.title('Number of vessels per vessel type')
    plt.xlabel('Vessel type')
    plt.ylabel('Number of vessels (unique MMSI)')
    plt.xticks(rotation=70)
    plt.show()

# Remove vessel categories (only keep necessary category)
# def vessel_categories_CT(data):
#     drop_list = list()
#     for row in data.itertuples():
#         if row.type != 'Container Vessels' and row.type != 'None':
#             drop_list.append(row.Index)
#     data = drop_and_report(data, drop_list, 'Keep only certain vessel categories (CT)')
#     return data

# New category filtering (july-2020)
def vessel_categories_CT(data):
    drop_list = list()
    for row in data.itertuples():
        if row.type != 'General Cargo' and row.type != 'Other Dry Cargo' and row.type != 'Passenger/General Cargo' and \
                row.type != 'Refrigerated Cargo' and row.type != 'Inland Waterways Dry Cargo / Passenger' and \
                row.type != 'Inland Waterways Others Non Seagoing' and row.type != 'Other Activities' and \
                row.type != 'Container' and row.type != 'Other Activities cont' and row.type != 'None':
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Keep only certain vessel categories (CT)')
    return data


# def vessel_categories_DBT(data):
#     drop_list = list()
#     for row in data.itertuples():
#         if row.type != ' Laker' and row.type != 'Cargo Vessels' and row.type != 'Dry Bulk Carriers'\
#                 and row.type != 'None':
#             drop_list.append(row.Index)
#     data = drop_and_report(data, drop_list, 'Keep only certain vessel categories (DBT)')
#     return data


# New category filtering (july-2020)
def vessel_categories_DBT(data):
    drop_list = list()
    for row in data.itertuples():
        if row.type != 'Bulk Dry' and row.type != 'Bulk Dry/Liquid' and row.type != 'Other Bulk Dry' and \
                row.type != 'Self Discharging Bulk Dry' and row.type != 'Passenger/General Cargo' and \
                row.type != 'Refrigerated Cargo' and row.type != 'Inland Waterways Dry Cargo / Passenger' and \
                row.type != 'Inland Waterways Others Non Seagoing' and row.type != 'Other Activities cont' and \
                row.type != 'Other Activities' and row.type != 'None':
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Keep only certain vessel categories (DBT)')
    return data


def vessel_categories_LBT(data):
    drop_list = list()
    for row in data.itertuples():
        # Old way of category's
        # if row.type != 'Tankers' and row.type != 'Cargo Vessels' and row.type != 'None':
        #     drop_list.append(row.Index)
        # New way of category's
        if row.type != 'Oil' and row.type != 'Inland Waterways Tanker' and row.type != 'Other Activities cont' \
            and row.type != 'Chemical' and row.type != 'Gas tankers' and row.type != 'None' and \
                row.type != 'Inland Waterways Others Non Seagoing' and row.type != 'Bulk Dry/Liquid' and \
                row.type != 'Other liquids' and row.type != 'Other Activities':
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, 'Keep only certain vessel categories (LBT)')
    return data


def drop_and_report(data, drop_list, filter_name):
    """ Drops a set of rows from a pandas Dataframe and reports on the amount of rows dropped
    :param data:        pandas Dataframe containing the data before dropping indices
    :param drop_list:   list of indices to be dropping
    :param filter_name: name of the filter step, used for documented changes
    :return:            pandas Dataframe with the passed rows removed
    """
    data.drop(drop_list, inplace=True)
    data.reset_index(drop=True, inplace=True)
    print(filter_name + " check has the following effect:")
    print("From a dataset of " + str(len(data) + len(drop_list)) + " it removed " + str(len(drop_list)) + " data points"
          + ", leaving a dataset of " + str(len(data)) + ".")
    return data


"""" Input latitude and longitude locations of terminal polygon. Every coordinate is a corner of the polygon,
    example coordX = (lon, lat) """
Coord1_term = (4.020561, 51.979217)
Coord2_term = (4.046034, 51.973346)
Coord3_term = (4.044223, 51.971183)
Coord4_term = (4.019244, 51.976948)


"""" Input latitude and longitude locations of port polygon. """
Coord1_port = (3.837200, 51.724568)
Coord2_port = (4.2325, 51.9967)
Coord3_port = (3.4722, 52.3689)
Coord4_port = (3.1339, 51.8934)


"""" Input latitude and longitude locations of anchorage polygon 1. """
Coord1_anch_1 = (3.4360, 52.1520)
Coord2_anch_1 = (3.7326, 52.0457)
Coord3_anch_1 = (3.8892, 52.1436)
Coord4_anch_1 = (3.5678, 52.2497)


"""" Input latitude and longitude locations of anchorage polygon 2. """
Coord1_anch_2 = (3.3289, 51.9189)
Coord2_anch_2 = (3.8068, 52.0305)
Coord3_anch_2 = (3.9056, 51.9019)
Coord4_anch_2 = (3.4579, 51.8171)


# Create the new polygon from the lat and lon list
poly_term = Polygon([Coord1_term, Coord2_term, Coord3_term, Coord4_term])

poly_port = Polygon([Coord1_port, Coord2_port, Coord3_port, Coord4_port])

poly_anch1 = Polygon([Coord1_anch_1, Coord2_anch_1, Coord3_anch_1, Coord4_anch_1])
poly_anch2 = Polygon([Coord1_anch_2, Coord2_anch_2, Coord3_anch_2, Coord4_anch_2])


# Add new column: whether location is inside the smaller terminal or not (yes=1, no=0)
def add_present_polygon_1(data, poly_term, poly_anch1):
    data['in_terminal'] = 0
    data['in_anchorage'] = 0
    for row in data.itertuples():
        # If coordinate from data_big is in polygon from small: return Yes
        if poly_term.contains(Point(row.lon, row.lat)) == bool(True):
            data.at[row.Index, 'in_terminal'] = 1
        elif poly_anch1.contains(Point(row.lon, row.lat)) == bool(True):
                # or poly_anch2.contains(Point(row.lon, row.lat)) == bool(True):
            data.at[row.Index, 'in_anchorage'] = 1

    return data


# Same code but for 2 anchorage area's
def add_present_polygon_2(data, poly_term, poly_anch1, poly_anch2):
    data['in_terminal'] = 0
    data['in_anchorage'] = 0
    for row in data.itertuples():
        # If coordinate from data_big is in polygon from small: return Yes
        if poly_term.contains(Point(row.lon, row.lat)) == bool(True):
            data.at[row.Index, 'in_terminal'] = 1
        elif poly_anch1.contains(Point(row.lon, row.lat)) == bool(True) \
                or poly_anch2.contains(Point(row.lon, row.lat)) == bool(True):
            data.at[row.Index, 'in_anchorage'] = 1

    return data


# Add new column: whether location is inside the terminal or not (yes=1, no=0)
def add_present_terminal(data, poly_term):
    data['in_terminal'] = 0
    for row in data.itertuples():
        if poly_term.contains(Point(row.lon, row.lat)) == bool(True):
            data.at[row.Index, 'in_terminal'] = 1
    return data


# Label each separate vessel track, for terminal and anchorage (0 = not a track / not in polygon)
def label_vessel_tracks(data):
    track_numbers_term = 0
    track_numbers_anch = 0
    data['track_number_term'] = 0
    data['track_number_anch'] = 0
    data.timestamp = pd.to_datetime(data.timestamp, format='%Y-%m-%d %H:%M')
    for row in data.itertuples():
        # A new track: if row before was not in terminal AND in the last 2 hours, for same MMSI, there has not been a
        # vessel track in terminal
        if row.Index != 0 and row.in_terminal == 1 and data.at[row.Index - 1, 'in_terminal'] == 0 and \
                data.loc[(data.timestamp < row.timestamp) & (data.timestamp > row.timestamp -
                                                             pd.Timedelta(2, unit='h'))].loc[
                    data.mmsi == row.mmsi].in_terminal.sum() == 0:
            track_numbers_term += 1
            data.at[row.Index, 'track_number_term'] = track_numbers_term
        elif row.Index != 0 and row.in_terminal == 1 and data.at[row.Index - 1, 'in_terminal'] == 1 and\
                data.at[row.Index - 1, 'mmsi'] == row.mmsi:
            data.at[row.Index, 'track_number_term'] = data.at[row.Index - 1, 'track_number_term']
        elif row.Index != 0 and row.in_terminal == 1 and data.at[row.Index - 1, 'in_terminal'] == 0:
            data.at[row.Index, 'track_number_term'] = data.loc[(data.timestamp < row.timestamp) &
                                                               (data.timestamp > row.timestamp -
                                                                pd.Timedelta(2, unit='h'))].track_number_term.max()

        elif row.Index != 0 and row.in_anchorage == 1 and data.at[row.Index - 1, 'in_anchorage'] == 0:
            track_numbers_anch += 1
            data.at[row.Index, 'track_number_anch'] = track_numbers_anch
        elif row.Index != 0 and row.in_anchorage == 1 and data.at[row.Index - 1, 'mmsi'] == row.mmsi:
            data.at[row.Index, 'track_number_anch'] = data.at[row.Index - 1, 'track_number_anch']

    return data


# Remove all data not in terminal polygon
def keep_data_terminal(data):
    drop_list = list()
    for row in data.itertuples():
        if row.track_number_term == 0:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, "Remove outside terminal ")
    return data


# Remove all data not in anchorage  polygons
def keep_data_anchorage(data):
    drop_list = list()
    for row in data.itertuples():
        if row.track_number_anch == 0:
            drop_list.append(row.Index)
    data = drop_and_report(data, drop_list, "Remove outside anchorage ")
    return data


# Make a polygon of input coordinates
# Make a list of the latitude and longitude points
lon_list_term = [Coord1_term[0], Coord2_term[0], Coord3_term[0], Coord4_term[0]]
lat_list_term = [Coord1_term[1], Coord2_term[1], Coord3_term[1], Coord4_term[1]]

lon_list_port = [Coord1_port[0], Coord2_port[0], Coord3_port[0], Coord4_port[0]]
lat_list_port = [Coord1_port[1], Coord2_port[1], Coord3_port[1], Coord4_port[1]]

lon_list_anch1 = [Coord1_anch_1[0], Coord2_anch_1[0], Coord3_anch_1[0], Coord4_anch_1[0]]
lat_list_anch1 = [Coord1_anch_1[1], Coord2_anch_1[1], Coord3_anch_1[1], Coord4_anch_1[1]]

lon_list_anch2= [Coord1_anch_2[0], Coord2_anch_2[0], Coord3_anch_2[0], Coord4_anch_2[0]]
lat_list_anch2 = [Coord1_anch_2[1], Coord2_anch_2[1], Coord3_anch_2[1], Coord4_anch_2[1]]


# Plot figure in google maps
def figure_google_maps_1(data_terminal, data_anchorage, lon_list_term, lat_list_term,
                         lat_list_port, lon_list_port, lon_list_anch1, lat_list_anch1):
    gmap = gmplot.GoogleMapPlotter(Coord1_term[1], Coord1_term[0], 13)
    gmap.scatter(data_terminal.lat, data_terminal.lon, 'orange', size=2, marker=False)
    gmap.scatter(data_anchorage.lat, data_anchorage.lon, 'orange', size=2, marker=False)
    # polygon method Draw a polygon with the help of coordinates
    gmap.polygon(lat_list_term, lon_list_term, color='grey')
    gmap.polygon(lat_list_port, lon_list_port, color='grey')
    gmap.polygon(lat_list_anch1, lon_list_anch1, color='grey')
    # gmap.polygon(lat_list_anch2, lon_list_anch2, color='grey')

    gmap.apikey = "AIzaSyBEwJIaiYm1Vd5GDbMOqDRh9zPYzz0hCaU"
    gmap.draw("C:\\Users\\909884\\Desktop\\visualise_location.html")


# Similar, for two anchorage area's
def figure_google_maps_2(data_terminal, data_anchorage, lon_list_term, lat_list_term,
                         lat_list_port, lon_list_port, lon_list_anch1, lat_list_anch1,
                         lon_list_anch2, lat_list_anch2):
    gmap = gmplot.GoogleMapPlotter(Coord1_term[1], Coord1_term[0], 13)
    gmap.scatter(data_terminal.lat, data_terminal.lon, 'orange', size=2, marker=False)
    gmap.scatter(data_anchorage.lat, data_anchorage.lon, 'orange', size=2, marker=False)
    # polygon method Draw a polygon with the help of coordinates
    gmap.polygon(lat_list_term, lon_list_term, color='grey')
    gmap.polygon(lat_list_port, lon_list_port, color='grey')
    gmap.polygon(lat_list_anch1, lon_list_anch1, color='grey')
    gmap.polygon(lat_list_anch2, lon_list_anch2, color='grey')

    gmap.apikey = "AIzaSyBEwJIaiYm1Vd5GDbMOqDRh9zPYzz0hCaU"
    gmap.draw("C:\\Users\\909884\\Desktop\\visualise_location.html")


# Test handle
if __name__ == '__main__':
    import time

    starttime = time.time()
    df_raw = pd.read_csv('Data-frames/Datasets_phase_2/Container_terminals/Rotterdam_Euromax/Raw_data_rdam_euromax_1.csv')

    # Change column names
    df_1 = adjust_rhdhv_data(df_raw)

    # Only keep certain types of vessels (Container Vessels, Dry Bulk Carriers, Tankers, None)
    df_1_a = vessel_categories_CT(df_1)

    # Add if present in terminal or anchorage area (1 = yes, 0 = no)
    df_2 = add_present_polygon_2(df_1_a, poly_term, poly_anch1, poly_anch2)

    # Label vessel tracks
    df_3 = label_vessel_tracks(df_2)
    df_4 = df_3.copy()

    # Data set with only data in terminal
    df_terminal = keep_data_terminal(df_3)
    df_terminal.to_csv('Raw_data_lb_belfast_TERMINAL.csv')

    # Data set with only data in anchorage
    df_anchorage = keep_data_anchorage(df_4)
    df_anchorage.to_csv('Raw_data_lb_belfast_ANCHORAGE.csv')

    # Plot data in google maps
    figure_google_maps_2(df_terminal[0:1000], df_anchorage[0:1000], lon_list_term, lat_list_term,
                         lat_list_port, lon_list_port, lon_list_anch1, lat_list_anch1,
                         lon_list_anch2, lat_list_anch2)

    print('Time for data_loading.py:', time.time() - starttime, 'seconds')

# Select a single terminal from the port data set
# Select the lat and lon coordinates of the polygon
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry import Point
import gmplot
import matplotlib.pyplot as plt
from data_cleaning import drop_and_report


# """" Input latitude and longitude locations of terminal polygon. Every coordinate is a corner of the polygon,
#     example coordX = (lon, lat) """
coord1 = (4.033257, 51.956685)
coord2 = (4.057206, 51.958485)
coord3 = (4.056691, 51.960019)
coord4 = (4.033087, 51.958406)


"""" Input latitude and longitude locations of larger polygon. Every coordinate is a corner of the polygon,
    example coordX = (lon, lat) """
coord1_large = (4.033257, 51.956685)
coord2_large = (4.057206, 51.958485)
coord3_large = (4.056433, 51.960998)
coord4_large = (4.032744, 51.959728)


""" ............................... Don't change anything after this line ................................. """
# Make a polygon of input coordinates
# Make a list of the latitude and longitude points
lon_list = [coord1[0], coord2[0], coord3[0], coord4[0]]
lat_list = [coord1[1], coord2[1], coord3[1], coord4[1]]

# Make a list of the latitude and longitude points
lon_list_large = [coord1_large[0], coord2_large[0], coord3_large[0], coord4_large[0]]
lat_list_large = [coord1_large[1], coord2_large[1], coord3_large[1], coord4_large[1]]

# Create the new polygon from the lat and lon list
poly = Polygon([coord1, coord2, coord3, coord4])

# Old polygon from lat and lon list:
poly_large = Polygon([coord1_large, coord2_large, coord3_large, coord4_large])

# Data visualisation of terminal selection
# Define for the visualisation of the polygon
x, y = poly.exterior.xy
x_large, y_large = poly_large.exterior.xy


# Extract data frame with only points outside the small terminal
def extract_data_polygon(data):
    drop_list = list()
    for row in data.itertuples():
        # If coordinate is not in the polygon, exclude data point
        if not poly.contains(Point(row.lon, row.lat)):
            drop_list.append(row.Index)
        # print(polygon_terminal.contains(Point(row.lon, row.lat)))
    data = drop_and_report(data, drop_list, 'Location outside extraction')

    return data


# Return all data with only points inside the small terminal
def data_removed(data):
    drop_list = list()
    for row in data.itertuples():
        # If coordinate is not in the polygon, exclude data point
        if poly.contains(Point(row.lon, row.lat)):
            drop_list.append(row.Index)
        # print(polygon_terminal.contains(Point(row.lon, row.lat)))
    data = drop_and_report(data, drop_list, 'Location inside extraction')

    return data


# Plot figure in google maps
def figure_google_maps(data, data_terminal):
    gmap = gmplot.GoogleMapPlotter(50.9, -1.4, 13)
    gmap.scatter(data.lat, data.lon, 'orange', size=2, marker=False)
    gmap.scatter(data_terminal.lat, data_terminal.lon, 'red', size=4, marker=False)
    # gmap.scatter(df_outside_terminal.lat, df_outside_terminal.lon, 'pink', size=5, marker=False)

    # polygon method Draw a polygon with the help of coordinates
    gmap.polygon(lat_list, lon_list, color='grey')
    gmap.polygon(lat_list_large, lon_list_large, color='grey')

    gmap.apikey = "AIzaSyBEwJIaiYm1Vd5GDbMOqDRh9zPYzz0hCaU"
    gmap.draw("C:\\Users\\909884\\Desktop\\map1.html")


# Print figures to check data outside and data inside
def figure_in_out_terminal_polygon(data_terminal, data_outside):
    # fig = plt.figure()
    plt.subplot(211)
    plt.title('Check terminal locations')
    plt.ylabel('Latitude')
    plt.scatter(data_terminal.lon, data_terminal.lat, c='orange', s=2)
    plt.plot(x, y, c='grey', label='Terminal Polygon')
    plt.plot(x_large, y_large, c='black', label='Large Polygon')
    plt.legend()

    plt.subplot(212)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.scatter(data_outside.lon, data_outside.lat, c='orange', label='Terminal only', s=2)
    plt.plot(x, y, c='grey', label='Terminal Polygon')
    plt.plot(x_large, y_large, c='black', label='Large Polygon')

    plt.show()


# Add new column: whether location is inside the smaller terminal or not
def add_if_present_in_small(data):
    data['in_small_polygon'] = 'Undefined'
    for row in data.itertuples():
        # If coordinate from data_big is in polygon from small: return Yes
        if poly.contains(Point(row.lon, row.lat)) == bool(True):
            isp = 'Yes'
        else:
            isp = 'No'

        data.at[row.Index, 'in_small_polygon'] = isp

    return data


def run_and_plot_all(data):
    df1 = data.copy()
    df2 = data.copy()
    df3 = data.copy()
    df_terminal_only = extract_data_polygon(df1)
    df_outside_terminal = data_removed(df2)
    data_google_maps = data[0:10000]
    data_terminal_google_maps = df_terminal_only[0:10000]
    figure_google_maps(data_google_maps, data_terminal_google_maps)
    figure_in_out_terminal_polygon(df_terminal_only, df_outside_terminal)
    add_if_present_in_small(df3)
    return df3


# Test handle
if __name__ == '__main__':
    import time
    from data_cleaning import clean_data_all
    from data_preprocessing import process_data_all

    data_read = pd.read_csv('Raw-dataset-APM-Rdam-CT-15042020-01022020.csv')
    data_input = clean_data_all(data_read)
    df = process_data_all(data_input)
    df_plot = run_and_plot_all(df)

    df1 = df.copy()
    df2 = df.copy()
    df3 = df.copy()

    start_time = time.time()
    data_1 = extract_data_polygon(df1)
    time_data_1 = time.time()
    print('Time for extract_data_polygon:', time_data_1 - start_time, 'seconds')

    data_2 = data_removed(df2)
    time_data_2 = time.time()
    print('Time for data_removed:', time_data_2 - time_data_1, 'seconds')

    data_3 = add_if_present_in_small(df3)
    time_data_3 = time.time()
    print('Time for add_if_present_in_small:', time_data_3 - time_data_2, 'seconds')

    print('The total time for data_select_terminal.py:', time.time() - start_time, 'seconds')

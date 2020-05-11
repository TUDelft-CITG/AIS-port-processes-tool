# Return list of vessel tracks which berthed
from sklearn.preprocessing import StandardScaler
import datetime


# Return a list of vessel tracks that berthed
def berthed_track_numbers(data):
    list_berthed_tracks = []
    for row in data.itertuples():
        if data.at[row.Index, 'berthed_predicted'] == 1:
            list_berthed_tracks.append(data.at[row.Index, 'track_number'])
    return list_berthed_tracks


# Add first arrival and last arrival timestamp
def new_data_middle_row(data):
    for row in data.itertuples():
        data.at[row.Index, 'timestamp_arrived'] = data.loc[data.track_number == row.track_number].timestamp.min()
    for row in data.itertuples():
        data.at[row.Index, 'timestamp_left'] = data.loc[data.track_number == row.track_number].timestamp.max()
    return data['timestamp_arrived'], data['timestamp_left']


# Find clustered center point for each track
def clustered_center_per_track(data, list_tracks):
    kmeans = KMeans(n_clusters=1, init='k-means++')
    # x = np.linspace(1, data.track_number.nunique(), data.track_number.nunique())
    data['lat_center_cluster'] = 0
    data['lon_center_cluster'] = 0
    for i in list_tracks:
        X = (data.loc[data.track_number == i]).loc[:, ['lat', 'lon']]
        kmeans.fit(X[X.columns[0:2]])
        centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
        centers_new = centers.flatten()
        data['lat_center_cluster'].loc[data.track_number == i] = centers_new[0]
        data['lon_center_cluster'].loc[data.track_number == i] = centers_new[1]

    return data['lat_center_cluster'], data['lon_center_cluster']


# Split timestamps
def split_timestamps(data):
    # Define starting and end point vessel
    t_start = data.timestamp_arrived.min()
    t_end = data.timestamp_left.max()
    # Make a list of all timestamps between begin and end (per hour)
    time = pd.date_range(t_start, t_end, freq='H')
    timestamps = [str(x) for x in time]
    timestamps = pd.to_datetime(timestamps, format='%Y-%m-%d %H:%M')
    return timestamps


# Count number of vessels present with input = certain timestamp
def vesselspresent_pertime(data, t):
    list_vesselspresent = 0
    for row in data.itertuples():
        if row.timestamp_arrived < t < row.timestamp_left:
            list_vesselspresent += 1
    return (t, list_vesselspresent)


# Make data frame with timestamps and number vessels present at certain time
def df_vessels_present(data):
    timestamps = split_timestamps(data)
    list_vessels = []
    for i in timestamps:
        list_vessels.append(vesselspresent_pertime(data, i)[1])
    data_time = pd.DataFrame({"timestamp": timestamps, 'vessels': 0})
    x = np.linspace(0, len(timestamps)-1, len(timestamps))
    for r in x:
        data_time['vessels'][int(r)] = list_vessels[int(r)]

    return data_time


def plot_figures_time(data):
    plt.figure(1)
    plt.xlabel('Time [per hour]')
    plt.ylabel('Number of vessels present at terminal')
    plt.title('Number of vessels present per hour')
    plt.plot(data.timestamp, data.vessels)
    plt.show()

    df_percentage = data.groupby('vessels').count()
    df_percentage['percentage_of_total_time'] = round((df_percentage.timestamp / data.vessels.count()) * 100, 2)
    plt.figure(2)
    sns.set(style="whitegrid")
    ax = sns.barplot(x=df_percentage.index, y="timestamp", data=df_percentage)
    ax.set(xlabel='Vessels present at certain hour', ylabel='Number of hours')
    plt.title('Number of vessels present per hour')
    plt.show()

    return df_percentage

# Test handle
if __name__ == '__main__':
    from Decisiontree import feature_cols, X, y
    from data_berthing_decisiontree import number_set_mmsi
    from data_cleaning import drop_and_report
    from sklearn.tree import DecisionTreeClassifier
    import pandas as pd
    import numpy as np
    import seaborn as sns
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from geopy.distance import great_circle
    from shapely.geometry import MultiPoint
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    import seaborn as sns;
    from sklearn.preprocessing import normalize
    from sklearn.decomposition import PCA
    from numpy import unique, where
    sns.set()
    pd.set_option('mode.chained_assignment', None)
    from sklearn.cluster import KMeans
    import gmplot

    # Load classifier
    classifier = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=0)
    classifier.fit(X, y)

    # Load data
    data_cleaned = pd.read_csv('Cleaned_dataset-Vliss-DB-01072019–01022020.csv')
    data_processed = pd.read_csv('Data_forDT_VlissDBT-01072019-01022020.csv')
    # Expected number of clusters
    cluster_numbers = 2

    # For new data, run model and return new data column with yes or no berthed
    X = data_processed[feature_cols]  # Features
    # New data frame
    new_df = data_processed
    new_df['berthed_predicted'] = classifier.predict(X)

    # List of vessel tracks that berthed
    list_tracks = berthed_track_numbers(new_df)

    # Return original data set, keep only tracks that berthed
    # From cleaned data set, add track numbers
    data_cleaned['track_number'] = number_set_mmsi(data_cleaned)

    # Remove all vessel tracks that did not berth (not in list_tracks)
    df_new = data_cleaned[data_cleaned['track_number'].isin(list_tracks)]

    # Add time arrived, add time left
    new_data_middle_row(df_new)
    df_new['timestamp_arrived'] = pd.to_datetime(df_new.timestamp_arrived, format='%Y-%m-%d %H:%M')
    df_new['timestamp_left'] = pd.to_datetime(df_new.timestamp_left, format='%Y-%m-%d %H:%M')
    # Add service time [s]
    df_new['service_time'] = (df_new.timestamp_left - df_new.timestamp_arrived) / np.timedelta64(1, 's')

    # Cluster tracks to find center of track
    df_new['lat_center_cluster'], df_new['lon_center_cluster'] = clustered_center_per_track(df_new, list_tracks)

    # Keep only one row per track
    df = df_new.drop_duplicates('track_number', keep='first')

    # Keep necessary columns
    df_2 = df[['mmsi', 'lat_center_cluster', 'lon_center_cluster', 'timestamp_arrived', 'timestamp_left',
               'service_time', 'track_number']]
    df_3 = df_2.copy()

    # Make new data frame per hour in time, number of vessels present
    df_vessels = df_vessels_present(df_3)

    # Visualise time in polygon
    # plot_figures_time(df_vessels)
    # df_percentage = plot_figures_time(df_vessels)

    """ Data clustering K-MEANS """
    X_df = ((df_2.loc[:,['lat_center_cluster', 'lon_center_cluster']]))
    X = X_df.to_numpy()
    Kmean = KMeans(n_clusters=cluster_numbers)
    Kmean.fit(X)
    # Define centers of clusters
    center_1 = Kmean.cluster_centers_[0].flatten()
    centers_lon = [center_1[1], center_1[1]]
    centers_lat = [center_1[0], center_1[0]]

    if cluster_numbers > 1:
        center_2 = Kmean.cluster_centers_[1].flatten()
        centers_lon = [center_1[1], center_2[1]]
        centers_lat = [center_1[0], center_2[0]]
        if cluster_numbers > 2:
            center_3 = Kmean.cluster_centers_[2].flatten()
            centers_lon = [center_1[1], center_2[1], center_3[1]]
            centers_lat = [center_1[0], center_2[0], center_3[0]]
            if cluster_numbers > 3:
                center_4 = Kmean.cluster_centers_[3].flatten()
                centers_lon = [center_1[1], center_2[1], center_3[1], center_4[1]]
                centers_lat = [center_1[0], center_2[0], center_3[0], center_4[0]]
                if cluster_numbers > 4:
                    center_5 = Kmean.cluster_centers_[4].flatten()
                    centers_lon = [center_1[1], center_2[1], center_3[1], center_4[1], center_5[1]]
                    centers_lat = [center_1[0], center_2[0], center_3[0], center_4[0], center_5[0]]


    # Plot clustering (single points) + k-means location centers
    # gmap = gmplot.GoogleMapPlotter(center_1[0], center_1[1], 13)
    # #gmap.scatter(list(X[:, 0]), list(X[:, 1]), s=10, c='orange', marker=False)
    # gmap.scatter(centers_lat, centers_lon, s=50, c='blue', marker=False)
    # gmap.apikey = "AIzaSyBEwJIaiYm1Vd5GDbMOqDRh9zPYzz0hCaU"
    # gmap.draw("C:\\Users\\909884\\Desktop\\map1_clustering.html")

    # Plot heatmap (single points) + k-means location centers
    #gmap_heat = gmplot.GoogleMapPlotter(center_1[0], center_1[1], 13)
    gmap_heat = gmplot.GoogleMapPlotter(50.9, -1.4, 13)
    gmap_heat.heatmap(X[:, 0], X[:, 1])
    gmap_heat.scatter(centers_lat, centers_lon, s=25, c='blue', marker=False)
    gmap_heat.draw("C:\\Users\\909884\\Desktop\\map1_clustering_heatmap.html")

    # Plot original heatmap (all points)
    # X_df_og = ((df_new.loc[:, ['lat', 'lon']])).to_numpy()
    # # Fit Kmean on total data
    # Kmean.fit(X_df_og)
    # # Define centers of clusters
    # center_1 = Kmean.cluster_centers_[0].flatten()
    # centers_lon = [center_1[1], center_1[1]]
    # centers_lat = [center_1[0], center_1[0]]
    #
    # if cluster_numbers > 1:
    #     center_2 = Kmean.cluster_centers_[1].flatten()
    #     centers_lon = [center_1[1], center_2[1]]
    #     centers_lat = [center_1[0], center_2[0]]
    #     if cluster_numbers > 2:
    #         center_3 = Kmean.cluster_centers_[2].flatten()
    #         centers_lon = [center_1[1], center_2[1], center_3[1]]
    #         centers_lat = [center_1[0], center_2[0], center_3[0]]
    #         if cluster_numbers > 3:
    #             center_4 = Kmean.cluster_centers_[3].flatten()
    #             centers_lon = [center_1[1], center_2[1], center_3[1], center_4[1]]
    #             centers_lat = [center_1[0], center_2[0], center_3[0], center_4[0]]
    #             if cluster_numbers > 4:
    #                 center_5 = Kmean.cluster_centers_[4].flatten()
    #                 centers_lon = [center_1[1], center_2[1], center_3[1], center_4[1], center_5[1]]
    #                 centers_lat = [center_1[0], center_2[0], center_3[0], center_4[0], center_5[0]]
    #
    # # #gmap_heat_og = gmplot.GoogleMapPlotter(center_1[0], center_1[1], 13)
    # #gmap_heat_og.heatmap(X_df_og[:, 0], X_df_og[:, 1])
    # gmap_heat_og.scatter(centers_lat, centers_lon, s=25, c='blue', marker=False)
    # gmap_heat_og.draw("C:\\Users\\909884\\Desktop\\map1_clustering_heatmap_og.html")

    # # Plot clustering (all points) + center
    # gmap_scatter_all = gmplot.GoogleMapPlotter(center_1[0], center_1[1], 13)
    # gmap_scatter_all.scatter(list(X_df_og[:, 0]), list(X_df_og[:, 1]), s=10, c='orange', marker=False)
    # # gmap_scatter_all.scatter(centers_lat, centers_lon, s=50, c='blue', marker=False)
    # gmap_scatter_all.apikey = "AIzaSyBEwJIaiYm1Vd5GDbMOqDRh9zPYzz0hCaU"
    # gmap_scatter_all.draw("C:\\Users\\909884\\Desktop\\map1_clustering_og.html")

    # Find optimum number of clusters (k)
    # Elbow method
    # Sum_of_squared_distances = []
    # K = range(1, 7)
    # for k in K:
    #     km = KMeans(n_clusters=k)
    #     km = km.fit(X_df_og)
    #     Sum_of_squared_distances.append(km.inertia_)
    # plt.figure(3)
    # plt.plot(K, Sum_of_squared_distances, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Sum_of_squared_distances')
    # plt.title('Elbow Method For Optimal k')

    # # Silhoutte method
    # from sklearn.metrics import silhouette_score
    # sil = []
    # kmax = 7
    # K = range(2, kmax + 1)
    # # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    # for k in K:
    #     kmeans = KMeans(n_clusters=k).fit(X_df)
    #     labels = kmeans.labels_
    #     sil.append(silhouette_score(X_df, labels, metric='euclidean'))
    # plt.figure(4)
    # plt.plot(K, sil, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Silhoutte score')
    # plt.title('Silhoutte method For Optimal k')
    # plt.show()







    # """ Data clustering DBSCAN   """
    # coords = (np.radians(df_2.loc[:, ['lat_center_cluster', 'lon_center_cluster']])).to_numpy()
    # # max distance (10 meter)
    # m_per_radian = 6371.008 * 1000
    # epsilon = 25. / m_per_radian
    # db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine')
    # X = db.fit_predict(coords)
    # # retrieve unique clusters
    # clusters = unique(X)
    # # create scatter plot for samples for each cluster
    # for cluster in clusters:
    #     # get row indexes for samples with this cluster
    #     row_ix = where(X == cluster)
    #     # create scatter of these samples
    #     plt.scatter(coords[row_ix, 0], coords[row_ix, 1])
    #     # show the plot
    # plt.show()
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_
    #
    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)
    #
    # print('Estimated number of clusters: %d' % n_clusters_)
    # print('Estimated number of noise points: %d' % n_noise_)

    # def dbscan(X, eps, min_samples):
    #     ss = StandardScaler()
    #     X = ss.fit_transform(X)
    #     db = DBSCAN(eps=eps, min_samples=min_samples)
    #     db.fit(X)
    #     y_pred = db.fit_predict(X)
    #     plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired')
    #     plt.title("DBSCAN")

    #dbscan(coords, epsilon, 5)

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(coords)
    # # cluster the data into five clusters
    # dbscan = DBSCAN(eps=0.123, min_samples=2)
    # clusters = dbscan.fit_predict(X_scaled)
    # # plot the cluster assignments
    # plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
    # plt.xlabel("Feature 0")
    # plt.ylabel("Feature 1")

    # cluster_labels = db.labels_
    # num_clusters = len(set(cluster_labels))
    # clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
    # print('Number of clusters: {}'.format(num_clusters))

    # # Cluster's center most point
    # def get_centermost_point(cluster):
    #     centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    #     centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    #     return tuple(centermost_point)
    #
    # centermost_points = clusters.map(get_centermost_point)
    #
    # lats, lons = zip(*centermost_points)
    # rep_points = pd.DataFrame({'lon': lons, 'lat': lats})
    #
    # rs = rep_points.apply(lambda row: df_new[(df_new['lat'] == row['lat']) &
    #                                          (df_new['lon'] == row['lon'])].iloc[0], axis=1)
    #
    # fig, ax = plt.subplots(figsize=[10, 6])
    # rs_scatter = ax.scatter(rs['lon'], rs['lat'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
    # df_scatter = ax.scatter(df_new['lon'], df_new['lat'], c='k', alpha=0.9, s=3)
    # ax.set_title('Full data set vs DBSCAN reduced set')
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # ax.legend([df_scatter, rs_scatter], ['Full set', 'Reduced set'], loc='upper right')
    # plt.show()

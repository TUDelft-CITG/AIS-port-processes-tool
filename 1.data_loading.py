# Loading the data
# For now: load data set from RHDHV AIS website (AWS server) [csv file]

# Loading and importing packages
import pandas as pd


def adjust_rhdhv_data(df):
    # Remove unnecessary columns
    df = df.copy()
    # If necessary, rename column name {"old_name":"new_name"}
    df.rename(columns={"mmsi": "mmsi", "timestamp": "timestamp", "latitude": "lat", "longitude": "lon"}, inplace=True)
    df['timestamp'] = pd.to_datetime(df.timestamp, format='%Y-%m-%d %H:%M')
    return df


# Sort the data set by MMSI number and timestamp (Already done by extracting data from AWS website)
def sort_data_rows(data):
    data.sort_values(by=['mmsi', 'timestamp'])
    return data


# Test handle
if __name__ == '__main__':
    import time

    starttime = time.time()
    dataread = pd.read_csv('Raw-dataset_SH_largeterminal_0101-0201.csv')
    df_test = sort_data_rows(adjust_rhdhv_data(dataread))

    print('Time for data_loading.py:', time.time() - starttime, 'seconds')

""" Step 6. Attach all (study parameter) columns
 Input: New data frame with entry and exit timestamps for the port area, anchorage area, terminal area
 Actions: Attach waiting times, service times, waiting times/service time ratio, sort by port entry, inter arrival
 times, sort by port entry (relative to  first moment in time)
 Output: Data frame with columns with waiting-, service- and inter arrival times
 """

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from _1_data_gathering import drop_and_report


# Remove all vessel tracks with time in polygon < 30 min:
def remove_little_messages_tot_second(data):
    drop_list = list()
    for row in data.itertuples():
        if data.at[row.Index, 'service_time[hr]'] < 0.5:
            drop_list.append(row.Index)
            continue
    data = drop_and_report(data, drop_list, 'Remove obvious non-berthing tracks (from new data frame)')
    return data


# Add service and waiting times
def service_waiting_times(df):
    df['port_entry_time'] = pd.to_datetime(df.port_entry_time, format='%Y-%m-%d %H:%M')
    df['port_exit_time'] = pd.to_datetime(df.port_exit_time, format='%Y-%m-%d %H:%M')
    df['terminal_entry_time'] = pd.to_datetime(df.terminal_entry_time, format='%Y-%m-%d %H:%M')
    df['terminal_exit_time'] = pd.to_datetime(df.terminal_exit_time, format='%Y-%m-%d %H:%M')
    df['anchorage_entry_time'] = pd.to_datetime(df.anchorage_entry_time, format='%Y-%m-%d %H:%M')
    df['anchorage_exit_time'] = pd.to_datetime(df.anchorage_exit_time, format='%Y-%m-%d %H:%M')

    # Add service times
    df['service_time[hr]'] = (df.terminal_exit_time - df.terminal_entry_time).astype('timedelta64[s]') / 3600.

    # Remove obvious non - berthing tracks that re-entered in new data frame (service time < 30 min)
    df = remove_little_messages_tot_second(df)

    # Add waiting times
    df['waiting_time[hr]'] = (df.anchorage_exit_time - df.anchorage_entry_time).astype('timedelta64[s]') / 3600.

    # Determine factor: waiting time in terms of service time
    df['waiting/service_time[%]'] = df['waiting_time[hr]'] * 100 / df['service_time[hr]']

    # Replace all NaN by zero
    df = df.fillna(0)

    return df


# Add inter arrival time, based on sorting by time entering port
def sort_by_port_entry(df):
    """ Based on port time arrivals"""
    df_p = df.copy()
    df_p = df_p.drop(columns=['Unnamed: 0', 'term_track_number', 'port_track_number'])
    df_p = df_p.sort_values(by=['port_entry_time'])
    df_p = df_p.reset_index(drop=True)

    # Add inter arrival time (based on port entries)
    df_p['inter_arrival_time_port[hr]'] = 0
    for row in df_p.itertuples():
        if row.Index != 0:
            df_p.at[row.Index, 'inter_arrival_time_port[hr]'] = (row.port_entry_time -
                                                                 df_p.at[row.Index - 1, 'port_entry_time']
                                                                 ).total_seconds() / 3600.
    return df_p


# Add inter arrival time, based on sorting by time entering port, relative to t0
def sort_by_port_entry_rel(df):
    """ Based on port time arrivals"""
    df_p = df.copy()
    # Normalise timestamps (first entry port = 0)
    t0 = df_p.port_entry_time.min()

    df_p.port_entry_time = (df_p.port_entry_time - t0).astype('timedelta64[s]') / 3600.
    df_p.port_exit_time = (df_p.port_exit_time - t0).astype('timedelta64[s]') / 3600.
    df_p.terminal_entry_time = (df_p.terminal_entry_time - t0).astype('timedelta64[s]') / 3600.
    df_p.terminal_exit_time = (df_p.terminal_exit_time - t0).astype('timedelta64[s]') / 3600.
    df_p.anchorage_entry_time = (df_p.anchorage_entry_time - t0).astype('timedelta64[s]') / 3600.
    df_p.anchorage_exit_time = (df_p.anchorage_exit_time - t0).astype('timedelta64[s]') / 3600.

    for row in df_p.itertuples():
        if row.anchorage_exit_time < 0:
            df_p.at[row.Index, 'anchorage_entry_time'] = 0
            df_p.at[row.Index, 'anchorage_exit_time'] = 0

    return df_p


# Test handle
if __name__ == '__main__':
    # Load raw data
    location = 'ct_rdam_euromax'
    df = pd.read_csv('Data-frames/Results_phase_2/' + location + '/Final_df_' + location + '.csv')

    # Add service, waiting times and WT/ST ratio
    df = service_waiting_times(df)

    # Based on port entry arrival time, return inter arrival time
    df_p = sort_by_port_entry(df)
    df_p.to_csv('Data-frames/Results_phase_2/' + location + '/Df_stats_' + location + '.csv')

    # Based on port entry arrival time, return all timestamps relative to t0
    df_p_rel = sort_by_port_entry_rel(df_p)


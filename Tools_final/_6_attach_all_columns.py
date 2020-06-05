import pandas as pd
pd.options.mode.chained_assignment = None


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

    # Add waiting times
    df['waiting_time[hr]'] = (df.anchorage_exit_time - df.anchorage_entry_time).astype('timedelta64[s]') / 3600.

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


# Add inter arrival time, based on sorting by time entering terminal
def sort_by_terminal_entry(df):
    """ Based on terminal time arrivals"""
    df_p = df.copy()
    df_p = df_p.drop(columns=['Unnamed: 0', 'term_track_number', 'port_track_number'])
    df_p = df_p.sort_values(by=['terminal_entry_time'])
    df_p = df_p.reset_index(drop=True)

    # Add inter arrival time (based on port entries)
    df_p['inter_arrival_time_term[hr]'] = 0
    for row in df_p.itertuples():
        if row.Index != 0:
            df_p.at[row.Index, 'inter_arrival_time_term[hr]'] = (row.terminal_entry_time -
                                                                 df_p.at[row.Index - 1, 'terminal_entry_time']
                                                                 ).total_seconds() / 3600.
    return df_p


# Add inter arrival time, based on sorting by time entering terminal, relative to t0
def sort_by_terminal_entry_rel(df):
    """ Based on terminal time arrivals"""
    df_p = df.copy()
    # Normalise timestamps (first entry port = 0)
    t0 = df_p.terminal_entry_time.min()

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
    location = 'ct_BEST'
    df = pd.read_csv('Data-frames/Final_df_' + location + '.csv')

    # Add service and waiting times
    df = service_waiting_times(df)

    # Based on port entry arrival time, return inter arrival time
    df_p = sort_by_port_entry(df)
    df_p.to_csv('Data-frames/New_df_p_' + location + '.csv', index=False)

    # Based on port entry arrival time, return all timestamps relative to t0
    df_p_rel = sort_by_port_entry_rel(df_p)

    # Based on terminal entry arrival, return inter arrival time
    df_t = sort_by_terminal_entry(df)
    df_t.to_csv('Data-frames/New_df_t_' + location + '.csv', index=False)

    # Based on terminal entry arrival, return inter arrival time, relative to t0
    df_t_rel = sort_by_terminal_entry_rel(df_t)


""" Extra visualisations
Based on arrivals and average service times, per vessel class or loa category, per month
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter


# Add vessel class if teu and loa are known (RHDHV Classification)
def add_vessel_class(df):
    df['vessel_class'] = 0
    for row in df.itertuples():
        if row.teu_capacity < 1000: # Small feeder
            df.at[row.Index, 'vessel_class'] = 1
        elif (row.teu_capacity >= 1000) and (row.teu_capacity < 2000):  # Regional feeder
            df.at[row.Index, 'vessel_class'] = 2
        elif (row.teu_capacity >= 2000) and (row.teu_capacity < 3000):  # Feeder max
            df.at[row.Index, 'vessel_class'] = 3
        elif (row.teu_capacity >= 3000) and (row.loa < 295):  # Panamax
            df.at[row.Index, 'vessel_class'] = 4
        elif (row.teu_capacity >= 3000) and (row.loa >= 295) and (row.loa < 366):  # New Panamax
            df.at[row.Index, 'vessel_class'] = 5
        elif (row.teu_capacity >= 3000) and (row.loa >= 366):  # New Panamax
            df.at[row.Index, 'vessel_class'] = 6
    return df['vessel_class']


# Add length classification
def add_loa_class(df):
    df['length_class'] = 0
    for row in df.itertuples():
        if row.loa < 100:
            df.at[row.Index, 'length_class'] = 1
        elif (row.loa >= 100) and (row.loa < 150):
            df.at[row.Index, 'length_class'] = 2
        elif (row.loa >= 150) and (row.loa < 200):
            df.at[row.Index, 'length_class'] = 3
        elif (row.loa >= 200) and (row.loa < 250):
            df.at[row.Index, 'length_class'] = 4
        elif row.loa >= 250:
            df.at[row.Index, 'length_class'] = 5
    return df['length_class']


# Add months-year column (based on terminal_entry_time)
def year_month(df):
    df.terminal_entry_time = pd.to_datetime(df.terminal_entry_time, format='%Y-%m-%d %H:%M')
    df['month'] = 0
    df['year'] = 0
    for row in df.itertuples():
        df.at[row.Index, 'month'] = row.terminal_entry_time.month
        df.at[row.Index, 'year'] = row.terminal_entry_time.year

    df['year_month'] = (df.year).astype(str) + '-' + (df.month).astype(str)

    return df['year_month']


# Plot vessel arrivals (using vessel class)
def plot_vessel_arrivals_class(df, title_name):
    df['year_month'] = pd.to_datetime(df.year_month, format='%Y-%m')

    # # Plot vessel arrivals per vessel class
    # fig, ax = plt.subplots(figsize=(15, 7))
    # df.groupby(['vessel_class']).count()['terminal_entry_time'].plot.bar(ax=ax)
    # plt.xticks(rotation=0)
    # plt.title('Vessel arrivals per vessel class: ' + title_name)
    # plt.xlabel('Vessel Class')
    # plt.ylabel('Number of vessel arrivals')
    # plt.show()

    # Plot vessel arrivals per month
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month']).count()['terminal_entry_time'].plot.bar(ax=ax)
    plt.xticks(rotation=00)
    plt.title('Vessel arrivals per month: ' + title_name)
    plt.xlabel('Month')
    positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
              '3-2020', '4-2020')
    plt.xticks(positions, labels)
    plt.ylabel('Number of vessel arrivals')
    plt.show()

    # # Plot vessel arrivals over time per class (bar chart)
    # fig, ax = plt.subplots(figsize=(15, 7))
    # df.groupby(['year_month', 'vessel_class']).count()['terminal_entry_time'].unstack().plot.bar(ax=ax)
    # handles, _ = ax.get_legend_handles_labels()
    # ax.legend(handles, ["1: Small Feeder", "2: Regional Feeder", '3: Feedermax', '4: Panamax', '5: New Panamax',
    #                     '6: Post New Panamax'], title='Vessel Class')
    # plt.title('Vessel arrivals per month per vessel class: ' + title_name)
    # plt.xticks(rotation=0)
    # plt.xlabel('Month')
    # plt.ylabel('Number of vessel arrivals')
    # positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    # labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
    #           '3-2020', '4-2020')
    # plt.xticks(positions, labels)
    # plt.show()

    # # Plot vessel arrivals over time per class (line graph)
    # fig, ax = plt.subplots(figsize=(15, 7))
    # df.groupby(['year_month', 'vessel_class']).count()['terminal_entry_time'].unstack().plot(ax=ax)
    # handles, _ = ax.get_legend_handles_labels()
    # ax.legend(handles, ["1: Small Feeder", "2: Regional Feeder", '3: Feedermax', '4: Panamax', '5: New Panamax',
    #                     '6: Post New Panamax'], title='Vessel Class')
    # plt.title('Vessel arrivals per month per vessel class: ' + title_name)
    # plt.xlabel('Month')
    # plt.ylabel('Number of vessel arrivals')
    # plt.show()


# Plot vessel arrivals (using vessel length)
def plot_vessel_arrivals_loa(df, title_name):
    df['year_month'] = pd.to_datetime(df.year_month, format='%Y-%m')

    # Plot vessel arrivals per vessel class
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['length_class']).count()['terminal_entry_time'].plot.bar(ax=ax)
    plt.xticks(rotation=0)
    plt.title('Vessel arrivals per length category: ' + title_name)
    plt.xlabel('Length category')
    plt.ylabel('Number of vessel arrivals')
    plt.show()

    # Plot vessel arrivals per month
    fig, ax = plt.subplots(figsize=(15,7))
    df.groupby(['year_month']).count()['terminal_entry_time'].plot.bar(ax=ax)
    plt.xticks(rotation=70)
    plt.title('Vessel arrivals per month: ' + title_name)
    plt.xlabel('Month')
    plt.ylabel('Number of vessel arrivals')
    plt.show()

    # Plot vessel arrivals over time per class (bar chart)
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month', 'length_class']).count()['terminal_entry_time'].unstack().plot.bar(ax=ax)
    plt.title('Vessel arrivals per month per length category: ' + title_name)
    plt.xlabel('Month')
    plt.ylabel('Number of vessel arrivals')
    plt.xticks(rotation=70)
    # ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    plt.show()

    # Plot vessel arrivals over time per class (line graph)
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month', 'length_class']).count()['terminal_entry_time'].unstack().plot(ax=ax)
    plt.title('Vessel arrivals per month per length category: ' + title_name)
    plt.xlabel('Month')
    plt.ylabel('Number of vessel arrivals')
    plt.show()


# Plot service time (using vessel class)
def plot_service_time_class(df, title_name):
    # Plot average service time per month
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month']).mean()['service_time[hr]'].plot.bar(ax=ax)
    plt.title('Average service time [hr] per month: ' + title_name)
    plt.xlabel('Month')
    positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
              '3-2020', '4-2020')
    plt.xticks(positions, labels)
    plt.ylabel('Average service time [hr]')
    plt.xticks(rotation=0)
    plt.show()

    # Plot average service time per month over time (bar chart)
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month', 'vessel_class']).mean()['service_time[hr]'].unstack().plot.bar(ax=ax)
    plt.title('Average service time [hr] per month per vessel class: ' + title_name)
    plt.xlabel('Month')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["1: Small Feeder", "2: Regional Feeder", '3: Feedermax', '4: Panamax', '5: New Panamax',
                         '6: Post New Panamax'], title='Vessel Class')
    positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
              '3-2020', '4-2020')
    plt.xticks(positions, labels)
    plt.ylabel('Average service time [hr]')
    plt.xticks(rotation=0)
    plt.show()

    # Plot average service time per month over time (line graph)
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month', 'vessel_class']).mean()['service_time[hr]'].unstack().plot(ax=ax)
    plt.title('Average service time [hr] per month per vessel class: ' + title_name)
    plt.xlabel('Month')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["1: Small Feeder", "2: Regional Feeder", '3: Feedermax', '4: Panamax', '5: New Panamax',
                         '6: Post New Panamax'], title='Vessel Class')
    plt.ylabel('Average service time [hr]')
    plt.show()


# Plot service time (using vessel length)
def plot_service_time_loa(df, title_name):
    # Plot average service time per month
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month']).mean()['service_time[hr]'].plot.bar(ax=ax)
    plt.title('Average service time [hr] per month: ' + title_name)
    plt.xlabel('Month')
    plt.ylabel('Average service time [hr]')
    plt.xticks(rotation=0)
    plt.show()

    # Plot average service time per month over time (bar chart)
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month', 'length_class']).mean()['service_time[hr]'].unstack().plot.bar(ax=ax)
    plt.title('Average service time [hr] per month per length category: ' + title_name)
    plt.xlabel('Month')
    plt.ylabel('Average service time [hr]')
    plt.xticks(rotation=0)
    plt.show()

    # Plot average service time per month over time (line graph)
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month', 'length_class']).mean()['service_time[hr]'].unstack().plot(ax=ax)
    plt.title('Average service time [hr] per month per length category: ' + title_name)
    plt.xlabel('Month')
    plt.ylabel('Average service time [hr]')
    plt.show()


# Plot extra visuals for container vessels
def plot_visuals_ct(df, location):
    # Add vessel classification
    df['vessel_class'] = add_vessel_class(df)

    # Add month-year classification
    df['year_month'] = year_month(df)

    # Plot variations on the vessel arrivals
  #  plot_vessel_arrivals_class(df, location)

    # Plot variations on the average service times
    plot_service_time_class(df, location)


# Plot extra visuals for dry bulk and liquid bulk terminals
def plot_visuals_loa(df, location):
    # Add length classification
    df['length_class'] = add_loa_class(df)

    # Add month-year classification
    df['year_month'] = year_month(df)

    # Plot variations on the vessel arrivals
    plot_vessel_arrivals_loa(df, location)

    # Plot variations on the average service times
    plot_service_time_loa(df, location)


if __name__ == '__main__':
    # Load data
    location = 'ct_rdam_apm2'
    df = pd.read_csv('Data-frames/Results_phase_2/' + location + '/Df_stats_' + location + '.csv')

    # Plot for containers
    plot_visuals_ct(df, location)

    # # Plot for dry bulk and liquid bulk
    # plot_visuals_loa(df, location)

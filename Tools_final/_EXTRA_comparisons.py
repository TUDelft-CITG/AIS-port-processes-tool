""" Extra comparisons
Comparisons in between every terminal type (container, dry bulk and liquid bulk): waiting time, service time, wt-st
ratio and length or berth occupancy
 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from _EXTRA_visuals import add_vessel_class, add_loa_class, year_month
import seaborn as sns


# Comparisons for container terminals
def comparisons_ct(location1, location2, location3, location4):
    # Load data
    df1 = pd.read_csv('Data-frames/Results_phase_2/' + location1 + '/Df_stats_' + location1 + '.csv')
    df2 = pd.read_csv('Data-frames/Results_phase_2/' + location2 + '/Df_stats_' + location2 + '.csv')
    df3 = pd.read_csv('Data-frames/Results_phase_2/' + location3 + '/Df_stats_' + location3 + '.csv')
    df4 = pd.read_csv('Data-frames/Results_phase_2/' + location4 + '/Df_stats_' + location4 + '.csv')

    # Add terminal name to data
    df1['Terminal_location'] = (location1 + ': N = ' + str(df1['waiting_time[hr]'].count()))
    df2['Terminal_location'] = (location2 + ': N = ' + str(df2['waiting_time[hr]'].count()))
    df3['Terminal_location'] = (location3 + ': N = ' + str(df3['waiting_time[hr]'].count()))
    df4['Terminal_location'] = (location4 + ': N = ' + str(df4['waiting_time[hr]'].count()))

    # Concat all data frames into one
    df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)

    # Attach vessel class
    df['vessel_class'] = add_vessel_class(df)

  #   # Plot the wt-st ratio
  #   plt.figure(figsize=(16, 8))
  #   sns.boxplot(x=df["Terminal_location"], y=df["waiting/service_time[%]"])
  #   plt.title('WT/ST ratio for different terminal locations')
  #   plt.xlabel('Terminal location (N = sample size)')
  # #  plt.ylim(0,100)
  #   plt.ylabel('WT/ST ratio [%]')
  #   plt.show()

    # # Plot the service time
    # plt.figure(figsize=(16, 8))
    # ax = sns.boxplot(x=df["Terminal_location"], y=df["service_time[hr]"], hue=df['vessel_class'])
    # handles, _ = ax.get_legend_handles_labels()
    # ax.legend(handles, ["1: Small Feeder", "2: Regional Feeder", '3: Feedermax', '4: Panamax', '5: New Panamax',
    #                     '6: Post New Panamax'], title='Vessel Class')
    # plt.title('Service time per vessel class, for different terminal locations')
    # plt.xlabel('Terminal location')
    # plt.ylabel('Service time [hr]')
    # plt.show()

    # # Plot the waiting time
    # plt.figure(figsize=(16, 8))
    # ax = sns.boxplot(x=df["Terminal_location"], y=df["waiting_time[hr]"], hue=df['vessel_class'])
    # handles, _ = ax.get_legend_handles_labels()
    # ax.legend(handles, ["1: Small Feeder", "2: Regional Feeder", '3: Feedermax', '4: Panamax', '5: New Panamax',
    #                     '6: Post New Panamax'], title='Vessel Class')
    # plt.title('Waiting time per vessel class, for different terminal locations')
    # plt.ylim(0, 50)
    # plt.xlabel('Terminal location')
    # plt.ylabel('Waiting time [hr]')
    # plt.show()

    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x=df["Terminal_location"], y=df["waiting_time[hr]"])
    # plt.title('Waiting time, for different terminal locations')
    # plt.xlabel('Terminal location')
    # plt.ylim(0, 25)
    # plt.ylabel('Waiting time [hr]')
    # plt.show()
    # #
    # # Plot the wt-st ratio
    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x=df["Terminal_location"], y=df["waiting/service_time[%]"])
    # plt.title('WT/ST ratio for different terminal locations')
    # plt.xlabel('Terminal location')
    # plt.ylim(0, 100)
    # plt.ylabel('WT/ST ratio [%]')
    # plt.show()
    # #
    # # Plot vessel arrivals per vessel class
    # fig, ax = plt.subplots(figsize=(15, 7))
    # df.groupby(['vessel_class', 'Terminal_location']).count()['terminal_entry_time'].unstack().plot.bar(ax=ax)
    # plt.xticks(rotation=0)
    # plt.title('Vessel arrivals per vessel class')
    # plt.xlabel('Vessel Class')
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[0] = '1: Small Feeder'
    # labels[1] = "2: Regional Feeder"
    # labels[2] = '3: Feedermax'
    # labels[3] = '4: Panamax'
    # labels[4] = '5: New Panamax'
    # labels[5] = '6: Post New Panamax'
    # ax.set_xticklabels(labels)
    # plt.ylabel('Number of vessel arrivals')
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(15, 7))
    # ((df.groupby(['vessel_class', 'Terminal_location']).count()['terminal_entry_time']).
    #  groupby(level=1).apply(lambda x: 100 * x / float(x.sum()))).unstack().plot.bar(ax=ax)
    # plt.xticks(rotation=0)
    # plt.title('Percentage vessel arrivals per vessel class')
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[0] = '1: Small Feeder'
    # labels[1] = "2: Regional Feeder"
    # labels[2] = '3: Feedermax'
    # labels[3] = '4: Panamax'
    # labels[4] = '5: New Panamax'
    # labels[5] = '6: Post New Panamax'
    # ax.set_xticklabels(labels)
    # plt.xlabel('Vessel Class')
    # plt.ylabel('Number of vessel arrivals compared to total arrivals [%]')
    # plt.ylim(0, 100)
    # plt.show()

    # Plot vessel arrivals per month
    df['year_month'] = year_month(df)
    df['year_month'] = pd.to_datetime(df.year_month, format='%Y-%m')

    # fig, ax = plt.subplots(figsize=(15, 7))
    # df.groupby(['year_month', 'Terminal_location']).count()['terminal_entry_time'].unstack().plot.bar(ax=ax)
    # plt.xticks(rotation=0)
    # plt.title('Vessel arrivals per month, for every terminal')
    # plt.xlabel('Month')
    # positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    # labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
    #           '3-2020', '4-2020')
    # plt.xticks(positions, labels)
    # plt.ylabel('Number of vessel arrivals')
    # plt.show()

    # fig, ax = plt.subplots(figsize=(15, 7))
    # ((df.groupby(['year_month', 'Terminal_location']).count()['terminal_entry_time']).
    #     groupby(level=1).apply(lambda x: 100 * x / float(x.sum()))).unstack().plot.bar(ax=ax)
    # plt.xticks(rotation=0)
    # plt.ylim(0, 20)
    # positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    # labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
    #           '3-2020', '4-2020')
    # plt.xticks(positions, labels)
    # plt.title('Percentage vessel arrivals per month, for every terminal')
    # plt.xlabel('Month')
    # plt.ylabel('Number of vessel arrivals compared to total arrivals, per terminal [%]')
    # plt.show()

    # Plot average service time per month
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month', 'Terminal_location']).mean()['service_time[hr]'].unstack().plot.bar(ax=ax)
    plt.xticks(rotation=0)
    positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
              '3-2020', '4-2020')
    plt.xticks(positions, labels)
    plt.title('Average service time per month, for every terminal')
    plt.xlabel('Month')
    plt.ylabel('Average service time [hr]')
    plt.show()

    # # Read all length occupancy data
    # df1 = pd.read_csv('Data-frames/Results_phase_2/' + location1 + '/Df_length_occup_' + location1 + '.csv')
    # df2 = pd.read_csv('Data-frames/Results_phase_2/' + location2 + '/Df_length_occup_' + location2 + '.csv')
    # df3 = pd.read_csv('Data-frames/Results_phase_2/' + location3 + '/Df_length_occup_' + location3 + '.csv')
    # df4 = pd.read_csv('Data-frames/Results_phase_2/' + location4 + '/Df_length_occup_' + location4 + '.csv')
    #
    # df1['Terminal_location'] = (location1 + ': N = ' + str(df1['occupancy_length_adjust[%]'].count()))
    # df2['Terminal_location'] = (location2 + ': N = ' + str(df2['occupancy_length_adjust[%]'].count()))
    # df3['Terminal_location'] = (location3 + ': N = ' + str(df3['occupancy_length_adjust[%]'].count()))
    # df4['Terminal_location'] = (location4 + ': N = ' + str(df4['occupancy_length_adjust[%]'].count()))
    #
    # # Concat all data frames into one
    # df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)
    #
    # # Plot the length occupancy
    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x=df["Terminal_location"], y=df["occupancy_length_adjust[%]"])
    # plt.title('Length occupancy for different terminal locations')
    # plt.xlabel('Terminal location')
    # plt.ylim(0, 100)
    # plt.ylabel('(Adjusted) length occupancy [%]')
    # plt.show()

    return df


# Comparisons for dry bulk terminals
def comparisons_db(location1, location2, location3, location4):
    # Load data
    df1 = pd.read_csv('Data-frames/Results_phase_2/' + location1 + '/Df_stats_' + location1 + '.csv')
    df2 = pd.read_csv('Data-frames/Results_phase_2/' + location2 + '/Df_stats_' + location2 + '.csv')
    df3 = pd.read_csv('Data-frames/Results_phase_2/' + location3 + '/Df_stats_' + location3 + '.csv')
    df4 = pd.read_csv('Data-frames/Results_phase_2/' + location4 + '/Df_stats_' + location4 + '.csv')

    # Add terminal name to data
    df1['Terminal_location'] = (location1 + ': N = ' + str(df1['waiting_time[hr]'].count()))
    df2['Terminal_location'] = (location2 + ': N = ' + str(df2['waiting_time[hr]'].count()))
    df3['Terminal_location'] = (location3 + ': N = ' + str(df3['waiting_time[hr]'].count()))
    df4['Terminal_location'] = (location4 + ': N = ' + str(df4['waiting_time[hr]'].count()))

    # Concat all data frames into one
    df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)

    # Attach vessel class
    df['loa_category'] = add_loa_class(df)

    # Plot the service time
    # plt.figure(figsize=(16, 8))
    # ax = sns.boxplot(x=df["Terminal_location"], y=df["service_time[hr]"], hue=df['loa_category'])
    # plt.title('Service time per loa category, for different terminal locations')
    # handles, _ = ax.get_legend_handles_labels()
    # ax.legend(handles, ["1: 0 - 99 [m]", "2: 100 - 149 [m]", '3: 150 - 199 [m]', '4: 200 - 249 [m]', '5: 250 - ... [m]']
    #                     ,title='LOA category')
    # plt.xlabel('Terminal location')
    # plt.ylabel('Service time [hr]')
    # plt.show()

    # # Plot the waiting time
    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x=df["Terminal_location"], y=df["waiting_time[hr]"], hue=df['loa_category'])
    # plt.title('Waiting time per vessel class, for different terminal locations')
    # plt.xlabel('Terminal location')
    # plt.ylabel('Waiting time [hr]')
    # plt.show()
    #
    # # Plot the wt-st ratio
    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x=df["Terminal_location"], y=df["waiting/service_time[%]"])
    # plt.title('WT/ST ratio for different terminal locations')
    # plt.xlabel('Terminal location')
    # plt.ylabel('WT/ST ratio [%]')
    # plt.ylim(0, 100)
    # plt.show()
    #
    # Plot vessel arrivals per month
    df['year_month'] = year_month(df)
    df['year_month'] = pd.to_datetime(df.year_month, format='%Y-%m')

    # fig, ax = plt.subplots(figsize=(15, 7))
    # df.groupby(['year_month', 'Terminal_location']).count()['terminal_entry_time'].unstack().plot.bar(ax=ax)
    # plt.xticks(rotation=0)
    # plt.title('Vessel arrivals per month, for every terminal')
    # plt.xlabel('Month')
    # positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    # labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
    #           '3-2020', '4-2020')
    # plt.xticks(positions, labels)
    # plt.ylabel('Number of vessel arrivals')
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(15, 7))
    # ((df.groupby(['year_month', 'Terminal_location']).count()['terminal_entry_time']).
    #     groupby(level=1).apply(lambda x: 100 * x / float(x.sum()))).unstack().plot.bar(ax=ax)
    # plt.xticks(rotation=0)
    # positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    # labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
    #           '3-2020', '4-2020')
    # plt.xticks(positions, labels)
    # plt.title('Percentage vessel arrivals per month, for every terminal')
    # plt.xlabel('Month')
    # plt.ylabel('Number of vessel arrivals compared to total arrivals, per terminal [%]')
    # plt.show()
    #
    # Plot average service time per month
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month', 'Terminal_location']).mean()['service_time[hr]'].unstack().plot.bar(ax=ax)
    plt.xticks(rotation=0)
    positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
              '3-2020', '4-2020')
    plt.xticks(positions, labels)
    plt.title('Average service time per month, for every terminal')
    plt.xlabel('Month')
    plt.ylabel('Average service time [hr]')
    plt.show()

    # # Read all length occupancy data
    # df1 = pd.read_csv('Data-frames/Results_phase_2/' + location1 + '/Df_length_occup_' + location1 + '.csv')
    # df2 = pd.read_csv('Data-frames/Results_phase_2/' + location2 + '/Df_length_occup_' + location2 + '.csv')
    # df3 = pd.read_csv('Data-frames/Results_phase_2/' + location3 + '/Df_length_occup_' + location3 + '.csv')
    # df4 = pd.read_csv('Data-frames/Results_phase_2/' + location4 + '/Df_length_occup_' + location4 + '.csv')
    #
    # df1['Terminal_location'] = (location1 + ': N = ' + str(df1['occupancy_length_adjust[%]'].count()))
    # df2['Terminal_location'] = (location2 + ': N = ' + str(df2['occupancy_length_adjust[%]'].count()))
    # df3['Terminal_location'] = (location3 + ': N = ' + str(df3['occupancy_length_adjust[%]'].count()))
    # df4['Terminal_location'] = (location4 + ': N = ' + str(df4['occupancy_length_adjust[%]'].count()))
    #
    # # Concat all data frames into one
    # df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)
    #
    # # Plot the length occupancy
    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x=df["Terminal_location"], y=df["occupancy_length_adjust[%]"])
    # plt.title('Length occupancy for different terminal locations')
    # plt.xlabel('Terminal location')
    # plt.ylim(0, 100)
    # plt.ylabel('(Adjusted) length occupancy [%]')
    # plt.show()

    return df


# Comparisons for liquid bulk terminals
def comparisons_lb(location1, location2, location3, location4):
    # Load data
    df1 = pd.read_csv('Data-frames/Results_phase_2/' + location1 + '/Df_stats_' + location1 + '.csv')
    df2 = pd.read_csv('Data-frames/Results_phase_2/' + location2 + '/Df_stats_' + location2 + '.csv')
    df3 = pd.read_csv('Data-frames/Results_phase_2/' + location3 + '/Df_stats_' + location3 + '.csv')
    df4 = pd.read_csv('Data-frames/Results_phase_2/' + location4 + '/Df_stats_' + location4 + '.csv')

    # Add terminal name to data
    df1['Terminal_location'] = (location1 + ': N = ' + str(df1['waiting_time[hr]'].count()))
    df2['Terminal_location'] = (location2 + ': N = ' + str(df2['waiting_time[hr]'].count()))
    df3['Terminal_location'] = (location3 + ': N = ' + str(df3['waiting_time[hr]'].count()))
    df4['Terminal_location'] = (location4 + ': N = ' + str(df4['waiting_time[hr]'].count()))

    # Concat all data frames into one
    df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)

    # Attach vessel class
    df['loa_category'] = add_loa_class(df)
    #
    # # Plot the service time
    # plt.figure(figsize=(16, 8))
    # ax = sns.boxplot(x=df["Terminal_location"], y=df["service_time[hr]"], hue=df['loa_category'])
    # plt.title('Service time per loa category, for different terminal locations')
    # plt.xlabel('Terminal location')
    # handles, _ = ax.get_legend_handles_labels()
    # ax.legend(handles, ["1: 0 - 99 [m]", "2: 100 - 149 [m]", '3: 150 - 199 [m]', '4: 200 - 249 [m]', '5: 250 - ... [m]']
    #                     ,title='LOA category')
    # plt.ylabel('Service time [hr]')
    # plt.show()
    #
    # # Plot the waiting time
    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x=df["Terminal_location"], y=df["waiting_time[hr]"], hue=df['loa_category'])
    # plt.title('Waiting time per loa category, for different terminal locations')
    # plt.xlabel('Terminal location')
    # plt.ylabel('Waiting time [hr]')
    # plt.show()
    #
    #     # Plot the wt-st ratio
    #     plt.figure(figsize=(16, 8))
    #     sns.boxplot(x=df["Terminal_location"], y=df["waiting/service_time[%]"])
    #     plt.title('WT/ST ratio for different terminal locations')
    #     plt.xlabel('Terminal location')
    #     plt.ylabel('WT/ST ratio [%]')
    #     plt.ylim(0, 200)
    #     plt.show()

    # Plot vessel arrivals per month
    df['year_month'] = year_month(df)
    df['year_month'] = pd.to_datetime(df.year_month, format='%Y-%m')

    # fig, ax = plt.subplots(figsize=(15, 7))
    # df.groupby(['year_month', 'Terminal_location']).count()['terminal_entry_time'].unstack().plot.bar(ax=ax)
    # plt.xticks(rotation=0)
    # positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    # labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
    #           '3-2020', '4-2020')
    # plt.xticks(positions, labels)
    # plt.title('Vessel arrivals per month, for every terminal')
    # plt.xlabel('Month')
    # plt.ylabel('Number of vessel arrivals')
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize=(15, 7))
    # ((df.groupby(['year_month', 'Terminal_location']).count()['terminal_entry_time']).
    #     groupby(level=1).apply(lambda x: 100 * x / float(x.sum()))).unstack().plot.bar(ax=ax)
    # plt.xticks(rotation=0)
    # positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    # labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
    #           '3-2020', '4-2020')
    # plt.xticks(positions, labels)
    # plt.title('Percentage vessel arrivals per month, for every terminal')
    # plt.xlabel('Month')
    # plt.ylabel('Number of vessel arrivals compared to total arrivals, per terminal [%]')
    # plt.show()

    # Plot average service time per month
    fig, ax = plt.subplots(figsize=(15, 7))
    df.groupby(['year_month', 'Terminal_location']).mean()['service_time[hr]'].unstack().plot.bar(ax=ax)
    plt.xticks(rotation=0)
    positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    labels = ("5-2019", "6-2019", "7-2019", '8-2019', '9-2019', '10-2019', '11-2019', '12-2019', '1-2020', '2-2020',
              '3-2020', '4-2020')
    plt.xticks(positions, labels)
    plt.title('Average service time per month, for every terminal')
    plt.xlabel('Month')
    plt.ylabel('Average service time [hr]')
    plt.show()

    # # Read all berth occupancy data
    # df1 = pd.read_csv('Data-frames/Results_phase_2/' + location1 + '/Df_berth_occup_' + location1 + '.csv')
    # df2 = pd.read_csv('Data-frames/Results_phase_2/' + location2 + '/Df_berth_occup_' + location2 + '.csv')
    # df3 = pd.read_csv('Data-frames/Results_phase_2/' + location3 + '/Df_berth_occup_' + location3 + '.csv')
    # df4 = pd.read_csv('Data-frames/Results_phase_2/' + location4 + '/Df_berth_occup_' + location4 + '.csv')
    #
    # df1['Terminal_location'] = location1
    # df2['Terminal_location'] = location2
    # df3['Terminal_location'] = location3
    # df4['Terminal_location'] = location4
    #
    # # Concat all data frames into one
    # df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)
    #
    # # Plot the terminal occupancy
    # plt.figure(figsize=(16, 8))
    # sns.boxplot(x=df["Terminal_location"], y=df["occupancy[%]"])
    # plt.title('Berth occupancy for different terminal locations')
    # plt.xlabel('Terminal location')
    # plt.ylim(0, 100)
    # plt.ylabel('Terminal occupancy [%]')
    # plt.show()

    return df


if __name__ == '__main__':
    # # Container terminals
    # ct_location1 = 'ct_rdam_euromax'
    # ct_location2 = 'ct_rdam_apm'
    # ct_location3 = 'ct_rdam_apm2'
    # ct_location4 = 'ct_lehavre_atlantic'
    #
    # # Plot comparisons for container terminals
    # df_ct = comparisons_ct(ct_location1, ct_location2, ct_location3, ct_location4)
    #
    # Dry bulk terminals
    db_location1 = 'db_rdam_emo'
    db_location2 = 'db_vliss_ovet'
    db_location3 = 'db_antw_bt_leopold'
    db_location4 = 'db_dunkirk_AM'

    # Plot comparisons for dry bulk terminals
    df_db = comparisons_db(db_location1, db_location2, db_location3, db_location4)

    # Liquid bulk terminals
    lb_location1 = 'lb_rdam_gate'
    lb_location2 = 'lb_belfast_puma'
    lb_location3 = 'lb_rdam_shell'
    lb_location4 = 'lb_vliss_total'

    # Plot comparisons for liquid bulk terminals
    df_lb = comparisons_lb(lb_location1, lb_location2, lb_location3, lb_location4)

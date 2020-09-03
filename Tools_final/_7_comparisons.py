""" Step 7. Comparisons between all 12 terminlas, based on the processed AIS data
Input: Processed data of all terminals [csv file]
Actions: Plot different comparisons between all the terminals
Output: Visuals including comparisons between all terminals
 """

import pandas as pd
pd.options.mode.chained_assignment = None
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
import seaborn as sns


# Compare vessel classes between terminals
def compare_classes(location1, location2, location3, location4):
    df1 = pd.read_csv('Data-frames/Results_phase_3/' + location1 + '/Df_stats_' + location1 + '.csv')
    df2 = pd.read_csv('Data-frames/Results_phase_3/' + location2 + '/Df_stats_' + location2 + '.csv')
    df3 = pd.read_csv('Data-frames/Results_phase_3/' + location3 + '/Df_stats_' + location3 + '.csv')
    df4 = pd.read_csv('Data-frames/Results_phase_3/' + location4 + '/Df_stats_' + location4 + '.csv')

    # Add terminal name to data
    df1['Terminal_location'] = location1
    df2['Terminal_location'] = location2
    df3['Terminal_location'] = location3
    df4['Terminal_location'] = location4

    # Concat all data frames into one
    df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)

    fig, ax = plt.subplots()
    df.groupby(['vessel_class', 'Terminal_location']).count()['terminal_entry_time'].unstack().plot.bar(ax=ax)
    plt.xticks(rotation=0)
    plt.legend(title='Terminal')
    plt.xlabel('Vessel class')
    plt.ylabel('Number of arrivals per vessel class')
    plt.title('Vessel class per terminal')


# Compare service times between terminals
def compare_st(location1, location2, location3, location4, terminal_type):
    df1 = pd.read_csv('Data-frames/Results_phase_3/' + location1 + '/Df_stats_' + location1 + '.csv')
    df2 = pd.read_csv('Data-frames/Results_phase_3/' + location2 + '/Df_stats_' + location2 + '.csv')
    df3 = pd.read_csv('Data-frames/Results_phase_3/' + location3 + '/Df_stats_' + location3 + '.csv')
    df4 = pd.read_csv('Data-frames/Results_phase_3/' + location4 + '/Df_stats_' + location4 + '.csv')

    # Add terminal name to data
    df1['Terminal_location'] = location1
    df2['Terminal_location'] = location2
    df3['Terminal_location'] = location3
    df4['Terminal_location'] = location4

    df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)
    if terminal_type == 'container':
        df_c = df.loc[df.vessel_class.isin(['c_1', 'c_2', 'c_3', 'c_4', 'c_5'])]
        order = ['c_1', 'c_2', 'c_3', 'c_4', 'c_5']
    if terminal_type == 'dry_bulk':
        df_c = df.loc[df.vessel_class.isin(['db_1', 'db_2', 'db_3', 'db_4', 'db_5'])]
        order = ['db_1', 'db_2', 'db_3', 'db_4', 'db_5']
    if terminal_type == 'liquid_bulk':
        df_c = df.loc[df.vessel_class.isin(['lng_1', 'lng_2', 'lng_3', 'lng_4'])]
        order = ['lng_1', 'lng_2', 'lng_3', 'lng_4']

    plt.figure()
    ax = sns.boxplot(x=df_c["vessel_class"], y=df_c["service_time[hr]"], hue=df_c['Terminal_location'],
                     order=order)
    plt.title('Service time per class, for different terminal locations')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(title='Terminal location')
    plt.xlabel('Vessel class')
    plt.ylabel('Service time [hr]')
    plt.show()

    plt.figure()
    ax = sns.violinplot(x=df_c["vessel_class"], y=df_c["service_time[hr]"], hue=df_c['Terminal_location'],
                        order=order)
    plt.title('Service time per class, for different terminal locations')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(title='Terminal location')
    plt.xlabel('Vessel class')
    plt.ylabel('Service time [hr]')
    plt.show()


# Compare service times between terminals
def compare_iat(location1, location2, location3, location4, terminal_type):
    df1 = pd.read_csv('Data-frames/Results_phase_3/' + location1 + '/Df_stats_' + location1 + '.csv')
    df2 = pd.read_csv('Data-frames/Results_phase_3/' + location2 + '/Df_stats_' + location2 + '.csv')
    df3 = pd.read_csv('Data-frames/Results_phase_3/' + location3 + '/Df_stats_' + location3 + '.csv')
    df4 = pd.read_csv('Data-frames/Results_phase_3/' + location4 + '/Df_stats_' + location4 + '.csv')

    # Add terminal name to data
    df1['Terminal_location'] = location1
    df2['Terminal_location'] = location2
    df3['Terminal_location'] = location3
    df4['Terminal_location'] = location4

    df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)
    if terminal_type == 'container':
        df_c = df.loc[df.vessel_class.isin(['c_1', 'c_2', 'c_3', 'c_4', 'c_5'])]
    if terminal_type == 'dry_bulk':
        df_c = df.loc[df.vessel_class.isin(['db_1', 'db_2', 'db_3', 'db_4', 'db_5'])]
    if terminal_type == 'liquid_bulk':
        df_c = df.loc[df.vessel_class.isin(['lng_1', 'lng_2', 'lng_3', 'lng_4'])]

    plt.figure()
    ax = sns.violinplot(x=df_c["Terminal_location"], y=df_c["inter_arrival_time_port[hr]"])
    plt.title('Inter arrival time per class, for different terminal locations')
    plt.xlabel('Terminal location')
    plt.ylabel('Inter arrival time [hr]')
    plt.show()


# Compare the length occupancy for container and dry bulk terminals
def compare_length_occup(location1, location2, location3, location4, l_1, l_2, l_3, l_4):
    df1 = pd.read_csv('Data-frames/Results_phase_3/' + location1 + '/Df_length_occup_' + location1 + '.csv')
    df2 = pd.read_csv('Data-frames/Results_phase_3/' + location2 + '/Df_length_occup_' + location2 + '.csv')
    df3 = pd.read_csv('Data-frames/Results_phase_3/' + location3 + '/Df_length_occup_' + location3 + '.csv')
    df4 = pd.read_csv('Data-frames/Results_phase_3/' + location4 + '/Df_length_occup_' + location4 + '.csv')

    # Add terminal name to data
    df1['Terminal_location'] = location1
    df2['Terminal_location'] = location2
    df3['Terminal_location'] = location3
    df4['Terminal_location'] = location4

    df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)

    # Plot terminal occupancy for every terminal
    plt.figure()
    sns.violinplot(x=df['Terminal_location'], y=df["occupancy_length_adjust[%]"])
    plt.title('Length occupancy per terminal locations')
    plt.xlabel('Terminal location')
    plt.ylabel('Length occupancy [%]')
    plt.show()

    # Plot average length occupancy vs terminal length
    plt.figure()
    plt.scatter(x=l_1, y=df1['occupancy_length_adjust[%]'].mean(), label=location1)
    plt.scatter(x=l_2, y=df2['occupancy_length_adjust[%]'].mean(), label=location2)
    plt.scatter(x=l_3, y=df3['occupancy_length_adjust[%]'].mean(), label=location3)
    plt.scatter(x=l_4, y=df4['occupancy_length_adjust[%]'].mean(), label=location4)
    plt.legend(title='Terminal location')
    plt.title('Length occupancy per terminal length')
    plt.xlabel('Terminal length [m]')
    plt.xlim(0, 3000)
    plt.ylabel('Average (adjusted) terminal occupancy [%]')
    plt.ylim(0, 100)
    plt.show()


# Compare the berth occupancy for liquid bulk terminals
def compare_berth_occup(location1, location2, location3, location4, bn_1, bn_2, bn_3, bn_4):
    df1 = pd.read_csv('Data-frames/Results_phase_3/' + location1 + '/Df_berth_occup_' + location1 + '.csv')
    df2 = pd.read_csv('Data-frames/Results_phase_3/' + location2 + '/Df_berth_occup_' + location2 + '.csv')
    df3 = pd.read_csv('Data-frames/Results_phase_3/' + location3 + '/Df_berth_occup_' + location3 + '.csv')
    df4 = pd.read_csv('Data-frames/Results_phase_3/' + location4 + '/Df_berth_occup_' + location4 + '.csv')

    # Add terminal name to data
    df1['Terminal_location'] = location1
    df2['Terminal_location'] = location2
    df3['Terminal_location'] = location3
    df4['Terminal_location'] = location4

    df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)

    # Plot terminal occupancy for every terminal
    plt.figure()
    sns.violinplot(x=df['Terminal_location'], y=df["occupancy[%]"],
                   order=[location1, location2, location3, location4])
    plt.title('Terminal occupancy per terminal locations')
    plt.xlabel('Terminal location')
    plt.ylabel('Terminal occupancy [%]')
    plt.show()

    # Plot average terminal occupancy vs number of berths
    plt.figure()
    plt.scatter(x=bn_1, y=df1['occupancy[%]'].mean(), label=location1)
    plt.scatter(x=bn_2, y=df2['occupancy[%]'].mean(), label=location2)
    plt.scatter(x=bn_3, y=df3['occupancy[%]'].mean(), label=location3)
    plt.scatter(x=bn_4, y=df4['occupancy[%]'].mean(), label=location4)
    plt.legend(title='Terminal location')
    plt.title('Terminal occupancy vs number of berths')
    plt.xlabel('Number of berths')
    plt.ylabel('Average terminal occupancy [%]')
    plt.ylim(0, 50)
    plt.xlim(0, 5)
    plt.show()


# Compare the service times between all 12 terminals
def compare_st_all(c_loc1, c_loc2, c_loc3, c_loc4, db_loc1, db_loc2, db_loc3, db_loc4, lb_loc1, lb_loc2, lb_loc3,
                   lb_loc4):
    c_df1 = pd.read_csv('Data-frames/Results_phase_3/' + c_loc1 + '/Df_stats_' + c_loc1 + '.csv')
    c_df2 = pd.read_csv('Data-frames/Results_phase_3/' + c_loc2 + '/Df_stats_' + c_loc2 + '.csv')
    c_df3 = pd.read_csv('Data-frames/Results_phase_3/' + c_loc3 + '/Df_stats_' + c_loc3 + '.csv')
    c_df4 = pd.read_csv('Data-frames/Results_phase_3/' + c_loc4 + '/Df_stats_' + c_loc4 + '.csv')

    db_df1 = pd.read_csv('Data-frames/Results_phase_3/' + db_loc1 + '/Df_stats_' + db_loc1 + '.csv')
    db_df2 = pd.read_csv('Data-frames/Results_phase_3/' + db_loc2 + '/Df_stats_' + db_loc2 + '.csv')
    db_df3 = pd.read_csv('Data-frames/Results_phase_3/' + db_loc3 + '/Df_stats_' + db_loc3 + '.csv')
    db_df4 = pd.read_csv('Data-frames/Results_phase_3/' + db_loc4 + '/Df_stats_' + db_loc4 + '.csv')

    lb_df1 = pd.read_csv('Data-frames/Results_phase_3/' + lb_loc1 + '/Df_stats_' + lb_loc1 + '.csv')
    lb_df2 = pd.read_csv('Data-frames/Results_phase_3/' + lb_loc2 + '/Df_stats_' + lb_loc2 + '.csv')
    lb_df3 = pd.read_csv('Data-frames/Results_phase_3/' + lb_loc3 + '/Df_stats_' + lb_loc3 + '.csv')
    lb_df4 = pd.read_csv('Data-frames/Results_phase_3/' + lb_loc4 + '/Df_stats_' + lb_loc4 + '.csv')

    # Add terminal names to data
    c_df1['Terminal_location'] = 'term1'
    c_df2['Terminal_location'] = 'term2'
    c_df3['Terminal_location'] = 'term3'
    c_df4['Terminal_location'] = 'term4'

    db_df1['Terminal_location'] = 'term1'
    db_df2['Terminal_location'] = 'term2'
    db_df3['Terminal_location'] = 'term3'
    db_df4['Terminal_location'] = 'term4'

    lb_df1['Terminal_location'] = 'term1'
    lb_df2['Terminal_location'] = 'term2'
    lb_df3['Terminal_location'] = 'term3'
    lb_df4['Terminal_location'] = 'term4'

    # Add terminal type to data
    c_df1['Terminal_type'] = 'container'
    c_df2['Terminal_type'] = 'container'
    c_df3['Terminal_type'] = 'container'
    c_df4['Terminal_type'] = 'container'

    db_df1['Terminal_type'] = 'dry_bulk'
    db_df2['Terminal_type'] = 'dry_bulk'
    db_df3['Terminal_type'] = 'dry_bulk'
    db_df4['Terminal_type'] = 'dry_bulk'

    lb_df1['Terminal_type'] = 'liquid_bulk'
    lb_df2['Terminal_type'] = 'liquid_bulk'
    lb_df3['Terminal_type'] = 'liquid_bulk'
    lb_df4['Terminal_type'] = 'liquid_bulk'

    df = pd.concat([c_df1, c_df2, c_df3, c_df4, db_df1, db_df2, db_df3, db_df4, lb_df1, lb_df2, lb_df3, lb_df4],
                   ignore_index=True, sort=False)

    plt.figure()
    ax = sns.violinplot(x=df["Terminal_type"], y=df["service_time[hr]"], hue=df['Terminal_location'])
    plt.title('Service time per class, for different terminal types')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(title='Terminal')
    plt.xlabel('Terminal type')
    plt.ylabel('Service time [hr]')
    plt.show()

    plt.figure()
    ax = sns.violinplot(x=df["Terminal_type"], y=df["service_time[hr]"], hue=df['Terminal_location'])
    plt.title('Service time per class, for different terminal types [zoom]')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(title='Terminal')
    plt.xlabel('Terminal type')
    plt.ylabel('Service time [hr]')
    plt.ylim(0, 250)
    plt.show()


# Test handle
if __name__ == '__main__':
    # Terminal locations
    location_c1 = 'ct_rdam_apm2'
    location_c2 = 'ct_rdam_apm'
    location_c3 = 'ct_rdam_euromax'
    location_c4 = 'ct_lehavre_atlantic'

    compare_classes(location_c1, location_c2, location_c3, location_c4)
    compare_st(location_c1, location_c2, location_c3, location_c4, 'container')
    compare_length_occup(location_c1, location_c2, location_c3, location_c4, 1500, 1500, 1900, 800)
    compare_iat(location_c1, location_c2, location_c3, location_c4, 'container')

    # Terminal locations
    location_db1 = 'db_rdam_emo'
    location_db2 = 'db_vliss_ovet'
    location_db3 = 'db_rdam_eecv'
    location_db4 = 'db_dunkirk'

    compare_classes(location_db1, location_db2, location_db3, location_db4)
    compare_st(location_db1, location_db2, location_db3, location_db4, 'dry_bulk')
    compare_length_occup(location_db1, location_db2, location_db3, location_db4, 2700, 950, 1090, 675)
    compare_iat(location_db1, location_db2, location_db3, location_db4, 'dry_bulk')

    # Terminal locations
    location_lb1 = 'lb_rdam_gate'
    location_lb2 = 'lb_zeebrugge'
    location_lb3 = 'lb_dunkirk'
    location_lb4 = 'lb_france_montoir'

    compare_classes(location_lb1, location_lb2, location_lb3, location_lb4)
    compare_st(location_lb1, location_lb2, location_lb3, location_lb4, 'liquid_bulk')
    compare_berth_occup(location_lb1, location_lb2, location_lb3, location_lb4, 2, 2, 1, 2)
    compare_iat(location_lb1, location_lb2, location_lb3, location_lb4, 'liquid_bulk')

    # # Comparisons between all terminals (service times)
    # compare_st_all(location_c1, location_c2, location_c3, location_c4, location_db1, location_db2, location_db3,
    #                location_db4, location_lb1, location_lb2, location_lb3, location_lb4)
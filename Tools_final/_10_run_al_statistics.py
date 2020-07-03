""" Step 10. Run all steps 6,7,8,9
 Input: New data frame with entry and exit timestamps for the port area, anchorage area, terminal area
 Actions: Attach waiting times, service times, waiting times/service time ratio, sort by port entry, inter arrival
 times, sort by port entry (relative to  first moment in time), visualise study parameters, fit multiple distributions
 Output: Data frame with study parameters, and multiple visualisations
 """

import pandas as pd
import numpy as np
from _6_attach_all_columns import service_waiting_times, sort_by_port_entry, sort_by_port_entry_rel
from _7_occupancy import run_all_occupancy
from _8_visualisation import plot_service_times, plot_inter_arrival_times, plot_waiting_times, plot_wt_st_ratio,\
    plot_wt_st, jointplot_wt_st
from _9_distribution_fitting import st_distributions, iat_distributions
import seaborn as sns


# Run all steps from _6_attach_all_columns
def attach_all_columns(df, relative):
    # Add service and waiting times
    df = service_waiting_times(df)
    # Based on port entry arrival time, return inter arrival time
    df_p = sort_by_port_entry(df)
    # Based on port entry arrival time, return all timestamps relative to t0
    # Only normalize data if input is given
    if relative == 1:
        df_p_rel = sort_by_port_entry_rel(df_p)
        df_p = df_p_rel

    # Return averages
    print('Average service time: ', np.round(df_p['service_time[hr]'].mean(), 2), 'hr')
    print('Average waiting time: ', np.round(df_p['waiting_time[hr]'].mean(), 2), 'hr')
    print('Average WT/ST ratio: ', np.round(df_p['waiting/service_time[%]'].mean(), 2), '%')

    return df_p


# Visualisations
def plot_visuals(df, ST, IAT, WT, adjust_WT, WT_ST, WT_ST_plt, WT_ST_joint):
    if ST == 1:
        # Plot service times
        plot_service_times(df, location + ': Service times')

    if IAT == 1:
        # Plot inter arrival times (based on port entry)
        plot_inter_arrival_times(df, location + ': Inter arrival times based on port entry')

    if WT == 1:
        # Plot waiting times
        plot_waiting_times(df, location + ': Waiting times', location + ': Adjusted waiting times', adjust_WT)

    if WT_ST == 1:
        # Plot WT/ST ratio
        plot_wt_st_ratio(df, location + ': Waiting vs service times', 0)
        plot_wt_st_ratio(df, location + ': Waiting vs service times (adjusted y lim)', 1)

    if WT_ST_plt == 1:
        # Plot waiting times across service times
        plot_wt_st(df, location + ': Service times vs waiting times', 10, 20, 30)

    if WT_ST_joint == 1:
        # Jointplot of waiting times vs service times
        jointplot_wt_st(df)


# Test handle
if __name__ == '__main__':
    # Load raw data
    location = 'db_dunkirk_AM'
    df = pd.read_csv('Data-frames/Results_phase_2/' + location + '/Final_df_' + location + '.csv')

    """ ....... INPUTS ......... """
    # Relative = 0: keep original data timestamps, = 1: normalize data
    relative = 0
    # Number of berths: (1,2,3... number, or if unknown: 0)
    number_of_berths = 0 # Input
    # Operating hours per year:
    operating_hours = 365 * 24  # Input
    # Visualise berth occupancy over time (1 = yes) (visualises original occupancy, not relative)
    visualise_berth_oc = 0  # Input
    # Total length terminal [m] (if unknown: 0)
    length_term = 1700 # Input
    # Visualise length occupancy over time (1 = yes) (visualises original length, not relative)
    visualise_length_oc = 1  # Input
    # Visualise service times # 1 = yes, 0 = no
    ST_vis = 1
    # Visualise inter arrival times # 1 = yes, 0 = no
    IAT_vis = 1
    # Visualise waiting times # 1 = yes, 0 = no.
    adjust_WT = 5 # number of first x hrs to delete from adjusted visualisation of waiting times
    WT_vis = 1
    # Visualise WT/ST ratio in box-plot # 1 = yes, 0 = no.
    WT_ST_vis = 1
    # Visualise WT in terms of ST # 1 = yes, 0 = no.
    WT_ST_plt_vis = 1
    # Visualise joint-plot wt - st
    WT_ST_joint = 1

    """ ........ Analyse processed AIS data frame ......... """
    # Attach all columns (ST, WT, IAT)
    df = attach_all_columns(df, relative)

    # Run all occupancy steps
    df_berth_occupancy, df_length_occupancy, df = run_all_occupancy(df, number_of_berths, operating_hours,
                                                                        visualise_berth_oc, length_term,
                                                                        visualise_length_oc)

    # Visualisations
    plot_visuals(df, ST_vis, IAT_vis, WT_vis, adjust_WT, WT_ST_vis, WT_ST_plt_vis, WT_ST_joint)

    # Correlation between WT and ST
    print('The correlation between WT and ST is: ', (df['service_time[hr]']).corr(df['waiting_time[hr]'])*100, '%')

    # Fit distributions on ST
    df_st = np.around(st_distributions(df, location), 4)

    # Fit distributions on IAT
    df_iat = np.around(iat_distributions(df, location), 4)

    # Save data-frames
    df.to_csv('Data-frames/Results_phase_2/' + location + '/Df_stats_' + location + '.csv')
    df_st.to_csv('Data-frames/Results_phase_2/' + location + '/Df_st_' + location + '.csv')
    df_iat.to_csv('Data-frames/Results_phase_2/' + location + '/Df_iat_' + location + '.csv')
    # Only if length occupancy is available
    if length_term > 0:
        df_length_occupancy.to_csv('Data-frames/Results_phase_2/' + location + '/Df_length_occup_' + location + '.csv')
    # Only if terminal occupancy is available
    if number_of_berths > 0:
        df_berth_occupancy.to_csv('Data-frames/Results_phase_2/' + location + '/Df_berth_occup_' + location + '.csv')


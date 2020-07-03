""" Step 9. Distribution fitting
 Input: Data frame with study parameters
 Actions: Fit multiple distributions on data set, find best fit based on K-S test and Chi-squared test
 Output: Fitted distributions for service and inter arrival times
 """

import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
import scipy.stats
import matplotlib.pyplot as plt
from fitter import Fitter


"""" ..............  INTER ARRIVAL TIMES .................... """


def iat_distributions(df, location):
    data = df['inter_arrival_time_port[hr]']
    bin_number = 25
    x_0 = np.linspace(0, df['inter_arrival_time_port[hr]'].max(), 100)
    y, x = np.histogram(data, bins=bin_number, density=True)  # sse dependent on number of bins
    x = (x + np.roll(x, -1))[:-1] / 2.0

    distribution = []
    location_par = []
    scale_par = []
    shape_par = []
   # sse = []
    p = []
    chi = []

    histo, bin_edges = np.histogram(data, bins=bin_number, density=False) # chisquare dependent on number of bins
    number_of_bins = bin_number
    observed_values = histo
    n = len(data)

    # Exponential distribution (same as Gamma with a = 1 and Weibull with c =1)
    distribution.append('Exponential')
    exp_loc, exp_scale = scipy.stats.distributions.expon.fit(data)
    fitted_data_exp = scipy.stats.distributions.expon.pdf(x_0, exp_loc, exp_scale)
    location_par.append(exp_loc)
    scale_par.append(exp_scale)
    shape_par.append(1)
    pdf_exp = scipy.stats.expon.pdf(x, loc=exp_loc, scale=exp_scale)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_exp, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.expon.cdf, args=scipy.stats.expon.fit(data))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.expon.cdf(bin_edges, exp_loc, exp_scale)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Gamma distribution
    distribution.append('Gamma')
    a_gam, loc_gam, scale_gam = np.around(scipy.stats.distributions.gamma.fit(data), 5)
    fitted_data_gam = scipy.stats.distributions.gamma.pdf(x_0, a_gam, loc_gam, scale_gam)
    location_par.append(loc_gam)
    scale_par.append(scale_gam)
    shape_par.append(a_gam)
    pdf_gam = scipy.stats.gamma.pdf(x, a=a_gam, loc=loc_gam, scale=scale_gam)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_gam, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.gamma.cdf(bin_edges, a_gam, loc_gam, scale_gam)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Erlang-2 distribution (same as Gamma with a = 2)
    distribution.append('Erlang_2')
    a_gam2, loc_gam2, scale_gam2 = np.around(scipy.stats.distributions.gamma.fit(data, fa=2), 5)
    fitted_data_gam2 = scipy.stats.distributions.gamma.pdf(x_0, a_gam2, loc_gam2, scale_gam2)
    location_par.append(loc_gam2)
    scale_par.append(scale_gam2)
    shape_par.append(a_gam2)
    pdf_gam2 = scipy.stats.gamma.pdf(x, a=a_gam2, loc=loc_gam2, scale=scale_gam2)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_gam2, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data, fa=2))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.gamma.cdf(bin_edges, a_gam2, loc_gam2, scale_gam2)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Weibull distribution
    distribution.append('Weibull_min')
    c_weib, loc_weib, scale_weib = np.around(scipy.stats.distributions.weibull_min.fit(data), 5)
    fitted_data_weib = scipy.stats.distributions.weibull_min.pdf(x_0, c_weib, loc_weib, scale_weib)
    location_par.append(loc_weib)
    scale_par.append(scale_weib)
    shape_par.append(c_weib)
    pdf_weib = scipy.stats.weibull_min.pdf(x, c=c_weib, loc=loc_weib, scale=scale_weib)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_weib, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.weibull_min.cdf, args=scipy.stats.weibull_min.fit(data))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.weibull_min.cdf(bin_edges, c_weib, loc_weib, scale_weib)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Plot
    plt.figure()
    plt.hist(data, bins=bin_number, density=True)
    plt.plot(x_0, fitted_data_exp, 'r-', label='Exponential')
    plt.plot(x_0, fitted_data_gam, 'b-', label='Gamma')
    plt.plot(x_0, fitted_data_gam2, 'm-', label='Erlang-2')
    plt.plot(x_0, fitted_data_weib, 'y-', label='Weibull_min')
    plt.legend(title='Distributions')
    plt.ylabel('Pdf [%]')
    plt.xlabel('Inter arrival times [hr]')
    plt.title(location + ': distribution fitting for inter arrival times')
    plt.show()

    df_results_iat = pd.DataFrame(distribution, columns=['Distribution_type'])
    df_results_iat['Location_parameter'] = location_par
    df_results_iat['Scale_parameter'] = scale_par
    df_results_iat['Shape_parameter'] = shape_par
  #  df_results_iat['sse'] = sse
    df_results_iat['p'] = p
    df_results_iat['chi'] = chi
  #  df_results_iat = df_results_iat.sort_values(by=['p'], ascending=False)


    return df_results_iat


"""" ..............  SERVICE TIMES .................... """


def st_distributions(df, location):
    data = df['service_time[hr]']
    bin_number = 25
    x_0 = np.linspace(0, df['service_time[hr]'].max(), 100)
    y, x = np.histogram(data, bins=bin_number, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    distribution = []
    location_par = []
    scale_par = []
    shape_par = []
    # sse = []
    p = []
    chi = []

    histo, bin_edges = np.histogram(data, bins=bin_number, density=False)
    number_of_bins = len(bin_edges) - 1
    observed_values = histo
    n = len(data)

    # Exponential distribution (same as Gamma with a = 1 and Weibull with c =1)
    distribution.append('Exponential')
    exp_loc, exp_scale = scipy.stats.distributions.expon.fit(data)
    fitted_data_exp = scipy.stats.distributions.expon.pdf(x_0, exp_loc, exp_scale)
    location_par.append(exp_loc)
    scale_par.append(exp_scale)
    shape_par.append(1)
    pdf_exp = scipy.stats.expon.pdf(x, loc=exp_loc, scale=exp_scale)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_exp, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.expon.cdf, args=scipy.stats.expon.fit(data))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.expon.cdf(bin_edges, exp_loc, exp_scale)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Gamma distribution
    distribution.append('Gamma')
    a_gam, loc_gam, scale_gam = np.around(scipy.stats.distributions.gamma.fit(data), 5)
    fitted_data_gam = scipy.stats.distributions.gamma.pdf(x_0, a_gam, loc_gam, scale_gam)
    location_par.append(loc_gam)
    scale_par.append(scale_gam)
    shape_par.append(a_gam)
    pdf_gam = scipy.stats.gamma.pdf(x, a=a_gam, loc=loc_gam, scale=scale_gam)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_gam, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.gamma.cdf(bin_edges, a_gam, loc_gam, scale_gam)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Erlang-2 distribution (same as Gamma with a = 2)
    distribution.append('Erlang_2')
    a_gam2, loc_gam2, scale_gam2 = np.around(scipy.stats.distributions.gamma.fit(data, fa=2), 5)
    fitted_data_gam2 = scipy.stats.distributions.gamma.pdf(x_0, a_gam2, loc_gam2, scale_gam2)
    location_par.append(loc_gam2)
    scale_par.append(scale_gam2)
    shape_par.append(a_gam2)
    pdf_gam2 = scipy.stats.gamma.pdf(x, a=a_gam2, loc=loc_gam2, scale=scale_gam2)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_gam2, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data, fa=2))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.gamma.cdf(bin_edges, a_gam2, loc_gam2, scale_gam2)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Erlang-3 distribution (same as Gamma with a = 3)
    distribution.append('Erlang_3')
    a_gam3, loc_gam3, scale_gam3 = np.around(scipy.stats.distributions.gamma.fit(data, fa=3), 5)
    fitted_data_gam3 = scipy.stats.distributions.gamma.pdf(x_0, a_gam3, loc_gam3, scale_gam3)
    location_par.append(loc_gam3)
    scale_par.append(scale_gam3)
    shape_par.append(a_gam3)
    pdf_gam3 = scipy.stats.gamma.pdf(x, a=a_gam3, loc=loc_gam3, scale=scale_gam3)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_gam3, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data, fa=3))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.gamma.cdf(bin_edges, a_gam3, loc_gam3, scale_gam3)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Erlang-4 distribution (same as Gamma with a = 4)
    distribution.append('Erlang_4')
    a_gam4, loc_gam4, scale_gam4 = np.around(scipy.stats.distributions.gamma.fit(data, fa=4), 5)
    fitted_data_gam4 = scipy.stats.distributions.gamma.pdf(x_0, a_gam4, loc_gam4, scale_gam4)
    location_par.append(loc_gam4)
    scale_par.append(scale_gam4)
    shape_par.append(a_gam4)
    pdf_gam4 = scipy.stats.gamma.pdf(x, a=a_gam4, loc=loc_gam4, scale=scale_gam4)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_gam4, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data, fa=4))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.gamma.cdf(bin_edges, a_gam4, loc_gam4, scale_gam4)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Erlang-5 distribution (same as Gamma with a = 5)
    distribution.append('Erlang_5')
    a_gam5, loc_gam5, scale_gam5 = np.around(scipy.stats.distributions.gamma.fit(data, fa=5), 5)
    fitted_data_gam5 = scipy.stats.distributions.gamma.pdf(x_0, a_gam5, loc_gam5, scale_gam5)
    location_par.append(loc_gam5)
    scale_par.append(scale_gam5)
    shape_par.append(a_gam5)
    pdf_gam5 = scipy.stats.gamma.pdf(x, a=a_gam5, loc=loc_gam5, scale=scale_gam5)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_gam5, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.gamma.cdf, args=scipy.stats.gamma.fit(data, fa=5))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.gamma.cdf(bin_edges, a_gam5, loc_gam5, scale_gam5)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Normal distribution
    distribution.append('Normal')
    loc_norm, scale_norm = np.around(scipy.stats.distributions.norm.fit(data), 5)
    fitted_data_norm = scipy.stats.distributions.norm.pdf(x_0, loc_norm, scale_norm)
    location_par.append(loc_norm)
    scale_par.append(scale_norm)
    shape_par.append(np.nan)
    pdf_norm = scipy.stats.norm.pdf(x, loc=loc_norm, scale=scale_norm)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_norm, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.norm.cdf, args=scipy.stats.norm.fit(data))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.norm.cdf(bin_edges, loc_norm, scale_norm )
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Beta distribution
    distribution.append('Beta')
    a_beta, b_beta, loc_beta, scale_beta = np.around(scipy.stats.distributions.beta.fit(data), 5)
    fitted_data_beta = scipy.stats.distributions.beta.pdf(x_0, a_beta, b_beta, loc_beta, scale_beta)
    location_par.append(loc_beta)
    scale_par.append(scale_beta)
    shape_par.append(a_beta)
    pdf_beta = scipy.stats.beta.pdf(x, a_beta, b_beta, loc=loc_beta, scale=scale_beta)
    # # sum of square error
    # sse.append(np.around(np.sum(np.power(y - pdf_beta, 2.0)), 5))
    # Obtain the KS test P statistic
    p.append(np.around(scipy.stats.kstest(data, scipy.stats.beta.cdf, args=scipy.stats.beta.fit(data))[1], 5))
    # Chi-squared test
    cdf = scipy.stats.beta.cdf(bin_edges, a_beta, b_beta, loc_beta, scale_beta)
    expected_values = n * np.diff(cdf)
    chi.append((scipy.stats.chisquare(observed_values, expected_values))[0])

    # Plot
    plt.figure()
    plt.hist(data, bins=bin_number, density=True)
    plt.plot(x_0, fitted_data_exp, 'r-', label='Exponential')
    plt.plot(x_0, fitted_data_gam, 'b-', label='Gamma')
    plt.plot(x_0, fitted_data_gam2, 'm-', label='Erlang-2')
    plt.plot(x_0, fitted_data_gam3, 'y-', label='Erlang-3')
    plt.plot(x_0, fitted_data_gam4, 'g-', label='Erlang-4')
    plt.plot(x_0, fitted_data_gam5, 'k-', label='Erlang-5')
    plt.plot(x_0, fitted_data_norm, 'c-', label='Normal')
    plt.plot(x_0, fitted_data_beta, 'pink', label='Beta')
    plt.legend(title='Distributions')
    plt.title(location + ': distribution fitting for service times')
    plt.ylabel('Pdf [%]')
    plt.xlabel('Service times [hr]')
    plt.show()

    df_results_st = pd.DataFrame(distribution, columns=['Distribution_type'])
    df_results_st['Location_parameter'] = location_par
    df_results_st['Scale_parameter'] = scale_par
    df_results_st['Shape_parameter'] = shape_par
    # df_results_st['sse'] = sse
    df_results_st['p'] = p
    df_results_st['chi'] = chi
  #  df_results_st = df_results_st.sort_values(by=['p'], ascending=False)

    return df_results_st


# Test handle
if __name__ == '__main__':
    # Load data frame with attached columns, sorted by port entry time
    location = 'ct_rdam_euromax'
    df = pd.read_csv('Data-frames/Results_phase_2/' + location + '/Df_stats_' + location + '.csv')

    # Fit expected distributions for inter arrival times
    df_iat = iat_distributions(df, location)

    # Choose best distribution for service times
    df_st = st_distributions(df, location)


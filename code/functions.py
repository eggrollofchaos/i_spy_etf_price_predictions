from random import random
import pandas as pd
import matplotlib
import numpy as np
import csv
import itertools
import pickle as pkl

import time
import datetime
from dateutil.relativedelta import relativedelta

import pmdarima as pm
from pmdarima import pipeline
import prophet
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter
# plt.style.use('ggplot')
sns.set_theme(style="darkgrid")
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

font = {'size'   : 12}
matplotlib.rc('font', **font)

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',25)

from pathlib import Path
top = Path(__file__ + '../../..').resolve()
# columns for Alpha Vantage output
AV_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'volume']

# custom formatter

class CustomFormatter(FuncFormatter):
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''

        return num2date(self.dates[ind]).strftime(self.fmt)


# def format_date(x, N, pos=None):
#     thisind = np.clip(int(x + 0.5), 0, N-1)
#     return df.index[thisind].strftime('%Y-%m-%d')

# displaying years / months in figures
years = mdates.YearLocator()
years_fmt = mdates.DateFormatter('%Y')
months = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%B %Y')
days = mdates.DayLocator()
days_fmt = mdates.DateFormatter('%B %d, %Y')
hours = mdates.HourLocator()
hours_fmt = mdates.DateFormatter('%B %d, %Y %H:00')
print('Functions loaded.')

################################################################################

def get_av_all_data_slices(symbol, ts, y=2, m=12, interval = '60min', verbose=0):
    '''
    Takes in a symbol, alpha_vantage TimeSeries object, and calls get_av_next_data_slice
    iteratively until all slices are returned.
    symbol : str
    ts : alpha_vantage TimeSeries object
        e.g. ts = alpha_vantage.timeseries.TimeSeries(key=av_key, output_format='csv')
    y : int, optional (default = 2)
        total historical years to return [1,2]
    m : int, optional (default = 12)
        total historical months per year to return [1,12]
    interval : str, optional (default '15min')
        time interval between two conscutive values,
        supported values are '1min', '5min', '15min', '30min', '60min'
    verbose : int, optional (default = 0)
        verbose output [0,1]
    '''
    df = pd.DataFrame(columns=AV_COLUMNS)
    df.set_index('time', inplace=True)
    df.index = pd.to_datetime(df.index)
    print(f'Requesting historical intraday price data slices for {symbol}...\n')
    for y in range(y):
        for m in range(m):
            try:
                fieldnames, data_reader, data_slice, tries = get_av_next_data_slice(symbol, ts, y, m, interval, 0, verbose)
            except TypeError:
                print('Error in get_intraday_extended function call.') if verbose == 1 else None
                raise
            except:
                print("Error in running get_next_data_slice function:", sys.exc_info()[0]) if verbose == 1 else None
                raise
            else:
                print("GET_INTRADAY_EXTENDED ran with no errors.") if verbose == 1 else None
            try:
                df_month = pd.DataFrame(data_reader, columns=fieldnames)
                assert(len(df_month)>0), f"Returned dataset is empty, please check Alpha Vantage params."
            except AssertionError:
                raise
            else:
                df_month.set_index('time', inplace=True)
                df_month.index = pd.to_datetime(df_month.index)
                if verbose == 1:
                    print('First record:')
                    print(df_month.head(1))
                    print('Last record:')
                    print(df_month.tail(1))
                df = df.append(df_month)
                print(f"Processed and appended {data_slice} to DataFrame.\n") if verbose == 1 else None
    print(f'Finished getting all data slices for {symbol}')
    return df

def get_av_next_data_slice(symbol, ts, y, m, interval = '60min', tries=0, verbose=0):
    data_slice = f'year{y+1}month{m+1}'
    try:
        if verbose==1:
            if tries>0:
                print(f'Re-requesting slice: {data_slice}')
            else:
                print(f'Requesting slice: {data_slice}')
        data_reader, meta_data = ts.get_intraday_extended(symbol=symbol, interval=interval, slice=data_slice)
    except TypeError:
        raise
    else:
        try:
            fieldnames = next(data_reader)
#             print(fieldnames)
            assert(len(fieldnames) == 6), f"Error: Header row length is {len(fieldnames)}, expected 6."
            assert(fieldnames==AV_COLUMNS), f"Columns mismatch from Alpha Vantage output."
        except AssertionError as e:
            sleep_time = 10 + tries + tries*random()
            if verbose == 1:
                print(e)
                print(f'Sleeping for {sleep_time}...')
            time.sleep(sleep_time)
            tries+=1
            fieldnames, data_reader, data_slice, tries = get_av_next_data_slice(symbol, ts, y, m, interval, tries=tries, verbose=verbose)
    return fieldnames, data_reader, data_slice, tries

# def visualize_data(df):
#     pass

# def create_df_dict(df):
#     zipcodes = list(set(df.zipcode))
#     keys = [zipcode for zipcode in map(str,zipcodes)]
#     data_list = []
#
#     for key in keys:
#         new_df = df[df.zipcode == int(key)]
#         new_df.drop('zipcode', inplace=True, axis=1)
#         new_df.columns = ['date', 'value']
#         new_df.date = pd.to_datetime(new_df.date)
#         new_df.set_index('date', inplace=True)
#         new_df = new_df.asfreq('M')
#         data_list.append(new_df)
#
#     df_dict = dict(zip(keys, data_list))
#
#     return df_dict

def test_stationarity(df_all, diffs=0):
    if diffs == 2:
        dftest = adfuller(df_all.diff().diff().dropna())
    elif diffs == 1:
        dftest = adfuller(df_all.diff().dropna())
    else:
        dftest = adfuller(df_all)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value', '#Lags Used',
                                             'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print (dfoutput)

def plot_pacf_with_diff(df, symbol, n, period, freq, lags, alpha=0.05):
    timeframe = f'{n} {period.title()}'
    pacf_fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    pacf_fig.suptitle(f'Partial Autocorrelations of {symbol} Time Series: {timeframe}, Frequency = {freq}', fontsize=18)
    plot_pacf(df, ax=ax[0], lags=lags, alpha=alpha)
    ax[0].set_title('Undifferenced PACF', size=14)
    ax[0].set_xlabel('Lags', size=14)
    ax[0].set_ylabel('PACF', size=14)
    plot_pacf(df.diff().dropna(), ax=ax[1], lags=lags, alpha=alpha)
    ax[1].set_title('Differenced PACF', size=14)
    ax[1].set_xlabel('Lags', size=14)
    ax[1].set_ylabel('PACF', size=14)
    plot_pacf(df.diff().diff().dropna(), ax=ax[2], lags=lags, alpha=alpha)
    ax[2].set_title('Twice-Differenced PACF', size=14)
    ax[2].set_xlabel('Lags', size=14)
    ax[2].set_ylabel('ACF', size=14)
    pacf_fig.tight_layout()
    pacf_fig.subplots_adjust(top=0.9)
    plt.savefig(f'{top}/images/{symbol}_{timeframe}_{freq}_PACF.png')

def plot_acf_with_diff(df, symbol, n, period, freq, lags, alpha=0.05):
    timeframe = f'{n} {period.title()}'
    acf_fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    acf_fig.suptitle(f'Autocorrelations of {symbol} Time Series: {timeframe}, Frequency = {freq}', fontsize=18)
    plot_acf(df, ax=ax[0], lags=lags, alpha=alpha)
    ax[0].set_title('Undifferenced ACF', size=14)
    ax[0].set_xlabel('Lags', size=14)
    ax[0].set_ylabel('ACF', size=14)
    plot_acf(df.diff().dropna(), ax=ax[1], lags=lags, alpha=alpha)
    ax[1].set_title('Once-Differenced ACF', size=14)
    ax[1].set_xlabel('Lags', size=14)
    ax[1].set_ylabel('ACF', size=14)
    plot_acf(df.diff().diff().dropna(), ax=ax[2], lags=lags, alpha=alpha)
    ax[2].set_title('Twice-Differenced ACF', size=14)
    ax[2].set_xlabel('Lags', size=14)
    ax[2].set_ylabel('ACF', size=14)
    acf_fig.tight_layout()
    acf_fig.subplots_adjust(top=0.9)
    plt.savefig(f'{top}/images/{symbol}_{timeframe}_{freq}_ACF.png')

def plot_seasonal_decomposition(df, symbol, n, period, freq, seas):
    timeframe = f'{n} {period.title()}'
    delta_freq = freq.split()[1].lower() + 's'
    delta = {}
    delta[delta_freq] = 15
    decomp = seasonal_decompose(df, period=seas)
    dc_obs = decomp.observed
    dc_trend = decomp.trend
    dc_seas = decomp.seasonal
    dc_resid = decomp.resid
    dc_df = pd.DataFrame({"observed": dc_obs, "trend": dc_trend,
                            "seasonal": dc_seas, "residual": dc_resid})
    start = dc_df.iloc[:, 0].index[0]
    # end = dc_df.iloc[:, 0].index[-1] + relativedelta(months=+15) + relativedelta(day=31)
    try:
        end = dc_df.iloc[:, 0].index[-1] + relativedelta(**delta)
    except:
        print("Error in relativedelta function call, please check params.")
        raise
    # formatter = FuncFormatter(format_date)

    decomp_fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    for i, ax in enumerate(axes):
        # print(dc_df.iloc[:,i].dropna())
        ax.plot(dc_df.iloc[:, i])
        # ax.plot(range(dc_df.iloc[:,i].dropna().size), dc_df.iloc[:,i].dropna())
        # ax.set_xticklabels(dc_df.dropna().index.date.tolist())
        # ax.set_xticklabels(dc_df.iloc[:,i].dropna().index.date.tolist());
        # decomp_fig.autofmt_xdate()
        # ax.set_xlim(lims[i])
        ax.set_xlim(start, end)
        # ax.set_xlim(0, (dc_df.iloc[:,i].dropna().size)+15)
        # ax.xaxis.set_major_locator(months)
        # ax.xaxis.set_major_formatter(months_fmt)
        # formatter = CustomFormatter(df)
        # ax.xaxis.set_major_formatter(formatter)
        ax.set_ylabel(dc_df.iloc[:, i].name)
        # if i != 2:
        #     ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation=45, rotation_mode="anchor")

    decomp_fig.suptitle(
        f'Seasonal Decomposition of {symbol} Time Series: {timeframe}, Frequency = {freq}', fontsize=24)
    decomp_fig.tight_layout()
    decomp_fig.subplots_adjust(top=0.94)
    plt.savefig(f'{top}/images/{symbol}_{timeframe}_{freq}_seasonal_decomp.png')


def train_test_split_data(df, train_size=80, verbose=0):
    if verbose==1:
        print('##### Train-Test Split #####')
        print(f'Using a {train_size}/{100-train_size} train-test split...')
    cutoff = round((train_size/100)*len(df))
    train_df = df[:cutoff]
    test_df = df[cutoff:]
    return train_df, test_df

def gridsearch_SARIMAX(train_df, seas = round(24*365/4), p_min=2, p_max=2, q_min=0, q_max=0, d_min=1, d_max=1,
                       s_p_min=2, s_p_max=2, s_q_min=0, s_q_max=0, s_d_min=1, s_d_max=1):
    p = range(p_min, p_max+1)
    q = range(q_min, q_max+1)
    d = range(d_min, d_max+1)
    s_p = range(s_p_min, s_p_max+1)
    s_q = range(s_q_min, s_q_max+1)
    s_d = range(s_d_min, s_d_max+1)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], seas) for x in list(itertools.product(s_p, s_d, s_q))]
    print('Parameters for SARIMAX grid search...')
    for i in pdq:
        for s in seasonal_pdq:
            print('SARIMAX: {} x {}'.format(i, s))

    param_list = []
    param_seasonal_list = []
    aic_list = []

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            mod = SARIMAX(train_df,
                          order=param,
                          seasonal_order=param_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit()
            zipcodes.append(zipcode[-5:])
            param_list.append(param)
            param_seasonal_list.append(param_seasonal)
            aic = mod.aic
            aic_list.append(aic)
            print(f'Zip code {zipcode}: {aic}')
    return param_list, param_seasonal_list, aic_list

def get_best_params(zipcodes, param_list, param_seasonal_list, aic_list, bedrooms):
    # intialize list of model params
    model_data = {'zipcode': zipcodes,
                  'param': param_list,
                  'param_seasonal': param_seasonal_list,
                  'aic': aic_list
                  }
    # Create model params DataFrames
    sarimax_details_df = pd.DataFrame(model_data)
#     print(sarimax_details_df.shape)

    best_params_df = sarimax_details_df.loc[sarimax_details_df.groupby('zipcode')['aic'].idxmin()]
    best_params_df.set_index('zipcode', inplace=True)
    print(best_params_df)
    best_params_df.to_csv(f'data/{bedrooms}_bdrm_best_params.csv')
    return best_params_df

def evaluate_model(train_dict, test_dict, model_best_df):
    predict_dict = {}
    cat_predict_dict = train_dict.copy()
    for _ in range(5):
        for zipcode, df in cat_predict_dict.items():
            if cat_predict_dict[zipcode].index[-1] >= pd.to_datetime('2021-02-28'):
                continue
            sari_mod = SARIMAX(df,
                               order=model_best_df.loc[zipcode].param,
                               seasonal_order=model_best_df.loc[zipcode].param_seasonal,
                               enforce_stationarity=False,
                               enforce_invertibility=False).fit()

            predict = sari_mod.forecast(steps = 12, dynamic = False)
            print((zipcode,predict.index[-1],predict[-1]))
            predict_dict[zipcode] = predict
            dfB = pd.DataFrame(predict_dict[zipcode])
            dfB.columns = ['value']
            dfA = cat_predict_dict[zipcode]
            cat_predict_dict[zipcode] = pd.concat([dfA, dfB], axis=0)
    return cat_predict_dict

def calc_RMSE(test_dict, predictions_dict, bedrooms):
    zipcodes = []
    RMSE_list = []
    hv = []
    for zipcode, df in test_dict.items():
        window = len(df)
        RMSE = metrics.mean_squared_error(test_dict[zipcode], predictions_dict[zipcode].iloc[-window:], squared=False)
        zipcodes.append(zipcode)
        RMSE_list.append(RMSE)

    # get last observed house value per zip code
    for zipcode, df in test_dict.items():
        hv.append(df.iloc[-1].value)
    RMSE_data = {'zipcode': zipcodes,
                 'RMSE': RMSE_list,
                 'last_value': hv
                 }
    RMSE_df = pd.DataFrame(RMSE_data)
    RMSE_df = RMSE_df.sort_values('RMSE', axis=0, ascending=False)
    RMSE_df['RMSE_vs_value'] = 100*RMSE_df.RMSE/RMSE_df.last_value
    RMSE_df.set_index('zipcode', inplace=True)
    print(RMSE_df)
    RMSE_df.to_csv(f'data/{bedrooms}_bdrm_RMSE.csv')
    return RMSE_df

def gridsearch_SARIMAX_test_predict(train_dict, test_dict, seas = 12, p_min=2, p_max=2, q_min=0, q_max=0, d_min=1, d_max=1,
                       s_p_min=2, s_p_max=2, s_q_min=0, s_q_max=0, s_d_min=1, s_d_max=1):
    p = range(p_min, p_max+1)
    q = range(q_min, q_max+1)
    d = range(d_min, d_max+1)
    s_p = range(s_p_min, s_p_max+1)
    s_q = range(s_q_min, s_q_max+1)
    s_d = range(s_d_min, s_d_max+1)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], seas) for x in list(itertools.product(s_p, s_d, s_q))]
    print('Parameters for SARIMAX grid search for test predictions...')
    for i in pdq:
        for s in seasonal_pdq:
            print('SARIMAX: {} x {}'.format(i, s))

    zipcodes = []
    param_list = []
    param_seasonal_list = []
    RMSE_list = []
    predict_dict = {}
    cat_predict_dict = train_dict.copy()

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            predict_dict = {}
            cat_predict_dict = train_dict.copy()
            for count in range(5):
                for zipcode, df in cat_predict_dict.items():
                    if cat_predict_dict[zipcode].index[-1] >= pd.to_datetime('2021-02-28'):
                        print(param, param_seasonal)
                        window = len(test_dict[zipcode])
                        RMSE = metrics.mean_squared_error(test_dict[zipcode], cat_predict_dict[zipcode].iloc[-window:], squared=False)
                        zipcodes.append(zipcode)
                        param_list.append(param)
                        param_seasonal_list.append(param_seasonal)
                        RMSE_list.append(RMSE)
                        print(f'Zip code {zipcode}: {RMSE}')
                        continue

                    sari_mod = SARIMAX(df,
                                       order=param,
                                       seasonal_order=param_seasonal,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False).fit()

                    predict = sari_mod.forecast(steps = 12, dynamic = False)
                    print((zipcode,predict.index[-1],predict[-1]))
                    predict_dict[zipcode] = predict
                    dfB = pd.DataFrame(predict_dict[zipcode])
                    dfB.columns = ['value']
                    dfA = cat_predict_dict[zipcode]
                    cat_predict_dict[zipcode] = pd.concat([dfA, dfB], axis=0)

    return zipcodes, param_list, param_seasonal_list, RMSE_list

def plot_train_test(test_dict, predictions_dict, model_best_df, bedrooms):
    for zipcode, df in test_dict.items():
        fig, ax = plt.subplots()
        ax.plot(df.index, df.value, label='Test')
        ax.plot(predictions_dict[zipcode].index, predictions_dict[zipcode].value, label='Test Predictions')
        ax.set_title(
            f'{bedrooms}-Bedroom San Francisco {zipcode} Home Values: Test vs Predictions\nusing SARIMAX{model_best_df.loc[zipcode].param}x{model_best_df.loc[zipcode].param_seasonal}')
        plt.legend()
        plt.savefig(f'{top}/images/{bedrooms}_bdrm_test_predict{zipcode}.png')

def plot_RMSE(RMSE_df, bedrooms):
    fig, ax = plt.subplots(figsize = (12,8))
    ax.bar(x=RMSE_df.index, height=RMSE_df.RMSE, color = 'b', alpha=0.4, label = 'RMSE')
    ax.set_ylabel('RMSE (USD)', size = 18)
    ax.set_xlabel('Zip Code', size = 18)
    ax.set_ylim(0,4.2e5)
    ax1 = ax.twinx()
    ax1.bar(x=RMSE_df.index, height=RMSE_df.RMSE_vs_value, color = 'g', alpha=0.3, label = 'RMSE as % of Home Value')
    ax1.set_ylabel('RMSE as Percentage of Home Value (%)', size = 18)
    ax1.set_ylim(0,42)
    ax.set_title(f'{bedrooms}-Bedroom San Francisco Home Values: Test Prediction RMSE', size = 24)
    plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation=45, rotation_mode="anchor")
    fig.legend(bbox_to_anchor = (0.85, 0.86))
    plt.savefig(f'{top}/images/{bedrooms}_bdrm_RMSE.png')

def run_forecast(df_dict, model_best_df):
    forecast_dict = {}

    for zipcode, df in df_dict.items():

        zipcode = zipcode[-5:]
        sari_mod = SARIMAX(df.dropna(),
                           order=model_best_df.loc[zipcode].param,
                           seasonal_order=model_best_df.loc[zipcode].param_seasonal,
                           enforce_stationarity=False,
                           enforce_invertibility=False).fit()

        forecast = sari_mod.forecast(steps=12)
        forecast_dict[zipcode] = forecast
        fig, ax = plt.subplots()
        ax.plot(df.index, df.value, label='Historical')
        ax.plot(forecast, label='Forecast')
        ax.set_title(
            f'1-Bedroom San Francisco {zipcode} Home Values: 1 Year Forecast\nusing SARIMAX{model_best_df.loc[zipcode].param}x{model_best_df.loc[zipcode].param_seasonal}')
        plt.legend()
        plt.savefig(f'{top}/images/1_bdrm_forecast_{zipcode}.png')
    return forecast_dict

def create_final_df(df_dict, forecast_dict, bedrooms):
    final_dict = {'zipcode': list(forecast_dict.keys()),
                  'current_value': [df.iloc[-1].values[0] for df in list(df_dict.values())],
                  'forecasted_value': [df.iloc[-1] for df in list(forecast_dict.values())]
                  }
    final_df = pd.DataFrame(final_dict)
    final_df['percent_change'] = round(100*(final_df.forecasted_value - final_df.current_value )/final_df.current_value,2)
    final_sorted_df = final_df.sort_values('percent_change', axis = 0)
    final_sorted_df.set_index('zipcode', inplace=True)
    final_sorted_df.to_csv(f'data/{bedrooms}_bdrm_final_forecasts.csv')
    return final_sorted_df

def visualize_forecasts(df, forecast_df, bedrooms):
    fig, ax = plt.subplots(figsize=(20,12))
    ax.set_title(f'{bedrooms}-Bedroom Home Values in San Franciso by Zip Code: Forecast', size=24)
    sns.lineplot(data=df, x=df.date, y=df.value, ax=ax,
        hue='zipcode', style='zipcode', label = 'Historical')
    sns.lineplot(data=forecast_df, x=forecast_df.index, y=forecast_df.value,
        ax=ax, hue='zipcode', style='zipcode', label='Forecast')
    ax.set_xlabel('Year', size=20)
    ax.set_ylabel('Home Value (USD)', size=20)
    ax.set_xlim(pd.Timestamp('1996'), pd.Timestamp('2022-05-31'))
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.set_yticks(np.linspace(1e5,1.5e6,15))
    ax.set_ylim((1e5, 1.5e6))
    ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.savefig(f'{top}/images/{bedrooms}_bdrm_home_values_forecast.png')

def visualize_results(df1, df2):
    fig, ax = plt.subplots(2, 1, figsize = (12,16))
    ax[0].bar(x=df1.index, height=df1.percent_change)
    ax[0].set_title('Percent Change of 1-Bedroom Home Values in San Francisco', size=24)
    ax[0].set_xlabel('Zip Code', size=18)
    ax[0].set_ylabel('Percent Change after 1 Year (%)', size=18)
    # ax[1].bar(x=sf_1_sorted_df.index, height=sf_2_sorted_df.loc[sf_1_sorted_df.index].percent_change)
    ax[1].bar(x=df2.index, height=df2.percent_change)
    ax[1].set_title('Percent Change of 2-Bedroom Home Values in San Francisco', size=24)
    ax[1].set_xlabel('Zip Code', size=18)
    ax[1].set_ylabel('Percent Change after 1 Year (%)', size=18)
    plt.setp(ax[0].xaxis.get_majorticklabels(), ha="right", rotation=45, rotation_mode="anchor")
    plt.setp(ax[1].xaxis.get_majorticklabels(), ha="right", rotation=45, rotation_mode="anchor")
    fig.tight_layout(pad=2.0)
    plt.savefig(f'{top}/images/final_forecasts.png')

def best_3_zipcodes(sorted_df, bedrooms):
    print(f'The zipcodes with the greatest projected growth in mid-tier {bedrooms}-bedroom home values are:\n{sorted_df.iloc[-3]}\n {sorted_df.iloc[-2]}\n {sorted_df.iloc[-1]}')

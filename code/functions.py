import os
from random import random
import pandas as pd
from pandas.tseries.offsets import CustomBusinessMonthBegin, BDay
from pandas.tseries.holiday import *
import matplotlib
import numpy as np
import csv
import itertools
import pickle as pkl

import time
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas_market_calendars as mcal

import pmdarima as pm
from pmdarima import pipeline
import prophet
from sklearn.metrics import mean_squared_error as mse
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import confusion_matrix, plot_confusion_matrix,\
    precision_score, recall_score, accuracy_score, f1_score, log_loss,\
    roc_curve, roc_auc_score, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV,\
    cross_val_score, RandomizedSearchCV, StratifiedKFold

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
munits.registry[date] = converter
munits.registry[datetime] = converter

font = {'size'   : 12}
matplotlib.rc('font', **font)

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',25)

from pathlib import Path
TOP = Path(__file__ + '../../..').resolve()
# columns for Alpha Vantage output
AV_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'volume']
YF_COLUMNS = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
YF_F_COLUMNS = ['f_open', 'f_high', 'f_low', 'f_close', 'f_adj_close', 'f_volume']
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

NYSE = mcal.get_calendar('NYSE')
CBD = NYSE.holidays()

print(f'Functions loaded from {TOP}/data.')



################################################################################

def clear():
    os.system( 'cls' )

def round_sig_figs(x, p):
    '''
    https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy/63272943
    '''
    x=np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

class CustomUSHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Years Day', month=1, day=1, observance=nearest_workday),
        Holiday('GFordMourning', year=2007, month=1, day=2),
        USMartinLutherKingJr,
        Holiday('Washington\'s Birthday', month=2, day=15, offset=DateOffset(weekday=MO(1))),
        GoodFriday,
        USMemorialDay,
        Holiday('RWReaganMourning', year=2004, month=6, day=11),
        Holiday('July 4th', month=7, day=4, observance=nearest_workday),
        Holiday('Labor Day', month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('September11_1', year=2001, month=9, day=11),
        Holiday('September11_2', year=2001, month=9, day=12),
        Holiday('September11_3', year=2001, month=9, day=13),
        Holiday('September11_4', year=2001, month=9, day=14),
        Holiday('SandyHurricane_1', year=2012, month=10, day=29),
        Holiday('SandyHurricane_2', year=2012, month=10, day=30),
        USThanksgivingDay,
        Holiday('GWBushMourning', year=2018, month=12, day=5),
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
    ]


# def get_quandl_time_series(years, symbol, freq):
#     df = quandl.get("USTREASURY/YIELD")
#     df_all = df.sort_index().truncate(before=spy_df_all.index[0], after=spy_df_all.index[-1])
#     df_10Y = df_all.sort_index().truncate(before=spy_df_10Y.index[0])
#     df_5Y = df_10Y.sort_index().truncate(before=spy_df_5Y.index[0])
#     df_3Y = df_5Y.sort_index().truncate(before=spy_df_3Y.index[0])


def get_yf_time_series(yf, years, symbol, freq, fut=False):
    for year in years:
        start_year = pd.Timestamp.today().year - year
        year = f'{year}Y'
        start_date = pd.to_datetime(f'{start_year}-05-01')
        if start_date < pd.to_datetime(f'1993-01-29'):
            start_date = pd.to_datetime(f'1993-01-29')
            year = 'All'
        start_file = start_date.date() if freq.is_on_offset(start_date) else (start_date+freq).date()
        if pd.Timestamp.now() > (pd.Timestamp.today().date() + pd.offsets.Hour(16)):
            end_date = pd.Timestamp.today()
        else:
            end_date = pd.Timestamp.today().date() - freq + pd.offsets.Hour(16)
        end_file = end_date.date()

        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        df.index = df.index.rename('date')
        if fut:
            df.columns = YF_F_COLUMNS
        else:
            df.columns = YF_COLUMNS
        df.to_csv(f'{TOP}/data/{symbol}_{year}_CBD_{start_file}_{end_file}.csv')

def load_yf_time_series(yf, year, symbol, freq):
    start_year = pd.Timestamp.today().year - year
    year = f'{year}Y'
    start_date = pd.to_datetime(f'{start_year}-05-01')
    if start_date < pd.to_datetime(f'1993-01-29'):
        start_date = pd.to_datetime(f'1993-01-29')
        year = 'All'
    start_file = start_date.date() if freq.is_on_offset(start_date) else (start_date+freq).date()

    if pd.Timestamp.now() > (pd.Timestamp.today().date() + pd.offsets.Hour(16)):
        end_date = pd.Timestamp.today()
    else:
        end_date = pd.Timestamp.today().date() - freq + pd.offsets.Hour(16)
    end_file = end_date.date()

    df = pd.read_csv(f'../data/{symbol}_{year}_CBD_{start_file}_{end_file}.csv', index_col='date')
    df.index = pd.to_datetime(df.index)

    return df

def set_up_calendar_index(df, cal):
    start = df.index[0]
    end = df.index[-1]
    dates = cal.valid_days(start_date=start, end_date=end, tz='America/New_York')
    df.index = dates
    df.index.name = 'date'

    return df

def csv_read(csv_filepath, data):
    '''
    Given a target filepath and read next line.
    '''
    with open(csv_filepath, newline='') as csvfile:
#     fieldnames = ['first_name', 'last_name']
        reader = csv.reader(csvfile)
        line = next(reader)
    return line
    # print(fieldnames)

def csv_create(csv_filepath, headers):
    with open(csv_filepath, 'w', newline = '') as csvfile:
#         data_fields = parsed_results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames = headers)
        writer.writeheader()
    return data_fields

def check_mod_score_exists(csv_filepath, data_df, verbose=1):
    '''
    Given a target filepath and data as a dict,
    check if the model already exists and has been scored.
    '''
    # print(csv_filepath)
    if not os.path.exists(csv_filepath):
        # csv_create(csv_filepath)
        # csv_append(csv_filepath, data)
        return False
    else:
        file_df = pd.read_csv(csv_filepath, index_col='Model')
        # reset index just in case file was tampered
        file_df = file_df.reset_index(drop=True)
        file_df.index.name = 'Model'
        file_df.to_csv(csv_filepath)
        for index, row in file_df.iterrows():
            # print('Next model: \n', row[0:9]) if verbose else None
            if row[0:10].equals(data_df.iloc[0,0:10]):
                print('Model found in file. ', end='') if verbose else None
                if row[10]:
                    print('Already scored. ', end='') if verbose else None
                    return -1, file_df
                else:
                    print('Not yet scored. ', end='') if verbose else None
                    print(f'Index is {index}')
                    return index, file_df
        # else:
        print('Model not already present in file. ', end='') if verbose else None
        return -2, file_df

def __get_conf_ints_pc(conf_ints, data):
    conf_int_range = [int[1] - int[0] for int in conf_ints]
    cont_int_pc = 100*conf_int_range/np.mean(data)
    return conf_int_pc

def csv_write_data(csv_filepath, data_df, verbose=1):
    '''
    Given a target filepath and data_df as a dict,
    write or append to CSV file. Replaces unscored model with scored data_df.
    '''
    if not os.path.exists(csv_filepath):    # file does not exist
        # csv_create(csv_filepath)
        # csv_append(csv_filepath, data)
        data_df.to_csv(csv_filepath)
    else:                                   # file exists
        add_score = False
        if data_df.iloc[0,10]:              # data_df contains a score
            add_score = True
            print('Adding a new score to file... ', end='') if verbose else None
        else:
            print('Adding model params to file... ', end='') if verbose else None

        index, file_df = check_mod_score_exists(csv_filepath, data_df, verbose=verbose)
        if index == -1:                     # model found, already scored
            print('Nothing to do. ', end='') if verbose else None
            return index
        elif index == -2:                   # model not found
            print(f'Index = {index}. ') if verbose>1 else None
            if add_score:                   # have score to add
                print('Appending model scores to file. ', end='') if verbose else None
                print(data_df) if verbose>1 else None
            else:                           # only have model params to add
                print('Appending model params to file. ', end='') if verbose else None
                print(data_df) if verbose>1 else None
            # csv_append(csv_filepath, data_df.reset_index().to_dict('records'))
            file_df = file_df.append(data_df).drop_duplicates(keep='last')
            file_df = file_df.reset_index(drop=True)
            file_df.index.name = 'Model'
            # print('Appended: \n', file_df)
            file_df.to_csv(csv_filepath)
            return index

        if add_score:                       # model found, not yet scored, add score
            print(f'Adding scores to model, updating line {index}. ', end='') if verbose else None
            print(data_df) if verbose>1 else None
            file_df.iloc[index] = data_df.iloc[0]
            file_df.to_csv(csv_filepath)
            return index

        print(f'Model found at line {index}, nothing to do. ', end='') if verbose else None
        return
        # file_df = pd.read_csv(csv_filepath)
        # for index, row in file_df.iterrows():
        #     print('Model: \n', row[0:9]) if verbose else None
        #     if row[0:9].equals(data_df.iloc[0,0:9]):
        #         print('Model found.') if verbose else None
        #         if row[9]:
        #             print('Already scored.') if verbose else None
        #             return
        #         else:
        #             file_df.iloc[index] = data_df
        #             file_df.to_csv(csv_filepath)
        #             return

        # model not found
        # print('Model not found, appending: \n', data_df) if verbose else None
        # csv_append(csv_filepath, data_df)
        # return
                    # file_df.drop(index=index)
                    # file_df.append(data_df, index=index)
                    #

def csv_append(csv_filepath, data_dict):
    '''
    Given a target filepath and data as a list of dicts,
    append to CSV file (without Headers)
    '''
    # your code to open the csv file, concat the current data, and save the data.
    with open(csv_filepath, 'a', newline = '') as csvfile:
#         data_fields = ['id', 'name', 'is_closed', 'review_count', 'zip_code, 'rating', 'price]
        data_fields = data_dict[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames = data_fields)
        writer.writerows(data_dict)

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

def equidate_ax(fig, ax, dates, fmt="%Y-%m-%d", label="Date"):
    """
    https://stackoverflow.com/questions/1273472/how-to-skip-empty-dates-weekends-in-a-financial-matplotlib-python-graph
    Sets all relevant parameters for an equidistant date-x-axis.
    Tick Locators are not affected (set automatically)

    Args:
        fig: pyplot.figure instance
        ax: pyplot.axis instance (target axis)
        dates: iterable of datetime.date or datetime.datetime instances
        fmt: Display format of dates
        label: x-axis label
    Returns:
        None
    """
    N = len(dates)
    def format_date(index, pos):
        index = np.clip(int(index + 0.5), 0, N - 1)
        return dates[index].strftime(fmt)
    ax.xaxis.set_major_formatter(FuncFormatter(format_date))
    ax.set_xlabel(label, size = 18)
    fig.autofmt_xdate()

def plot_4_time_series(data):
    '''
    Plot the SPY, SPY Futures, 10Y Treasury Yield, and Fed Funds Curve
    '''
    colors = ['b','c','g','r']
    labels = ['SPY 3Y Close', 'SPY Futures 3Y Close', '10Y Treasury Yield', 'Fed Funds Rate']
    ylabels = ['SPY 3Y Close (USD)', 'SPY Futures 3Y Close (USD)', '10Y Treasury Yield %', 'Fed Funds Rate %']
    alpha = [0.9, 1, 1, 1]
    spy_range = (225, 425)
    yield_range = (0, 4)
    num_ticks = 9
    y_ranges = [spy_range, [10*x for x in spy_range], yield_range, yield_range]
    tick_params = dict(size=4, width=1.5, labelsize=16)
    fig, ax1 = plt.subplots(figsize=(20,15))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    axes = [ax1, ax2, ax3, ax4]
    for n, ax in enumerate(axes):
        if n==1:
            ax.spines['left'].set_position(("axes", -0.06))
            ax.spines['left'].set_edgecolor(colors[n])
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
        if n>1:
            ax.spines['right'].set_position(("axes", 1 + 0.04*(n-2)))
            ax.spines['right'].set_edgecolor(colors[n])
        data[n].plot(ax=ax, color=colors[n], alpha=alpha[n], label=labels[n])
        y_ticks = np.linspace(y_ranges[n][0], y_ranges[n][1], num_ticks)
        pad = (y_ranges[n][1] - y_ranges[n][0]) / (num_ticks-1)/5
        y_lim = (y_ranges[n][0]-pad, y_ranges[n][1]+pad)
        ax.set_yticklabels(y_ticks)
        ax.set_yticks(y_ticks)
        ax.set_ylim(y_lim)
        ax.tick_params(axis='y', colors=colors[n], **tick_params)

    ax1.set_xlabel('Date', size=24)
    ax1.tick_params(axis='x', **tick_params)
    fig.suptitle(', '.join(labels), size=32)
    fig.subplots_adjust(top=0.92)
    fig.legend(loc=(0.139, 0.8007), prop={"size":16})
    plt.savefig(f'{TOP}/images/SPY_3Y_Comparison_Graph.png')

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
    plt.savefig(f'{TOP}/images/{symbol}_{timeframe}_{freq}_ACF.png')

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
    dc_df = pd.DataFrame({"Observed": dc_obs, "Trend": dc_trend,
                            "Seasonal": dc_seas, "Residual": dc_resid})
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
        ax.set_ylabel(dc_df.iloc[:, i].name, size=20)
        # if i != 2:
        #     ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation=45, rotation_mode="anchor")

    decomp_fig.suptitle(
        f'Seasonal Decomposition of {symbol} Time Series\n{timeframe}, Seasonality = 1 Year', fontsize=30)
    decomp_fig.tight_layout()
    decomp_fig.subplots_adjust(top=0.9)
    plt.savefig(f'{TOP}/images/{symbol}_{timeframe}_{freq}_seasonal_decomp.png')


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
            param_list.append(param)
            param_seasonal_list.append(param_seasonal)
            aic = mod.aic
            aic_list.append(aic)
    return param_list, param_seasonal_list, aic_list

def model_stats(features, model, model_type, X_test, y_test, binary = False):
    '''
    Taking in a list of columns, a model, an X matrix, a y array, predicts
    labels and outputs model performance metrics.
    '''
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    print('Classifier: ', model_type)
    print('Num features: ', features.size)
    print('Model score: ', model.score(X_test, y_test))
    print('Accuracy score: ', accuracy_score(y_test, y_pred))
    print('Model F1 (micro): ', f1_score(y_test, y_pred, average='micro'))
    print('Model F1 (macro): ', f1_score(y_test, y_pred, average='macro'))
    print('Model F1 (weighted): ', f1_score(y_test, y_pred, average='weighted'))
    print('Cross validation score: ', cross_val_score(model, X_test, y_test, cv=5) )
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    # ax.set_ylabel(f'{model_type} Confusion Matrix', fontdict = {'fontsize': 12})
    if binary == False:
        macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                      average="macro")
        weighted_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo",
                                             average="weighted")
        macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                          average="macro")
        weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr",
                                             average="weighted")
        print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
              "(weighted by prevalence)"
              .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
        print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
              "(weighted by prevalence)"
              .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
        fig, ax = plt.subplots(figsize = [6,5])
        plot_confusion_matrix(model, X_test, y_test, ax = ax)
        ax.set_title(f'{model_type} Confusion Matrix', fontdict = {'fontsize': 14})
    elif binary == True:
        fig, ax = plt.subplots(2, 1, figsize = [6,10])
        plot_confusion_matrix(model, X_test, y_test, ax = ax[0])
        ax[0].set_title(f'{model_type} Confusion Matrix', fontdict = {'fontsize': 14})
        plot_roc_curve(model, X_test, y_test, ax = ax[1])
        ax[1].set_title(f'{model_type} Receiver Operating Characteristic Curve', fontdict = {'fontsize': 14})
    return

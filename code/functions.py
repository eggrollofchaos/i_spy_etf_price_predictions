from tqdm.notebook import trange, tqdm
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
from warnings import catch_warnings
from warnings import filterwarnings

import time
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas_market_calendars as mcal
import mplfinance as mpl

import pmdarima as pm
from pmdarima import pipeline
import prophet
from sklearn.metrics import mean_squared_error as mse
import scipy.stats as stats
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

# font = {'size'   : 16}
font = {
        'family' : 'sans-serif',
        'sans-serif' : 'Tahoma', # Verdana
        'weight' : 'normal',
        'size'   : '16'}
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
# years = mdates.YearLocator()
# years_fmt = mdates.DateFormatter('%Y')
# months = mdates.MonthLocator()
# months_fmt = mdates.DateFormatter('%B %Y')
# days = mdates.DayLocator()
# days_fmt = mdates.DateFormatter('%B %d, %Y')
# hours = mdates.HourLocator()
# hours_fmt = mdates.DateFormatter('%B %d, %Y %H:00')

NYSE = mcal.get_calendar('NYSE')
CBD = NYSE.holidays()

# print(f'Functions.py loaded from {TOP}/data.')



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

def get_quandl_time_series(quandl, year, symbol, freq):
    start_date, start_file, end_date, end_file, year = __ts_dates(year, freq)
    df = quandl.get(symbol)
    if symbol.find('TREASURY') != -1:
        df = df['10 YR']
        symbol = 'TSY'
        df = df.asfreq(freq).interpolate()
    else:
        df = df.asfreq(freq)
    if type(df) == pd.DataFrame:
        df = df.Value
    df.index.name = 'date'
    slash = symbol.find('/')
    symbol = symbol[slash+1:].lower()
    df.name = symbol
    df = df.sort_index().truncate(before=start_date, after=end_date)
    df.to_csv(f'{TOP}/data/{symbol.upper()}_{year}_CBD_{start_file}_{end_file}.csv')

    return df

def get_most_recent_trading_day(freq=CBD):
    now = pd.Timestamp.today()
    last = now if (freq.is_on_offset(now) and (now > (now.date() + pd.offsets.Hour(17)))) else \
        (now.date()-freq+pd.offsets.Hour(17))
    return last

def __ts_dates(year, freq=CBD):
    today = pd.Timestamp.today()
    start_year = today.year - year
    year = f'{year}Y'
    start_date = pd.to_datetime(f'{start_year}-05-01')
    if start_date < pd.to_datetime(f'1993-01-29'):
        start_date = pd.to_datetime(f'1993-01-29')
        year = 'All'
    start_date = start_date if freq.is_on_offset(start_date) else start_date+freq
    start_file = start_date.date()
    end_date = get_most_recent_trading_day(freq)
    # if freq.is_on_offset(today) and (pd.Timestamp.now() > (today.date() + pd.offsets.Hour(17))):
    #     # if today is a trading day and it's after 5pm (just to be safe)
    #     end_date = pd.Timestamp.now()
    # else:
        # set to 5pm on previous trading day
        # end_date = today.date() - freq + pd.offsets.Hour(17)
    end_file = end_date.date()
    return start_date, start_file, end_date, end_file, year

def get_yf_time_series(yf, years, symbol, freq=CBD, fut=False):
    for year in years:
        # today = pd.Timestamp.today()
        # start_year = today.year - year
        # year = f'{year}Y'
        # start_date = pd.to_datetime(f'{start_year}-05-01')
        # if start_date < pd.to_datetime(f'1993-01-29'):
        #     start_date = pd.to_datetime(f'1993-01-29')
        #     year = 'All'
        # start_file = start_date.date() if freq.is_on_offset(start_date) else (start_date+freq).date()
        # if (pd.Timestamp.now() > (today.date() + pd.offsets.Hour(16))):
        #     end_date = today
        # else:
        #     end_date = today.date() - freq + pd.offsets.Hour(16)
        # end_file = end_date.date()
        start_date, start_file, end_date, end_file, year = __ts_dates(year, freq)

        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        df.index = df.index.rename('date')
        if fut:
            df.columns = YF_F_COLUMNS
        else:
            df.columns = YF_COLUMNS
        df.to_csv(f'{TOP}/data/{symbol}_{year}_CBD_{start_file}_{end_file}.csv')

def load_yf_time_series(yf, year, symbol, freq=CBD, impute=True):
    # start_year = today.year - year
    # year = f'{year}Y'
    # start_date = pd.to_datetime(f'{start_year}-05-01')
    # if start_date < pd.to_datetime(f'1993-01-29'):
    #     start_date = pd.to_datetime(f'1993-01-29')
    #     year = 'All'
    # start_file = start_date.date() if freq.is_on_offset(start_date) else (start_date+freq).date()
    #
    # if pd.Timestamp.now() > (today.date() + pd.offsets.Hour(16)):
    #     end_date = today
    # else:
    #     end_date = today.date() - freq + pd.offsets.Hour(16)
    # end_file = end_date.date()
    start_date, start_file, end_date, end_file, year = __ts_dates(year, freq)

    df = pd.read_csv(f'{TOP}/data/{symbol}_{year}_CBD_{start_file}_{end_file}.csv', index_col='date')
    df.index = pd.to_datetime(df.index)
    df = df.asfreq(freq)
    if impute == True:
        df = df.interpolate()

    return df

def get_y_lim(series):
    max_value = max(series)
    min_value = min(series)
    if abs(max_value) - abs(min_value):
        upper, order = round_second_sigfig_5(max(series), 'up')
        lower, order = round_second_sigfig_5(min(series), 'down', order=order, limit=0)
    else:
        lower, order = round_second_sigfig_5(min(series), 'down', limit=0)
        upper, order = round_second_sigfig_5(max(series), 'up', order=order)
    return lower, upper

# def get_y_max(number):
#     '''
#     Helper function to round a large number to the second significant digit,
#     useful for setting the max y on a plot without knowledge of the y values.
#     '''
#     exp = np.floor(np.log10(number))
#     ten_exp = 10**exp
#     mantissa = number/ten_exp
#     numeral = round(mantissa%1,1)
#     mod_0_5 = numeral % 0.5
#     numeral += 0.5 - mod_0_5
#     # if mod_0_5 >= 0.25:
#     #     numeral += 0.5 - mod_0_5
#     # else:
#         # numeral -= mod_0_5
#     base = np.floor(mantissa)*ten_exp
#     add = numeral*ten_exp
#     return int(base + add)

def round_second_sigfig_5(number, direction, order=None, limit=None):
    '''
    Helper function to round a large (in absolute terms) number to the
    second significant digit in terms of 5, either upward or downard, useful
    for setting the min or max y on a plot without knowledge of the y values.
    '''
    pos = True
    if order:
        exp = order
    else:
        if number < 0:
            exp = np.floor(np.log10(-number))
        else:
            exp = np.floor(np.log10(number))
    ten_exp = 10**exp
    mantissa = number/ten_exp               # will be negative if negative
    numeral = round(mantissa%1,1)
    mod_0_5 = numeral % 0.5
    if direction == 'up':
        numeral += 0.5 - mod_0_5
    elif direction == 'down':
        numeral -= mod_0_5

    base = np.floor(mantissa)*ten_exp
    add = numeral*ten_exp
    result = int(base + add)

    if direction == 'up':
        limit = limit if limit else -np.inf
        return max(result, limit), exp

    if direction == 'down':
        limit = limit if limit else np.inf
        return min(result, limit), exp

def set_up_calendar_index(df, cal):
    start = df.index[0]
    end = df.index[-1]
    dates = cal.valid_days(start_date=start, end_date=end, tz='America/New_York')
    df.index = dates
    df.index.name = 'date'

    return df

def get_conf_ints_pc(conf_ints, y_hat):
    '''
    Return confidence interval as a percentage of the mean of the predictions
    '''
    conf_int_range = [int[1] - int[0] for int in conf_ints]
    avg_conf_in = np.mean(conf_int_range)
    cont_ints_pc = 100*conf_int_range/np.mean(y_hat)
    return conf_ints_pc

def pickle_data(data, pkl_filepath):
    '''
    Helper function for pickling data.
    '''
    if os.path.exists(pkl_filepath):
        print('File exists, overwriting.')
    pkl_out = open(pkl_filepath,'wb')
    pkl.dump(data, pkl_out)
    pkl_out.close()
    print(f'Saved to {pkl_filepath}.')

def unpickle_data(pkl_filepath):
    '''
    Helper function for unpickling data.
    '''
    if not os.path.exists(pkl_filepath):
        print('File does not exist.')
        return
    pkl_out = open(pkl_filepath,'rb')
    data = pkl.load(pkl_out)
    pkl_out.close()
    return data

def __option_expiry_offset(date_df):
    '''
    Helper function to check if a row with date as column is an options expiry date by comparing against pandas.offset.
    `date` : a date with weekmask of Monday, Wednesday, or Friday
    Rules:
    If Monday is a holiday, expiry will be Tuesday.
    If Wednesday is a holiday, expiry will be Tuesday.
    If Friday is a holoiday, expiry will be Thursday.
    '''
    conditions = [
        (date_df['on_offset'] == 0) & (date_df['date'].dt.day_of_week == 0),
        (date_df['on_offset'] == 0) & (date_df['date'].dt.day_of_week == 2),
        (date_df['on_offset'] == 0) & (date_df['date'].dt.day_of_week == 4),
        (date_df['on_offset'] == 1),
    ]
    choices = [
         1,          # add a day if Monday
        -1,          # subtract a day if Wednesday
        -1,          # subtract a day if Friday
         0           # do nothing
    ]
    series = np.select(conditions, choices, default = 0)
    return series

def __is_on_offset(row, freq=CBD):
    '''
    Function for checking through a pandas series if a row with date as column
    is on the specified frequency offset. Use with pandas apply().
    row : data row (iterable)
    '''
    if freq.is_on_offset(row['date']):
        return True
    else:
        return False

def create_option_expiry_df(date_index=None, N=0, func='historical', freq=CBD):
    '''
    Create an option expiries DataFrame starting with Mon/Wed/Fri weekmask, adjusted
    for freq offset, moves dates such that expiries that would fall on Holidays
    get their propert treatment, and then creates a full time series as a binary
    variable.
    Either `end` or `N` should be specified, not both.

    Holiday expiry rules:
    If Monday is a holiday, expiry will be Tuesday.
    If Wednesday is a holiday, expiry will be Tuesday.
    If Friday is a holiday, expiry will be Thursday.
    '''
    try:
        assert(func in ['forecast', 'historical']), "Parameter `func` must be 'forecast' or 'to_today'."
        assert((func=='forecast' and not type(date_index)==pd.DatetimeIndex and N) or (func=='historical' and type(date_index)==pd.DatetimeIndex and not N)), "For `forecast`, must specify 'N' days without `date_index`; for building `historical` dates, must specify `date_index` without 'N'."
    except AssertionError as e:
        print(e)
        raise

    MWF = pd.offsets.CustomBusinessDay(weekmask = 'Mon Wed Fri')
    today = pd.Timestamp.today()
    with catch_warnings():
        filterwarnings("ignore")
        last_CBD_date = get_most_recent_trading_day(freq).date()

    if func == 'forecast':
        start = last_CBD_date + freq
        with catch_warnings():
            filterwarnings("ignore")
            end = last_CBD_date + N*freq
    elif func == 'historical':
        start = date_index[0]
        end = date_index[-1]

    opt_dates = pd.date_range(start=start, end=end, freq=MWF)
    n = opt_dates.size

    opt_dates_df = pd.DataFrame(zip(opt_dates, np.zeros(n), np.zeros(n), np.ones(n)),
        columns=['date', 'on_offset', 'to_offset', 'is_option_expiry'])
    opt_dates_df['on_offset'] = opt_dates_df.apply(__is_on_offset, args=([freq]), axis=1)
    opt_dates_df['to_offset'] = __option_expiry_offset(opt_dates_df)
    with catch_warnings():
        filterwarnings("ignore")
        opt_dates_df['date'] = opt_dates_df['date'] + CBD*opt_dates_df['to_offset']
    opt_dates_df.drop(['on_offset', 'to_offset'], axis=1, inplace=True)
    opt_dates_df.drop_duplicates('date', ignore_index=True, inplace=True)
    # opt_dates_df.set_index('date', inplace=True)

    all_dates = pd.date_range(start=start, end=end, freq=freq)
    all_dates_df = pd.DataFrame(zip(all_dates, np.zeros(all_dates.size)),
        columns=['date', 'is_option_expiry'])
    # all_dates_df.set_index('date', inplace=True)
    all_dates_df = opt_dates_df.append(all_dates_df)
    all_dates_df.drop_duplicates('date', keep='first', ignore_index=True, inplace=True)
    all_dates_df['is_option_expiry'] = pd.to_numeric(all_dates_df['is_option_expiry'], downcast='integer')
    all_dates_df.set_index('date', inplace=True)
    all_dates_df.sort_index(inplace=True)

    return all_dates_df

# def __is_option_expiry(row, freq=CBD):
#     '''
#     Check if a row with date as column is an options expiry date by comparing against pandas.offset.
#     `date` : a date with weekmask of Monday, Wednesday, or Friday
#     Rules:
#     If Monday is a holiday, expiry will be Tuesday.
#     If Wednesday is a holiday, expiry will be Tuesday.
#     If Friday is a holoiday, expiry will be Thursday.
#     '''
#     if freq.is_on_offset(row['date']):
#         return 1
#     else:
#         return 0

# def csv_read(csv_filepath, data):
#     '''
#     Given a target filepath and read next line.
#     '''
#     with open(csv_filepath, newline='') as csvfile:
# #     fieldnames = ['first_name', 'last_name']
#         reader = csv.reader(csvfile)
#         line = next(reader)
#     return line
#     # print(fieldnames)

# def csv_create(csv_filepath, headers):
#     with open(csv_filepath, 'w', newline = '') as csvfile:
# #         data_fields = parsed_results[0].keys()
#         writer = csv.DictWriter(csvfile, fieldnames = headers)
#         writer.writeheader()
#     return data_fields

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
        # print(file_df['Fourier_m'].dtypes, end=' ') if verbose else None
        # file_df['Fourier_m'] = pd.to_numeric(file_df['Fourier_m'], downcast='integer')
        # file_df['Fourier_k'] = pd.to_numeric(file_df['Fourier_k'], downcast='integer')
        # file_df['Fourier_m'] = file_df['Fourier_k'].astype('object')
        # file_df['Fourier_k'] = file_df['Fourier_m'].astype('object')
        # print(file_df['Fourier_m'].dtypes, end=' ') if verbose else None
        file_df = file_df.reset_index(drop=True)
        file_df.index.name = 'Model'
        file_df.to_csv(csv_filepath)
        scored_col_index = file_df.columns.get_loc('Scored')
        for index, row in file_df.iterrows():
            # print('Next model: \n', row[0:9]) if verbose else None
            if row[0:scored_col_index].equals(data_df.iloc[0,0:scored_col_index]):
                print('Model found in file. ', end='') if verbose else None
                if row[scored_col_index]:
                    print('Already scored. ', end='') if verbose else None
                    return -1, file_df
                else:
                    print('Not yet scored. ', end='') if verbose else None
                    print(f'Index is {index}.') if verbose else None
                    return index, file_df
        # else:
        print('Model not already present in file. ', end='') if verbose else None
        return -2, file_df

def csv_write_data(csv_filepath, data_df, verbose=1):
    '''
    Given a target filepath and data_df as a dict,
    write or append to CSV file. Replaces unscored model with scored data_df.
    '''
    if not os.path.exists(csv_filepath):        # file does not exist
        # csv_create(csv_filepath)
        # csv_append(csv_filepath, data)
        data_df.to_csv(csv_filepath)

    else:                                       # file exists
        add_score = False
        scored_col_index = data_df.columns.get_loc('Scored')

        if data_df.iloc[0,scored_col_index]:    # data_df contains a score
            add_score = True
            print('Attempting to add a new score to file... ', end='') if verbose else None
        else:
            print('Attempting to add new model params to file... ', end='') if verbose else None

        index, file_df = check_mod_score_exists(csv_filepath, data_df, verbose=verbose)

        if index == -1:                         # model found, already scored
            print('Nothing to do. ', end='') if verbose else None
            return index

        elif index == -2:                       # model not found
            print(f'Index = {index}. ') if verbose>1 else None
            if add_score:                       # have score to add
                print('Appending model scores to file. ', end='') if verbose else None
                print(data_df) if verbose>1 else None
            else:                               # only have model params to add
                print('Appending model params to file. ', end='') if verbose else None
                print(data_df) if verbose>1 else None
            # csv_append(csv_filepath, data_df.reset_index().to_dict('records'))
            # print(data_df['Fourier_m'].dtypes, end=' ') if verbose else None
            # print(data_df['Fourier_m'].dtypes)
            file_df = file_df.append(data_df).drop_duplicates(keep='last')
            # file_df['Fourier_m'] = pd.to_numeric(file_df['Fourier_m'], downcast='integer')
            # file_df['Fourier_k'] = pd.to_numeric(file_df['Fourier_k'], downcast='integer')
            # print(file_df['Fourier_m'].dtypes, end=' ') if verbose else None
            file_df = file_df.reset_index(drop=True)
            # print(file_df['Fourier_m'].dtypes)
            file_df.index.name = 'Model'
            # print(file_df['Fourier_m'].dtypes)
            # print('Appended: \n', file_df)
            file_df.to_csv(csv_filepath)
            print() if verbose else None
            return index

        if add_score:                           # model found, not yet scored, add score
            print(f'Adding scores to model, updating line {index}.') if verbose else None
            print(data_df) if verbose>1 else None
            file_df.iloc[index] = data_df.iloc[0]
            file_df.to_csv(csv_filepath)
            return index

        else:                                   # model found, not yet scored, no score to add
            print(f'Model found at line {index}, nothing to do. ', end='') if verbose else None
            return

# def csv_append(csv_filepath, data_dict):
#     '''
#     Given a target filepath and data as a list of dicts,
#     append to CSV file (without Headers)
#     '''
#     # your code to open the csv file, concat the current data, and save the data.
#     with open(csv_filepath, 'a', newline = '') as csvfile:
# #         data_fields = ['id', 'name', 'is_closed', 'review_count', 'zip_code, 'rating', 'price]
#         data_fields = data_dict[0].keys()
#         writer = csv.DictWriter(csvfile, fieldnames = data_fields)
#         writer.writerows(data_dict)

def calc_buy_hold_profit(ohlc_df, shares=20000, start_date=None):
    if start_date==None:
        start_date=ohlc_df.index[0]
    # cost_basis = shares*ohlc_df['close'][ohlc_df.index==pd.to_datetime(start_date)]
    cost_basis = shares*ohlc_df['close'][start_date]
    final_portfolio_value = shares*ohlc_df['close'][-1]
    total_profit = final_portfolio_value - cost_basis
    total_profit_pc = 100*total_profit/cost_basis

    bh_profit_df = pd.DataFrame(index=ohlc_df.index)
    bh_profit_df['portfolio_value'] = ohlc_df['close']*shares
    bh_profit_df['eod_profit'] = bh_profit_df['portfolio_value'] - bh_profit_df['portfolio_value'][0]
    bh_profit_df['eod_profit_pc'] = 100*bh_profit_df['eod_profit']/bh_profit_df['portfolio_value'][0]
    # plot_profit(shares, start_date, ohlc_df, bh_profit_df)
    # print(f'Final portfolio value is ${"{:,.2f}".format(final_portfolio_value)}.')
    print(f'Starting portfolio value was ${cost_basis:,.2f}.')
    print(f'Final portfolio value is ${final_portfolio_value:,.2f}.')
    print(f'Total profit is ${total_profit:,.2f}, which is a {total_profit_pc:,.2f}% profit.')
    return total_profit, total_profit_pc, bh_profit_df

def plot_profits(shares, bh_profit_df, mod_profit_df, close_df, conf_ints, GS=True, start_date=None):
# def plot_profits(shares, spy_bh_vs_mod_data, start_date):
    # bh_profit_df = spy_bh_vs_mod_data[0]
    # mod_profit_df = spy_bh_vs_mod_data[1]
    # close_df = spy_bh_vs_mod_data[2]
    # conf_ints = spy_bh_vs_mod_data[3]
    data = [bh_profit_df.eod_profit, mod_profit_df.eod_profit, close_df, conf_ints]
    if not start_date:
        start_date = close_df.index[0]

    # bh_y_min = get_y_lim(bh_profit_df.eod_profit)[0]
    # bh_y_max = get_y_lim(bh_profit_df.eod_profit)[1]
    # mod_y_min = get_y_lim(mod_profit_df.eod_profit)[0]
    # mod_y_max = get_y_lim(mod_profit_df.eod_profit)[1]
    bh_y_min = get_y_lim(data[0])[0]
    bh_y_max = get_y_lim(data[0])[1]
    mod_y_min = get_y_lim(data[1])[0]
    mod_y_max = get_y_lim(data[1])[1]
    profit_y_min = min(bh_y_min, mod_y_min)
    profit_y_max = max(bh_y_max, mod_y_max)
    spy_y_min = get_y_lim(list(zip(*data[3]))[0])[0]
    spy_y_max = get_y_lim(list(zip(*data[3]))[1])[1]

    if GS == True:
        colors = ['b','magenta','g','orange']
        GS_str = ' GridSearch'
        GS_file_str = '_GS'
    else:
        colors = ['b','c','g','orange']
        GS_str = {}
        GS_file_str = {}
    labels = ['`Buy and Hold` Strategy', f'`I SPY`{GS_str} Model Strategy', 'SPY Close Price', 'Model Confidence Intervals']
    ylabels = ['Profit (USD)', '', 'Price (USD)', '']
    alpha = [0.9, 1, 1, 0.3]
    numticks = 11

    y_tick_formatter = ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    # y_tick_locator = ticker.LinearLocator(numticks=numticks)
    tick_params = dict(size=4, width=1.5, labelsize=16)

    fig, ax0 = plt.subplots(figsize=(20,16))
    ax1 = ax0.twinx()
    ax2 = ax0.twinx()
    ax3 = ax0.twinx()
    axes = [ax0, ax1, ax2, ax3]

    for n, ax in enumerate(axes):
        if n==0:
            ax.get_yaxis().set_major_formatter(y_tick_formatter)
            ax.set_ylabel('Profit (USD)', size=20)
        if n<=1:
            ax.set_ylim(profit_y_min, profit_y_max)
        if n>=2:
            ax.set_ylim(spy_y_min, spy_y_max)
        if n<=2:
            data[n].plot(ax=ax, color=colors[n], alpha=alpha[n], label=labels[n], x_compat=True)
        if n==3:
            conf_int = np.asarray(conf_ints)
            ax.fill_between(bh_profit_df.index,
                conf_int[:, 0], conf_int[:, 1],
                alpha=alpha[n], color=colors[n], animated=True,
                label=labels[n])
            ax.set_ylabel('Close Price (USD)', size=20)
        ax.tick_params(axis='y', **tick_params)
        # ax.get_yaxis().set_major_locator(y_tick_locator)

    ax1.yaxis.set_visible(False)
    ax0.tick_params(axis='x', **tick_params)
    ax0.set_xlim(close_df.index[0], close_df.index[-1])
    ax0.xaxis.get_offset_text().set_size(20)
    ax0.set_xlabel('Date', size=20)
    ax0.set_title(f'{shares:d} shares of SPY purchased @ price of ${close_df[0]:.2f} on {start_date.date()}.', size=20)
    fig.suptitle(f'SPY - Comparison of `Buy and Hold` vs `I SPY`{GS_str} Model Strategy', size=24)
    fig.subplots_adjust(top=0.93)
    fig.legend(loc=(0.105, 0.81), prop={"size":16})
    plt.savefig(f'{TOP}/images/SPY_Profit_Graph{GS_file_str}.png')

class Gridsearch_Calc_Profit:
    '''
    Run GridSearchCV on model profit historical simulation by tuning parameters
    standard deviation `z` and limit price offset percent `lim`.
    '''
    def __init__(self, ts, model, ohlc_df, exog_hist_df, shares=1000,
        steps=10, z_min=0.5, z_max=2.5, lim_min=0, lim_max=1,
        start_date=None, verbose=2):
        self.ts = ts
        self.model = model
        self.ohlc_df = ohlc_df
        self.exog_hist_df = exog_hist_df
        self.shares = shares
        self.steps = steps
        self.z_min = z_min
        self.z_max = z_max
        self.lim_min = lim_min
        self.lim_max = lim_max
        self.start_date = start_date
        self.verbose = verbose

        columns = ['z', 'Limit_Offset_pc', 'Final_Market_Value', 'Total_Profit', 'Total_Profit_pc']
        self.GS_mod_profit_df = pd.DataFrame(columns=columns)
        self.GS_mod_profit_df.index.name = 'Strategy'
        self.z_list = np.linspace(z_min, z_max, steps).round(3)
        self.lim_list = np.linspace(lim_min, lim_max, steps).round(3)
        self.num_sims = len(self.z_list)*len(self.lim_list)

    def __GS_calc_z_loop(self, z, verbose=0):
        # mod_profit_dict = {}
        if verbose:
            [self.__GS_calc_lim_loop(z, lim, verbose) for lim in tqdm(self.lim_list, desc='GridSearch `lim` Loop')]
        else:
            [self.__GS_calc_lim_loop(z, lim, verbose) for lim in self.lim_list]
        # print(mod_profit_dict)
        # return mod_profit_dict

    def __GS_calc_lim_loop(self, z, lim, verbose=0):
        # for z in tqdm(np.arange(z_min, z_max, 0.2), desc='GridSearch Loop: z'):
        #     for lim in tqdm(np.arange(lim_min, lim_max, 0.1), desc='GridSearch Loop: lim'):
        mod_profit_dict = {}
        if verbose:
            print(f'Parameter standard deviations (z) = {z}, Limit price offset percent = {lim}')
        y_hat, conf_ints, mod_profit, mod_profit_pc, mod_profit_df = \
            calc_model_profit(self.model, self.ohlc_df, self.exog_hist_df, shares=self.shares,
                z=z, limit_offset_pc=lim, verbose=verbose)
        # mod_profit_dict['Time Series'] = ts
        # mod_profit_dict['Model_Pipeline'] = model
        # mod_profit_dict['Exogenous_Variables'] = exog_hist_df.columns[1:].tolist()
        # mod_profit_dict['Initial_Shares'] = shares
        # mod_profit_dict['Cost_Basis'] = cost_basis
        mod_profit_dict['z'] = z
        mod_profit_dict['Limit_Offset_pc'] = lim
        mod_profit_dict['Final_Market_Value'] = mod_profit_df.eod_profit[-1]
        mod_profit_dict['Total_Profit'] = mod_profit
        mod_profit_dict['Total_Profit_pc'] = mod_profit_pc
        print('__________________________________________________________________') if verbose else None

        self.GS_mod_profit_df = self.GS_mod_profit_df.append(mod_profit_dict, ignore_index=True)
        # return
        # GS_mod_profit_df.append(mod_profit_dict, ignore_index=True)
        return mod_profit_dict

    # def run_gridsearch_calc_profit():
    def main(self):
        print(f'Running GridSearchCV on {self.ts} model trading strategy...')

        # columns = ['Time Series', 'Model_Pipeline', 'Exogenous_Variables', 'z',
        #     'Limit_Offset_pc', 'Initial_Shares', 'Cost_Basis', 'Final_Market_Value', 'Total_Profit', 'Total_Profit_pc']
        # columns = ['z', 'Limit_Offset_pc', 'Final_Market_Value', 'Total_Profit', 'Total_Profit_pc']
        # GS_mod_profit_df = pd.DataFrame(columns=columns)
        # GS_mod_profit_df.index.name = 'Strategy'
        # z_list = np.linspace(z_min, z_max, steps).round(3)
        # lim_list = np.linspace(lim_min, lim_max, steps).round(3)
        # num_sims = len(z_list)*len(lim_list)
        print(f'Running {self.num_sims} simulations using `z` in ({self.z_min}, {self.z_max}) and `lim` in ({self.lim_min}, {self.lim_max}).')
        # mod_profit_dict = []
        if self.verbose:
            [self.__GS_calc_z_loop(z, self.verbose) for z in tqdm(self.z_list, desc='GridSearch `z` Loop')]
        else:
            [self.__GS_calc_z_loop(z, self.verbose) for z in self.z_list]
        # print(mod_profit_dict)
        # GS_mod_profit_df = pd.DataFrame.from_records(mod_profit_dict)
        N = self.GS_mod_profit_df.shape[0]
        self.GS_mod_profit_df.insert(0, 'Cost_Basis', [self.shares*self.ohlc_df.close[0]]*N)
        self.GS_mod_profit_df.insert(0, 'Initial_Shares', [self.shares]*N)
        self.GS_mod_profit_df.insert(0, 'Exogenous_Variables', [self.exog_hist_df.columns[1:].tolist()]*N)
        self.GS_mod_profit_df.insert(0, 'Model_Pipeline', [self.model]*N)
        self.GS_mod_profit_df.insert(0, 'Time Series', [self.ts]*N)
            # for lim in tqdm(np.arange(lim_min, lim_max, 0.1), desc='GridSearch Loop: lim'):
            #     print(f'Parameter `z` = {z}')
            #     print(f'Limit price offset percent = {lim}')
            #     y_hat, conf_ints, mod_profit, mod_profit_pc, mod_profit_df = \
            #         calc_model_profit(model, ohlc_df, exog_hist_df, shares=shares,
            #             z=z, limit_offset_pc=lim, verbose=verbose)
            #     mod_profit_dict['Time Series'] = ts
            #     mod_profit_dict['Model_Pipeline'] = model
            #     mod_profit_dict['Exogenous_Variables'] = exog_hist_df.columns[1:].tolist()
            #     mod_profit_dict['z'] = z
            #     mod_profit_dict['Limit_Offset_pc'] = lim
            #     mod_profit_dict['Initial_Shares'] = shares
            #     mod_profit_dict['Cost_Basis'] = cost_basis
            #     mod_profit_dict['Final_Market_Value'] = mod_profit_df.eod_profit[-1]
            #     mod_profit_dict['Total_Profit'] = mod_profit
            #     mod_profit_dict['Total_Profit_pc'] = mod_profit_pc
            #     GS_mod_profit_df = GS_mod_profit_df.append(mod_profit_dict, ignore_index=True)
            #     print('__________________________________________________________________')
        ts_str = self.ts.replace(' ', '_').title()
        self.GS_mod_profit_df.to_csv(f'{TOP}/model_profit_CV/{ts_str}_Profit_CV.csv')
        pickle_data(self.GS_mod_profit_df, f'{TOP}/model_profit_CV/{ts_str}_Profit_CV.pkl')

        return self.GS_mod_profit_df

    if __name__ == "__main__":
        main()

def calc_model_profit(model, ohlc_df, exog_hist_df, shares=1000, z=1,
    limit_offset_pc=0.5, start_date=None, freq=CBD, verbose=1):
    '''
    Takes in model, fits it on the exogenous historical variables if not already,
    and iteratively calculates profit by implementing buy/sell signals using
    in-sample model prediction vs actual prices and summing the results.

    Order types will be Quote Triggered Limit Orders.

    Trigger price will be the low and high of the model prediction
    confidence intervals for BUY and SELL orders, respectively, and the
    limit price will be a percentage amount lower or higher, respectively.
    Pass this parameter as `limit_offset_pc`. A value of 0 indicates a
    Quote Triggered Market Order, or simply a Limit Order.

    This model assumes trading of fractional shares is allowed.

    Additionally, the # of standard deviations with which to calculate
    confidence intervals can be set as `z`.
    '''

    def __buy_shares(shares, cash, date, trades, ohlc_df, stop_price,
            limit_offset_pc, strat='all', verbose=0):
        limit_price = stop_price - limit_offset_pc*0.01*stop_price
        if ohlc_df.low[date] < limit_price:       # limit order will be filled
            if strat=='all':
                shares_to_buy = cash/limit_price
            cost = shares_to_buy*limit_price
            trades += 1
            if verbose>1:
                print('***********TRADE***********')
                print(f'   Buying max number of shares - {shares_to_buy:,.4f}.')
                print(f'   Cost = ${cost:,.2f}')
            return shares_to_buy, cost, trades
        else:
            return 0, 0, 0

    def __sell_shares(shares, cash, date, trades, ohlc_df, stop_price,
            limit_offset_pc=0.5, strat='all', verbose=0):
        limit_price = stop_price + limit_offset_pc*0.01*stop_price
        if ohlc_df.high[date] > limit_price:      # limit order will be filled
            if strat=='all':
                shares_to_sell = shares
            returns = shares_to_sell*limit_price
            trades += 1
            if verbose>1:
                print('***********TRADE***********')
                print(f'   Selling all {shares_to_sell:,.4f} shares.')
                print(f'   Revenue = ${returns:,.2f}')
            return shares_to_sell, returns, trades
        else:
            return 0, 0, 0

    def __update_portfolio_value(date, shares, cash, shares_delta, cash_delta):
        shares += shares_delta
        cash += cash_delta
        portfolio_value = shares*ohlc_df.low[date] + cash
        return shares, cash, portfolio_value

    print('Running `I SPY` model historical simulation...\n') if verbose else None
    if start_date == None:
        start_date = ohlc_df.index[0]
    cost_basis = shares*ohlc_df.close[start_date]

    # check if already fit
    try:
        model.named_steps['arima'].arroots()
    except Exception as e:
        if verbose>1:
            print('Fitting model to all exogenous variables... ', end='')
            print('Done.')
        model.fit(ohlc_df.close, exog_hist_df)

    cash = 0
    portfolio_value = cash + cost_basis
    trades = 0
    predictions = []
    conf_ints = []
    N = ohlc_df.shape[0]
    mod_profit_df = pd.DataFrame(index=ohlc_df.index)
    mod_profit_df['shares'] = np.zeros(N)
    mod_profit_df['cash'] = np.zeros(N)
    mod_profit_df['trade'] = np.zeros(N)
    mod_profit_df['portfolio_value'] = np.zeros(N)
    mod_profit_df['eod_profit'] = np.zeros(N)
    mod_profit_df['eod_profit_pc'] = np.zeros(N)

    print('Performing step-wise calculation of profit using time iterative model forecasts...\n') if verbose>1 else None
    # if verbose:
    #     [__calc_profit for index, current_date in enumerate(tqdm(ohlc_df.index, desc='Profit Calculation Loop'))]
    # else:
    #     [__calc_profit for index, current_date in enumerate(ohlc_df.index, desc='Profit Calculation Loop')]
    #
    # def __calc_profit():
    #     return

    for index, current_date in enumerate(tqdm(ohlc_df.index, desc='Profit Calculation Loop')):
        print(f'Analyzing Day {index+1} - {current_date.date()}:') if verbose>1 else None
        # a little animation to pass the time
        if not verbose:
            if index&1:
                print('>_', end='\r')
            else:
                print('> ', end='\r')
        traded = False
        if index == 0:
            # initialize as equal to the starting price to match dimensions
            predictions.append(ohlc_df.close[current_date])
            conf_ints.append([ohlc_df.close[current_date], ohlc_df.close[current_date]])
            print('Nothing to do! We only just bought all the dang shares.') if verbose>1 else None
        else:
            pred, conf_int = model.predict_in_sample(X=exog_hist_df[0:index+1],
                start=index, end=index, return_conf_int=True, alpha=(2-stats.norm.cdf(z)*2))
            predictions.append(pred[0])
            conf_ints.append(conf_int.tolist()[0])
            # print(len(y_hat))
            fcast = pred[0]
            fcast_low = conf_int[0][0]
            fcast_high = conf_int[0][1]
            if verbose>1:
                print('High  |   Low  |  Close | Predicted Close | Confidence Interval')
                print('%.2f  | %.2f | %.2f   | %.2f | %.2f - %.2f' % (ohlc_df.high[current_date], ohlc_df.high[current_date], ohlc_df.close[current_date], fcast, fcast_low, fcast_high))
                # print(f'{ohlc_df.high[current_date]:.2f}   {ohlc_df.low[current_date]:.2f}  {ohlc_df.close[current_date]:.2f}   | {fcast:.2f} | {fcast_low:.2f} - {fcast_high:.2f}')
            high_diff = ohlc_df.high[current_date] - fcast_high
            low_diff = ohlc_df.low[current_date] - fcast_low
            if shares>0 and high_diff>0:
                print('SPY high of day greater than top of confidence interval by %.2f' % high_diff) if verbose>1 else None
                shares_to_sell, cash_received, trades = __sell_shares(shares, cash,
                    current_date, trades, ohlc_df, stop_price=fcast_high,
                    limit_offset_pc=limit_offset_pc, strat='all', verbose=verbose)
                shares, cash, portfolio_value = __update_portfolio_value(current_date,
                    shares, cash, -shares_to_sell, cash_received)
                traded=True

            elif cash>0 and low_diff<0:
                print('SPY low of day less than bottom of confidence interval by %.2f' % low_diff) if verbose>1 else None
                shares_to_buy, cash_spent, trades = __buy_shares(shares, cash,
                    current_date, trades, ohlc_df, stop_price=fcast_low,
                    limit_offset_pc=limit_offset_pc, strat='all', verbose=verbose)
                shares, cash, portfolio_value = __update_portfolio_value(current_date,
                    shares, cash, shares_to_buy, -cash_spent)
                traded=True

            else:
                print('Price movement for today is within confidence interval.\n-----------Holding-----------') if verbose>1 else None
                shares, cash, portfolio_value = __update_portfolio_value(current_date, shares, cash, 0, 0)

        shares, cash, portfolio_value,

        mod_profit_df.loc[current_date,'shares'] = shares
        mod_profit_df.loc[current_date,'cash'] = cash
        mod_profit_df.loc[current_date,'trade'] = traded
        mod_profit_df.loc[current_date,'portfolio_value'] = portfolio_value
        mod_profit_df.loc[current_date,'eod_profit'] = portfolio_value - cost_basis
        mod_profit_df.loc[current_date,'eod_profit_pc'] = 100*(portfolio_value - cost_basis)/cost_basis

        if verbose>1:
            print('EOD portfolio snapshot:')
            print('Shares | Cash | Portfolio Value')
            print(f'{shares:,.4f} | ${cash:,.2f} | ${portfolio_value:,.2f} ')
            print('__________________________________________________________________')
        # time.sleep(5)
    total_profit = portfolio_value-cost_basis
    total_profit_pc = 100*(portfolio_value-cost_basis)/cost_basis
    if verbose>1:
        print('Done.')
        print(f'Made {trades} trades out of {N} trading days.')
        print(f'Starting portfolio value was ${cost_basis:,.2f}.')
        print(f'Final portfolio value is ${portfolio_value:,.2f}.')
    if verbose:
        print(f'Total profit is ${total_profit:,.2f}, which is a {total_profit_pc:,.2f}% profit.')

    return predictions, conf_ints, total_profit, total_profit_pc, mod_profit_df

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
    # fig.autofmt_xdate()

def plot_spy_fin(ohlc_data):
    fig, ax = plt.subplots(figsize=(16, 12))
    mpl.plot(ohlc_data, type='candle', style="yahoo", ax=ax)
    ax.set_ylabel('Price (USD)', size=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    fig.suptitle('SPY - 10 Year Performance', size=30)
    fig.subplots_adjust(top=0.92)

def plot_spy_fut_tsy_funds_time_series(data):
    '''
    Plot the SPY, SPY Futures, 10Y Treasury Yield, and Fed Funds Curve
    '''
    try:
        colors = ['b','c','g','r']
        labels = ['SPY Close', 'SPY Futures Close', '10Y Treasury Note Yield', 'Fed Funds Rate']
        ylabels = ['SPY Close (USD)', 'SPY Futures Close (USD)', 'Treasury Yield %', 'Fed Funds Rate %']
        alpha = [0.9, 1, 1, 1]
        spy_range = (100, 425)
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
            if n<=1:
                ax.spines['left'].set_position(("axes", 0-0.08*n))
                ax.spines['left'].set_edgecolor(colors[n])
                ax.yaxis.set_label_position("left")
                ax.yaxis.tick_left()
            if n>=2:
                ax.spines['right'].set_position(("axes", 1 + 0.04*(n-2)))
                ax.spines['right'].set_edgecolor(colors[n])
            # if n<=3:
            data[n].plot(ax=ax, color=colors[n], alpha=alpha[n], label=labels[n], x_compat=True)
            y_ticks = np.linspace(y_ranges[n][0], y_ranges[n][1], num_ticks)
            pad = (y_ranges[n][1] - y_ranges[n][0]) / (num_ticks-1)/5
            y_lim = (y_ranges[n][0]-pad, y_ranges[n][1]+pad)
            ax.set_yticklabels(y_ticks)
            ax.set_yticks(y_ticks)
            ax.set_ylim(y_lim)
            ax.tick_params(axis='y', colors=colors[n], **tick_params)

        ax1.set_ylabel('Price (USD)', size=20, rotation='horizontal')
        ax1.yaxis.set_label_coords(-0.1,1.005)
        ax3.set_ylabel('%', size=20, rotation='horizontal')
        ax3.yaxis.set_label_coords(1.04,-0.005)
        ax1.set_xlim(data[0].index[0], data[0].index[-1])
        ax1.set_xlabel('Year', size=24)
        ax1.tick_params(axis='x', **tick_params)
        fig.suptitle(', '.join(labels), size=32)
        fig.subplots_adjust(top=0.92)
        fig.legend(loc=(0.139, 0.8007), prop={"size":16})
        plt.savefig(f'{TOP}/images/SPY_3Y_Comparison_Graph.png')
    except Exception:
        raise

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

def plot_pacf_with_diff(df, symbol, n, period, freq, lags, diffs=1, alpha=0.05):
    timeframe = f'{n} {period.title()}'
    pacf_fig, ax = plt.subplots(1, diffs+1, figsize=((diffs+1)*6, 6))
    pacf_fig.suptitle(f'Partial Autocorrelations of {symbol} Time Series: {timeframe}, Frequency = {freq}', fontsize=18)
    plot_pacf(df, ax=ax[0], lags=lags, alpha=alpha)
    ax[0].set_title('Undifferenced PACF', size=14)
    ax[0].set_xlabel('Lags', size=14)
    ax[0].set_ylabel('PACF', size=14)
    plot_pacf(df.diff().dropna(), ax=ax[1], lags=lags, alpha=alpha)
    ax[1].set_title('Differenced PACF', size=14)
    ax[1].set_xlabel('Lags', size=14)
    ax[1].set_ylabel('PACF', size=14)
    if diffs == 2:
        plot_pacf(df.diff().diff().dropna(), ax=ax[2], lags=lags, alpha=alpha)
        ax[2].set_title('Twice-Differenced PACF', size=14)
        ax[2].set_xlabel('Lags', size=14)
        ax[2].set_ylabel('ACF', size=14)
    pacf_fig.tight_layout()
    pacf_fig.subplots_adjust(top=0.9)
    plt.savefig(f'{TOP}/images/{symbol}_{timeframe}_{freq}_PACF.png')

def plot_acf_with_diff(df, symbol, n, period, freq, lags, diffs=1, alpha=0.05):
    timeframe = f'{n} {period.title()}'
    acf_fig, ax = plt.subplots(1, diffs+1, figsize=((diffs+1)*6, 6))
    acf_fig.suptitle(f'Autocorrelations of {symbol} Time Series: {timeframe}, Frequency = {freq}', fontsize=18)
    plot_acf(df, ax=ax[0], lags=lags, alpha=alpha)
    ax[0].set_title('Undifferenced ACF', size=14)
    ax[0].set_xlabel('Lags', size=14)
    ax[0].set_ylabel('ACF', size=14)
    plot_acf(df.diff().dropna(), ax=ax[1], lags=lags, alpha=alpha)
    ax[1].set_title('Once-Differenced ACF', size=14)
    ax[1].set_xlabel('Lags', size=14)
    ax[1].set_ylabel('ACF', size=14)
    if diffs == 2:
        plot_acf(df.diff().diff().dropna(), ax=ax[2], lags=lags, alpha=alpha)
        ax[2].set_title('Twice-Differenced ACF', size=14)
        ax[2].set_xlabel('Lags', size=14)
        ax[2].set_ylabel('ACF', size=14)
    acf_fig.tight_layout()
    acf_fig.subplots_adjust(top=0.9)
    plt.savefig(f'{TOP}/images/{symbol}_{timeframe}_{freq}_ACF.png')

def plot_seasonal_decomposition(df, symbol, n, period, freq, seas, seas_str):
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
        f'Seasonal Decomposition of {symbol} Time Series\n{timeframe}, Seasonality = {seas_str}', fontsize=30)
    decomp_fig.tight_layout()
    decomp_fig.subplots_adjust(top=0.9)

    seas_str = seas_str.replace(' ', '_')
    plt.savefig(f'{TOP}/images/{symbol}_{timeframe}_{freq}_seasonal_decomp_{seas_str}.png')


def train_test_split_data(df, train_size=80, verbose=0):
    if verbose==1:
        print('##### Train-Test Split #####')
        print(f'Using a {train_size}/{100-train_size} train-test split...')
    cutoff = round((train_size/100)*len(df))
    train_df = df[:cutoff]
    test_df = df[cutoff:]
    return train_df, test_df

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

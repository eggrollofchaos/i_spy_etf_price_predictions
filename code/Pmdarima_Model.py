from tqdm.notebook import trange, tqdm
import pandas as pd
import matplotlib
import numpy as np
# import csv
from itertools import product
from functools import reduce
import pickle as pkl
from warnings import catch_warnings
from warnings import filterwarnings

import time
import datetime
from multiprocessing import cpu_count, Pool
# from joblib import Parallel
# from joblib import delayed
from dateutil.relativedelta import relativedelta
import pandas_market_calendars as mcal
import mplfinance as mpl

import pmdarima as pm
from pmdarima import pipeline
from pmdarima.metrics import smape
from sklearn.metrics import mean_squared_error as mse
import prophet
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter
# plt.style.use('ggplot')
sns.set_theme(style="darkgrid")
# import matplotlib.dates as mdates
# import matplotlib.units as munits
# converter = mdates.ConciseDateConverter()
# munits.registry[np.datetime64] = converter
# munits.registry[datetime.date] = converter
# munits.registry[datetime.datetime] = converter

font = {'family' : 'sans-serif',
        'sans-serif' : 'Tahoma', # Verdana
        'weight' : 'normal',
        'size'   : '16'}
matplotlib.rc('font', **font)

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',25)

try:
    from code.functions import *
except Exception as e:
    from functions import *
from pathlib import Path
TOP = Path(__file__ + '../../..').resolve()

NYSE = mcal.get_calendar('NYSE')
CBD = NYSE.holidays()

# print(f'Pmdarima_Model.py loaded from {TOP}/data..')

class Pmdarima_Model:
    def __init__(self, df, data_name, n, periods, freq, train_size=80, trend='c', with_intercept='auto',
                order=(0,1,0), s_order=(0,0,0), seas=0, fit_seas=False, f_seas=252, k=4,
                estimate_diffs=False, impute=False, AA_d=None, AA_D=None,
                #max_d=2, max_p=2, max_q=2, max_D=2, max_P=2, max_Q=2,
                date=True, fourier=True, box=False, log=False, verbose=1):

        try:
            assert(type(df) in (pd.Series, pd.DataFrame)), "Data is not of type Pandas Series or DataFrame."
            assert(type(df.index) == (pd.DatetimeIndex)), "Data index is not of type Pandas DatetimeIndex."
        except AssertionError as e:
            print(e)
            print('Failed to load data.')
            raise

        # if d:
        #     try:
        #         assert(order[1] == d), "Variables d and d in order conflict."
        #     except AssertionError as e:
        #         print(e)
        #         print('Failed to initialize Class.')
        #         raise

        if type(df) == pd.Series:
            self.df = pd.DataFrame(df)
        else:
            self.df = df
        if impute:
            self.df = df.interpolate()
        self.hist_dates_df = pd.DataFrame(self.df.index, columns=['date'])
        self.train_size = train_size
        self.df_train, self.df_test = pm.model_selection.train_test_split(self.df,
            train_size = self.train_size/100)
        self.dates = df.index
        self.length = df.shape[0]
        self.data_name = data_name
        self.ts = data_name.replace(' ', '_')
        self.timeframe = f'{n} {periods.title()}'
        self.tf = f'{n}{periods[0].upper()}'
        self.freq = freq
        self.f = freq.split()[0] + freq.split()[1][0].upper()
        self.m = seas
        self.f_m = f_seas
        self.k = k
        self.estimate_diffs = estimate_diffs
        # self.arima_order = order
        self.p = order[0]
        self.d = order[1]
        self.q = order[2]
        self.fit_seas = fit_seas
        self.P = s_order[0]
        self.D = s_order[1]
        self.Q = s_order[2]
        self.t = trend
        self.n_diffs = AA_d
        self.ns_diffs = AA_D
        if self.estimate_diffs:
            self.__estimate_diffs()
        self.with_intercept = with_intercept
        # self.no_intercept = no_intercept
        self.mod_order = f'({self.p}, {self.d}, {self.q})[\'{self.t}\']'
        self.date = date
        self.fourier = fourier
        self.box = box
        self.log = log
        self.__train_test_split_dates()
        self.AA_best_params, self.AA_mod_pipe = self.__reset_mod_params()
        self.GS_best_params, self.GS_best_mod_pipe = self.__reset_mod_params()
        self.mod_params, self.mod_params_df, self.mod_pipe = self.__reset_mod_params('adhoc')
        self.y_hat = None
        self.conf_ints = None
        self.AIC = None
        self.RMSE = np.inf
        self.RMSE_pc = np.inf
        self.SMAPE = np.inf
        self.GS_first_mod = True
        self.mod_CV_filepath = f'{TOP}/model_CV_scores/{self.ts}_{self.tf}_{self.f}.csv'
        print('Successfully created instance of Class Pmdarima_Model.') if verbose else None

    def __estimate_diffs(self):
        '''
        Helper function for calculation of diffs to use if
        estimate_diffs=True is passed at class initialization.
        '''
        kpss_diffs = pm.arima.ndiffs(self.df_train, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = pm.arima.ndiffs(self.df_train, alpha=0.05, test='adf', max_d=6)
        self.n_diffs = max(adf_diffs, kpss_diffs)

        if self.fit_seas:
            ocsb_diffs = pm.arima.nsdiffs(self.df_train, m=self.m, test='ocsb', max_D=6)
            ch_diffs = pm.arima.nsdiffs(self.df_train, m=self.m, test='ch', max_D=6)
            self.ns_diffs = max(ocsb_diffs, ch_diffs)

    def __reset_mod_params(self, init=None):
        '''
        Helper function for intializing a human-readable model params string
        as passed at class intialization.
        '''
        if init: # for an adhoc run
            mod_params, mod_params_df, mod_pipe = self.__setup_mod_params(self.p, self.d, self.q,
                self.t, self.P, self.D, self.Q, self.m, self.with_intercept,
                self.f_m, self.k, self.date, self.fourier, self.box,
                self.log, func='adhoc', verbose=1)
            return mod_params, mod_params_df, mod_pipe
        else:
            mod_pipe = None
            mod_params = None

        return mod_params, mod_pipe

    @staticmethod
    def __unpickle_model(ts, tf, f, func='GS'):
        pkl_filepath = Pmdarima_Model.__get_pkl_filepath(ts, tf, f, func=func)
        print(f'Loading best model from {pkl_filepath}.')
        mod_file = open(pkl_filepath,'rb')
        mod_data = pkl.load(mod_file)
        mod_file.close()

        return mod_data

    @staticmethod
    def __get_pkl_filepath(ts, tf, f, func='GS'):
        # pkl_filepath = f'{TOP}/models/{self.ts}_{self.tf}_{self.f}_{func}_best_model.pkl'
        pkl_filepath = f'{TOP}/models/{ts}_{tf}_{f}_{func}_best_model.pkl'

        return pkl_filepath

    def __pickle_model(self, func='AA', verbose=1):
        '''
        Helper function for pickling a model along with its params as a
        human-readable string.
        '''
        def __pickle_it(params, pipe, params_df, scores, results, func_type='adhoc', verbose=1):
            mod_file = open(pkl_filepath,'wb')
            pkl.dump((params, pipe, params_df, scores, results), mod_file)
            # if func_type == 'AutoARIMA':
                # pkl.dump((self.AA_best_params, self.AA_mod_pipe, self.AA_best_mod_params_df, scores, results), mod_file)
            # elif func_type == 'GridSearchCV':
            #     pkl.dump((self.GS_best_params, self.GS_best_mod_pipe, self.GS_best_mod_params_df, scores, results), mod_file)
            # else: # func_type == 'adhoc'
            #     pkl.dump((self.mod_params, self.mod_pipe, self.mod_params_df, scores, results), mod_file)
            mod_file.close()

        scores = (self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE)
        results = (self.y_hat, self.conf_ints)
        if func == 'AA':
            func_type = 'AutoARIMA'
            params = self.AA_best_params
            pipe = self.AA_mod_pipe
            params_df = self.AA_best_mod_params_df
        elif func == 'GS':
            func_type = 'GridSearchCV'
            params = self.GS_best_params
            pipe = self.GS_best_mod_pipe
            params_df = self.GS_best_mod_params_df
        else: # func == 'adhoc':
            func_type = 'adhoc'
            params = self.mod_params
            pipe = self.mod_pipe
            params_df = self.mod_params_df

        # var = self.data_name.lower()
        # pkl_filepath = __get_pkl_filepath(func='GS')
        # f'{TOP}/models/{self.ts}_{self.tf}_{self.f}_{func}_best_model.pkl'
        pkl_filepath = Pmdarima_Model.__get_pkl_filepath(self.ts, self.tf, self.f, func=func)

        if os.path.exists(pkl_filepath):
            # mod_file = open("../models/TSY_10Y_Note_3Y_1D_GS_best_model.pkl",'rb')
            # mod_file = open(pkl_filepath,'r+b')
            # mod_data = pkl.load(mod_file)
            mod_data = Pmdarima_Model.__unpickle_model(self.ts, self.tf, self.f, func=func)
            try:
                if self.RMSE < mod_data[3][2]:
                    __pickle_it(params, pipe, params_df, scores, results, func_type, verbose)
                    print(f'Model outperforms existing best {func_type} model at {pkl_filepath}, overwriting.') if verbose else None
                else:
                    # mod_file.close()
                    print(f'Model did not outperform existing {func_type} model at {pkl_filepath}, not pickling model.') if verbose else None
                    return
            except IndexError:
                __pickle_it(params, pipe, params_df, scores, results, func_type, verbose)
                print('Model file contains missing data, overwriting.') if verbose else None

        else:
            mod_file = open(pkl_filepath,'wb')
            __pickle_it(params, pipe, params_df, scores, results, func_type, verbose)
            print(f'Saved best {func_type} model as {pkl_filepath}.') if verbose else None
            return

    def __split_df_dates(self, train, test):
        '''
        Helper function of splitting train and test sets into date variables
        as X and data variables as y.
        '''
        X_train = pd.DataFrame(train.index)
        y_train = train.values
        X_test = pd.DataFrame(test.index, index=range(X_train.size, self.length))
        y_test = test.values
        return X_train, y_train, X_test, y_test

    def __train_test_split_dates(self):
        '''
        Helper function for initializing the date split train vs test sets.
        '''
        self.X_train, self.y_train, self.X_test, self.y_test = self.__split_df_dates(self.df_train, self.df_test)
        # return self.X_train, self.y_train, self.X_test, self.y_test

    def __fit_predict(self, model, days_fc, new_dates, index_fc, hist_df, hist_dates_df, en_ex, new_dates_df=None, exog_df=None, verbose=1):
        # model.fit(self.df, hist_dates_df)
        '''
        Helper function for fitting a model on the full input DataFrame and
        running an out of sample prediction.
        For final predictions on endogenous variable, `hist_df` and `exog_df` must have 'date' as a column - function will convert if found as index instead.
        '''

        if en_ex == 'exog':
            model.fit(y=self.df, X=hist_dates_df)
            print('Successfully fit model on historical observations.') if verbose else None
            y_hat, conf_ints = model.predict(X=new_dates_df, return_conf_int=True)
            fc_df = pd.DataFrame(y_hat, index=index_fc, columns=self.df.columns)
            fc_date_df = pd.DataFrame(zip(new_dates, y_hat), index=index_fc, columns=['date', self.df.columns[0]])
            fc_date_df.set_index('date', inplace=True)
        elif en_ex == 'endo':
            if type(exog_df.index) == pd.DatetimeIndex:
                exog_df.reset_index(inplace=True)
            if type(hist_df.index) == pd.DatetimeIndex:
                hist_df.reset_index(inplace=True)
            model.fit(y=self.df, X=hist_df)
            print('Successfully fit model on historical observations.') if verbose else None

            y_hat, conf_ints = model.predict(X=exog_df, return_conf_int=True)
            # y_hat, conf_ints = self.__run_stepwise_fc(self.exog_df, model, verbose)

            # results = model.predict(n_periods=days_fc, X=exog_df, return_conf_int=True)
            fc_date_df = pd.DataFrame(zip(new_dates, y_hat), index=index_fc, columns=['date', self.df.columns[0]])
            fc_date_df.set_index('date', inplace=True)
            fc_df = fc_date_df

        self.df_with_fc = self.df.append(fc_date_df)
        print(f'Successfully forecasted {days_fc} days forward.') if verbose else None

        # fc_df = pd.DataFrame(zip(self.new_dates_df.date.values,y_hat), columns=['date','close'])
        return fc_df, y_hat, conf_ints
        # return fc_df, results
        # return results

    # @classmethod
    # def get_next_dates(cls, today, df_size, days):
    @staticmethod
    def __get_next_dates(today, df_size, days_fc, freq=CBD):
        '''
        Static method for getting new dates for out of sample predictions.
        Returns a list of Pandas Timestamps, a list of numerical indices extending
        the original numerical indices of the input DataFrame, and a DataFrame consisting
        of the two aforementioned lists.
        '''
        next_day = today + freq
        new_dates = pd.date_range(start=next_day, periods=days_fc, freq=freq)
        index_fc = range(df_size, df_size + days_fc)
        new_dates_df = pd.DataFrame(new_dates, index=index_fc, columns=['date'])
        return new_dates, index_fc, new_dates_df

    @classmethod
    def join_exog_data(cls, *args):
        '''
        Takes any number of DataFrames with matching indexes and performs a join.
        First DataFrame must be the dates_df. Number of observations in each must match.
        '''
        # try:
        #     assert(len(set(map(lambda df: df.shape, args))) == 1), "Input DataFrame shapes do not match."
        # except AssertionError as e:
        #     print(e)
        #     print('Failed to perform join.')
        #     raise
        # today = args[0].date.iloc[-1]
        # df_size = args[0].size
        # days =
        # index_fc, new_dates_df = cls.get_next_dates(today, df_size, days)
        # args = [new_dates_df, args]

        exog_cat_df = reduce(lambda left, right: pd.merge(left,right,left_index=True,right_index=True), [*args])

        return exog_cat_df

    def fit_model(self, model, func='AA'):
        '''
        Manually fit model on the date-split class variables X_train, y_train.
        '''
        # if func=='AA':
            # model.fit(self.df, self.hist_dates_df)
        # else:
        model.fit(self.y_train, self.X_train)

    def __setup_mod_params(self, p=0, d=None, q=0, t='c', P=0, D=0, Q=0, m=0,
        with_intercept='auto', f_m=None, k=None, date=True, fourier=True, box=False,
        log=False, func='AA', verbose=1):
        '''
        Helper function for creating a human-readable model params string and
        initializing a pipeline.
        '''
        arima_params = dict(
            trace=4*verbose,
            maxiter=200,
            suppress_warnings=True)

        if func == 'AA':
            arima_params['stepwise']    = True
            # arima_params['out_of_sample_size'] = round(self.length/5)
            # diffs_str = ''
            if d:
                arima_params['d']       = d
                # arima_params['max_p']   = 2
                # arima_params['start_p'] = 2
                # arima_params['max_q']   = 1
                # arima_params['start_q'] = 1
                # diffs_str += f', d={d}'
            if self.fit_seas:
                arima_params['seasonal']= True
                arima_params['m']       = m
                arima_params['max_p']   = 1
                arima_params['start_p'] = 0
                arima_params['max_q']   = 1
                arima_params['start_q'] = 0
                arima_params['max_P']   = 1
                arima_params['start_P'] = 0
                arima_params['max_Q']   = 1
                arima_params['start_Q'] = 0
                if D:
                    arima_params['D']   = D
                    # diffs_str += f', D={D}'
                # arima_params['start_params']=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
            else:
                arima_params['seasonal']=False
            # mod_params = f'AutoARIMA{diffs_str}'
            mod_params = 'AutoARIMA'
            arima_params['with_intercept'] = with_intercept
            if with_intercept == True:
                mod_params += ', Intercept'
            if with_intercept == 'auto':
                mod_params += ', Check Intercept'

        else: # GridSearch or adhoc model
            arima_order = (p, d, q)
            arima_params['order'] = arima_order
            arima_params['trend'] = t
            order = f'({p}, {d}, {q})[\'{t}\']'
            mod_params = f'ARIMA Order{order}'
            if with_intercept:
                arima_params['with_intercept'] = True
                mod_params += ', Intercept'
            else:
                arima_params['with_intercept'] = False
            # mod_params = mod_params.replace('()', order)

        pipe_params = []
        if date:
            date_feat = pm.preprocessing.DateFeaturizer(
                column_name="date",  # the name of the date feature in the X matrix
                with_day_of_week=True,
                with_day_of_month=True)
            _, X_train_feats = date_feat.fit_transform(self.y_train, self.X_train)
            pipe_params.append(('date', date_feat))
            mod_params += ', DateFeaturizer'
        if fourier:
            pipe_params.append(('fourier', pm.preprocessing.FourierFeaturizer(m=f_m, k=k)))
            mod_params += ', FourierFeaturizer'
        if box:
            pipe_params.append(('box', pm.preprocessing.BoxCoxEndogTransformer()))
            mod_params += ', BoxCoxEndogTransformer'
        if log:
            pipe_params.append(('log', pm.preprocessing.LogEndogTransformer()))
            mod_params += ', LogEndogTransformer'

        if func == 'AA':
            pipe_params.append(('arima', pm.arima.AutoARIMA(**arima_params)))
        else: # GridSearch or adhoc model
            pipe_params.append(('arima', pm.arima.ARIMA(**arima_params)))

        print(mod_params) if (verbose >= 1 and func == 'GS') else None
        print(pipe_params) if verbose >= 2 else None
        pipe = pipeline.Pipeline(pipe_params)

        mod_params_dict = {}
        mod_params_dict['ARIMA_Order'] = f'({p}, {d}, {q})'
        mod_params_dict['Diffs'] = d
        mod_params_dict['Mod_Order'] = p+q
        mod_params_dict['Trend'] = t
        mod_params_dict['Intercept'] = with_intercept
        mod_params_dict['Date'] = date
        mod_params_dict['Fourier'] = fourier
        # mod_params_dict['Fourier_m'] = int(f_m) if f_m else ''
        # mod_params_dict['Fourier_k'] = int(k) if k else ''
        mod_params_dict['Fourier_m'] = f_m
        mod_params_dict['Fourier_k'] = k
        mod_params_dict['BoxCox'] = box
        mod_params_dict['Log'] = log
        mod_params_dict['Scored'] = False
        mod_params_dict['AIC'] = None
        mod_params_dict['RMSE'] = None
        mod_params_dict['RMSE%'] = None # (100*RMSE/y_train.mean()))
        mod_params_dict['SMAPE'] = None
        mod_params_dict['CV_Time'] = None
        mod_params_df = pd.DataFrame(mod_params_dict, index=[0])
        # mod_params_dict['Fourier_m'] = mod_params_dict['Fourier_m'].map(int)
        # mod_params_dict['Fourier_k'] = mod_params_dict['Fourier_k'].map(int)
        # print(mod_params_df['Fourier_m'].dtypes, end=' ')
        # mod_params_df['Fourier_m'] = pd.to_numeric(mod_params_df['Fourier_m'], downcast='integer')
        # mod_params_df['Fourier_k'] = pd.to_numeric(mod_params_df['Fourier_k'], downcast='integer')
        # print(mod_params_df['Fourier_m'].dtypes, end=' ')
        mod_params_df.index.name = 'Model'
        return mod_params, mod_params_df, pipe

    def __run_stepwise_CV(self, model=None, func='AA', dynamic=False, verbose=1, debug=False):
        '''
        Heavily modified from https://github.com/alkaline-ml/pmdarima/issues/339
        '''
        def __forecast_one_step(date_df):
            fc, conf_int = model.predict(X=date_df, return_conf_int=True)
            return fc.tolist()[0], conf_int.tolist()[0]

        def __run_CV_loop(n, new_obs, verbose=0):  # should be called iteratively
            fc, conf = __forecast_one_step(date_df)
            forecasts.append(fc)
            conf_ints.append(conf)

            # update the existing model with a small number of MLE steps
            if dynamic:
                model.update([fc], date_df)
            elif not dynamic:
                model.update([new_obs], date_df)

            ## make a little animation
            if verbose:
                if n&1:
                    print('>_', end='\r')
                else:
                    print('> ', end='\r')
                # date = pd.DataFrame([X_test.iloc[0].date + CBD], index=[X_train.size]columns=['date'])
            date_df.iloc[0].date += CBD
            date_df.index += 1
            # return forecasts, conf_ints

        if not model:
            model = self.AA_mod_pipe
            print('No model specified, defaulting to AutoARIMA best model.') if verbose else None

        date_df = pd.DataFrame([self.X_test.iloc[0].date], index=[self.X_train.size], columns=['date'])
        # date = X_test.iloc[0].date
        forecasts = []
        conf_ints = []
        dynamic_str = ''
        if dynamic:
            dynamic_str = ' dynamically with forecasted values'
        if verbose:
            print(f'Iteratively making in-sample predictions on \'{self.data_name}\' Time Series and updating model{dynamic_str}, beginning with first index of y_test ...')
        self.start = time.time()

        if verbose:
            # results = [__run_CV_loop(n, new_obs) for n, new_obs in tqdm(self.y_test, desc='Step-Wise Prediction Loop')]
            [__run_CV_loop(n, new_obs, verbose=verbose) for n, new_obs in enumerate(tqdm(self.y_test, desc='Step-Wise Prediction Loop'))]
        else:
            # results = [__run_CV_loop(n, new_obs) for n, new_obs in self.y_test]
            [__run_CV_loop(n, new_obs, verbose=verbose) for n, new_obs in enumerate(self.y_test)]
        print() if verbose else None

        self.end = time.time()
        print('Done.')

        return forecasts, conf_ints

    def __run_stepwise_fc(self, exog_df, model, verbose=0):
        def __forecast_one_step(exog_df):
            fc, conf_int = model.predict(X=exog_df, return_conf_int=True)
            return fc.tolist()[0], conf_int[0].tolist()

        def __run_CV_loop(n, exog_row, verbose=0):  # should be called iteratively
            exog_df = pd.DataFrame(exog_row)
            fc, conf = __forecast_one_step(exog_df)
            forecasts.append(fc)
            conf_ints.append(conf)

            # update the existing model with a small number of MLE steps
            model.update([fc], X=exog_df)

            ## make a little animation
            if verbose:
                if n&1:
                    print('>_', end='\r')
                else:
                    print('> ', end='\r')
            return

        # check if model was passed
        if not model:
            model = self.GS_best_mod_pipe
            print('No model specified, defaulting to GridSearch best model.') if verbose else None

        forecasts = []
        conf_ints = []
        dynamic_str = ' dynamically with forecasted values'
        if verbose:
            print(f'Iteratively making out-of-sample predictions on \'{self.data_name}\' Time Series and updating model{dynamic_str}, beginning with first index of y_test ...')
        self.start = time.time()

        if verbose:
            [__run_CV_loop(n, exog_df[n:n+1], verbose=verbose) for n in trange(exog_df.shape[0], desc='Step-Wise Prediction Loop')]
        else:
            [__run_CV_loop(n, exog_df[n:n+1], verbose=verbose) for n in range(exog_df[0])]

        print() if verbose else None

        self.end = time.time()
        print('Done.')

        return forecasts, conf_ints

    def __GS_score_model(self, model, verbose=0, debug=False):
        print('________________________________________________________________________\n') if verbose else None
        print(f'Running step-wise cross-validation on model {self.GS_curr_mod_num} of {self.GS_mod_count}... ', end='')
        if verbose:
            print()
            print(model[0])
            # print(model[1])
            print(model[2])
        # model[1].fit(self.y_train, self.X_train)
        result = None
        try:
            self.fit_model(model[2])
        except Exception as e:
            self.GS_curr_mod_num += 1
            print(e) if verbose else None
            print('Skipping model due to error.')
            return
        # convert config to a key
        key = str(model[0])
        # print(key)
        # model[1].fit(y_train, X_train)
        # show all warnings and fail on exception if debugging
        if debug:
            # X_train, X_test, y_train, y_test, y_hat, conf_ints = self.__run_stepwise_CV(model[1])
            print('Running in debug mode.') if verbose else None
            y_hat, conf_ints = self.__run_stepwise_CV(model[2], verbose=1, debug=debug)
            result = self.__get_model_scores(self.y_test, y_hat, self.y_train, model=model[2], debug=debug)
        else:
            # one failure during model validation suggests an unstable config
            try:
                # never show warnings when grid searching, too noisy
                with catch_warnings():
                    filterwarnings("ignore")
                    # X_train, X_test, y_train, y_test, y_hat, conf_ints = self.__run_stepwise_CV(model[1])
                    y_hat, conf_ints = self.__run_stepwise_CV(model[2], verbose=verbose)
                    result = self.__get_model_scores(self.y_test, y_hat, self.y_train, model=model[2], debug=0)
            except Exception as e:
                print(e) if verbose else None
                print('Error in step-wise CV, skipping model.')
                result = None
                self.GS_curr_mod_num += 1
                return
                # raise
                # check for an interesting result
        if result:
            print('Successfully completed step-wise CV on model.') if verbose else None
            print('Model[%s]: AIC=%.3f | RMSE=%.3f | RMSE_pc=%.3f%% | SMAPE=%.3f%%' % (key, *result))
            if result[1] < self.RMSE:
                print('Updated class attributes for GridSearch performance.') if verbose else None
                self.AIC = result[0]
                self.RMSE = result[1]
                self.RMSE_pc = result[2]
                self.SMAPE = result[3]
                self.y_hat = y_hat
                self.conf_ints = conf_ints
                self.GS_best_params = key
                self.GS_best_mod_params_df = model[1]
                self.GS_best_mod_pipe = model[2]
                if self.GS_first_mod:
                    print('First viable model found, RMSE=%.3f' % result[1])
                    self.GS_first_mod = False
                else:
                    print('Next best model found, RMSE=%.3f' % result[1])
                # pickle model
                self.__pickle_model(func='GS', verbose=verbose)
            model[1]['Scored'].values[0] = True
            model[1]['AIC'].values[0] = '%.4f' % (result[0])
            model[1]['RMSE'].values[0] = '%.4f' % (result[1])
            model[1]['RMSE%'].values[0] = '%.4f' % (result[2])
            model[1]['SMAPE'].values[0] = '%.4f' % (result[3])
            model[1]['CV_Time'].values[0] = '%.4f' % (self.end-self.start)

            # replace existing line with new line + scores
            scored_col_index = model[1].columns.get_loc('Scored')
            for index, row in self.GS_all_mod_params_df.iterrows():
                if row[0:scored_col_index].equals(model[1].iloc[0,0:scored_col_index]):
                    self.GS_all_mod_params_df.iloc[index] = model[1]
                    break
            csv_write_data(self.mod_CV_filepath, model[1], verbose=verbose)
            # if not verbose and not debug:
            #     clear()
            self.GS_curr_mod_num += 1
            return (key, *result)

    def __gridsearch_CV(self, min_p=0, max_p=10, min_q=0, max_q=10, min_d=0, max_d=2,
        min_order=0, max_order=6, t_list=['n','c','t','ct'], with_intercept=False,
        f_m=None, k=None, date=True, fourier=True, box=False, log=False,
        max_models=0, verbose=0, debug=False, parallel=True):
        '''
        Heavily modified from https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
        '''
        def __GS_setup_params(min_p=0, max_p=10, min_q=0, max_q=10, min_d=0, max_d=2,
                min_order=0, max_order=6, t_list=['n','c','t','ct'], with_intercept=False,
                f_m=f_m, k=k, date=True, fourier=True, box=False, log=False,
                max_models=0, verbose=0):
        # def __GS_setup_params():
            print('Building list of models to test.') if verbose else None

            mod_count, to_run_count, models = \
                __GS_build_grid(min_p, max_p, min_q, max_q, min_d, max_d,
                    min_order, max_order, t_list, with_intercept, f_m, k,
                    date, fourier, box, log, max_models, verbose)
            self.GS_mod_count = to_run_count
            if to_run_count > 0:
                print(f'Based on parameters given, 0 models built out of {mod_count} in grid.')
            else:
                print(f'Finished building list of {to_run_count} models.') if verbose else None
            return models
            # print(models) if debug else None

        def __GS_build_grid(min_p, max_p, min_q, max_q, min_d, max_d,
                min_order, max_order, t_list, with_intercept, f_m, k,
                date, fourier, box, log, max_models, verbose):

            columns = ['ARIMA_Order', 'Diffs', 'Mod_Order', 'Trend', 'Intercept', 'Date', 'Fourier',
                        'Fourier_m', 'Fourier_k', 'BoxCox', 'Log',
                        'Scored', 'AIC', 'RMSE', 'RMSE%', 'SMAPE', 'CV_Time']
            self.GS_all_mod_params_df = pd.DataFrame(columns=columns)
            self.GS_all_mod_params_df.index.name = 'Model'

            if with_intercept == 'auto':
                inter_iter = [True, False]
            else:
                inter_iter = [with_intercept]
            if date == 'auto':
                date = list(set([True, False]))[::-1]
            else:
                date = [date]
            if fourier == 'auto':
                fourier = list(set([True, False]))[::-1]
            else:
                fourier = [fourier]
            if box == 'auto':
                box = list(set([True, False]))[::-1]
            else:
                box = [box]
            if log == 'auto':
                log = list(set([True, False]))[::-1]
            else:
                log = [log]

            feats_iter = list(itertools.product(date, fourier, box, log))
            models = []
            mod_count = 0
            to_run_count = 0
            mod_params_dict = {}
            for d in range(min_d, max_d+1):
                # print(max_models, to_run_count)
                mod_order = 0
                for p in range(min_p, max_p+1):
                    for q in range(min_q, max_q+1):
                        mod_order = p+q
                        if mod_order < min_order:
                            continue
                        if mod_order > max_order:
                            continue
                        for intercept in inter_iter:
                            t_list_mod = t_list
                            if intercept:
                                if 'n' in t_list:
                                    t_list_mod.remove('n')
                            else:
                                t_list_mod = ['n']
                            for t in t_list_mod:
                                for date, fourier, box, log in feats_iter:
                                    if max_models and to_run_count>=max_models:
                                        print(f'Reached {max_models} models, ending grid-building.')
                                        return mod_count, to_run_count, models
                                    f_m_mod = int(f_m)
                                    k_mod = int(k)
                                    if not fourier:
                                        f_m_mod = None
                                        k_mod = None
                                    mod_params, mod_params_df, pipe = self.__setup_mod_params(
                                        p=p, d=d, q=q, t=t, with_intercept=intercept,
                                        f_m=f_m_mod, k=k_mod, date=date,
                                        fourier=fourier, box=box, log=log,
                                        func='GS', verbose=verbose)
                                    mod_count += 1
                                    # print(mod_params)
                                    # print(mod_params_df)
                                    index = csv_write_data(self.mod_CV_filepath, mod_params_df, verbose=verbose)
                                    if index == -1:  # model found and scored
                                        print('Not adding model to test grid.') if verbose else None
                                        continue
                                    print('Added a model to test grid.') if verbose else None
                                    # or if model found but not scored, no need to write anything
                                    # otherwise, if model not found, write model params to csv
                                    self.GS_all_mod_params_df = self.GS_all_mod_params_df.append(mod_params_df)
                                    self.GS_all_mod_params_df = self.GS_all_mod_params_df.reset_index(drop=True)
                                    self.GS_all_mod_params_df.index.name = 'Model'
                                    # print(self.GS_all_mod_params_df['Fourier_m'].dtypes)
                                    models.append((mod_params, mod_params_df, pipe))
                                    to_run_count += 1
            return mod_count, to_run_count, models

        def __GS_start(models, verbose=0, debug=False, parallel=True):

            print('Starting iterative search through all GridSearch params.') if verbose else None
            self.GS_curr_mod_num = 1
            scores = None
            # count = 0

            if parallel:
                # execute configs in parallel
                print('Running with multiprocessing enabled.')
                # executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
                # tasks = (delayed(self.__GS_score_model)(model, debug=debug) for model in tqdm(models, desc='Model Loop'))
                # # tasks = (delayed(self.__GS_score_model)(model, debug=debug) for model in models)
                # scores = executor(tasks)
                # workers = cpu_count()
                # pool = Pool(processes=cpu_count())
                # scores = pool.map(self.__GS_score_model(model, debug=debug), tqdm(models, desc='Model Loop'))
                with Pool(processes=cpu_count()) as pool:
                    # scores = pool.imap(self.__GS_score_model, (models, debug=debug))#, tqdm(models, desc='Model Loop'))
                    scores = pool.map(self.__GS_score_model, models) #, tqdm(models, desc='Model Loop'))
                    # scores = pool.map(self.__GS_score_model, models) #, tqdm(models, desc='Model Loop'))
            else:
                print('Running normally.')
                try:
                    if verbose:
                        scores = [self.__GS_score_model(model, debug=debug, verbose=verbose) for model in tqdm(models, desc='Model Loop')]
                    else:
                        scores = [self.__GS_score_model(model, debug=debug, verbose=verbose) for model in models]
                except Exception as e:
                    print(e) if verbose else None
                    print('Completed with some errors.') if verbose else None
            # scores = [self.__GS_score_model(model, debug=debug) for model in models]
            # remove empty results
            scores = [score for score in scores if score != None]
            # # sort configs by error, asc
            scores.sort(key=lambda tup: tup[2])
            return scores

        # if __name__ == '__main__':  # <- prevent RuntimeError for 'spawn'
        # models = __GS_setup_params(t_list=t_list, with_intercept=with_intercept,
        #     f_m=f_m, k=k, date=date,
        #     fourier=fourier, box=box, log=log, verbose=verbose)
        # models = __GS_setup_params()
        models = __GS_setup_params(min_p, max_p, min_q, max_q, min_d, max_d,
                min_order, max_order, t_list, with_intercept, f_m, k,
                date, fourier, box, log, max_models, verbose)
        if models:
            scores = __GS_start(models, debug=debug, verbose=verbose, parallel=parallel)
            try:
                assert(scores != None), "No valid scores returned."
            except AssertionError as e:
                print(e)
                raise

            return self.GS_best_mod_pipe, scores
        else:
            return None, None

    def __run_auto_pipeline(self, show_summary=False, return_conf_int=False, verbose=1):
        # display visualization of entire data set
        pm.tsdisplay(self.df, lag_max=60, title = f'{self.data_name} Time Series Visualization') \
            if show_summary else None

        AA_mod_params, AA_mod_params_df, AA_pipe = self.__setup_mod_params(d=self.n_diffs, D=self.ns_diffs,
            with_intercept=self.with_intercept, m=self.m, f_m=self.f_m, k=self.k,
            date=self.date, fourier=self.fourier, box=self.box, log=self.log,
            func='AA', verbose=verbose)

        if verbose:
            print(f'Parameters for AutoARIMA Pipeline: \n  {AA_mod_params}') if verbose else None
            print(AA_pipe, '\n')

        self.fit_model(AA_pipe, func='AA')

        best_arima = AA_pipe.named_steps['arima'].model_.order
        best_seas = AA_pipe.named_steps['arima'].model_.seasonal_order if self.fit_seas else ''
        AA_mod_params = AA_mod_params.replace('AutoARIMA', f'ARIMA{best_arima}{best_seas}')
        AA_mod_params_df['ARIMA_Order'].values[0] = best_arima
        with_intercept = AA_pipe.named_steps['arima'].model_.with_intercept
        trend = AA_pipe.named_steps['arima'].model_.trend
        AA_mod_params_df['Intercept'].values[0] = with_intercept
        if with_intercept:
            AA_mod_params = AA_mod_params.replace('Check ', '')
            self.with_intercept = True
            if trend:
                AA_mod_params = AA_mod_params.replace(AA_mod_params[AA_mod_params.rfind(')')], f')[\'{trend}\']')
                AA_mod_params_df['Trend'].values[0] = trend
                self.t = trend
            else:
                AA_mod_params = AA_mod_params.replace(AA_mod_params[AA_mod_params.rfind(')')], ')[\'c\']')
                AA_mod_params_df['Trend'].values[0] = 'c'
                self.t = 'c'
        else:
            AA_mod_params = AA_mod_params.replace(', Check Intercept', '')
            self.with_intercept = False
        self.AA_best_params = AA_mod_params
        self.AA_best_mod_params_df = AA_mod_params_df
        self.AA_mod_pipe = AA_pipe
        if self.fourier:
            self.f_m = AA_pipe.named_steps['fourier'].m
            self.k = AA_pipe.named_steps['fourier'].k

        if return_conf_int:
            self.y_hat, self.conf_ints = self.AA_mod_pipe.predict(X=self.X_test, return_conf_int=return_conf_int)
        elif not return_conf_int:
            self.y_hat = self.AA_mod_pipe.predict(X=self.X_test)
        # print("Test RMSE: %.3f" % mse(y_test, y_hat, squared=False))

        # return self.X_train, self.y_train, self.X_test, sself.y_test, self.y_hat, self.conf_ints

    def plot_test_predict(self, y_hat=None, conf_ints=True, ylabel=None, fin=True, func='AA'):
        '''
        Plot Test vs Predict with optional confidence intervals
        '''
        conf = ''
        if ylabel == None:
            ylabel = self.data_name
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.X_train, self.y_train, color='blue', alpha=0.5, label='Training Data')
        ax.plot(self.X_test, self.y_hat, color='green', marker=',', alpha=0.7, label='Predicted')
        ax.plot(self.X_test, self.y_test, color='magenta', alpha=0.3, label='Actual')
        if conf_ints:
            conf = '_Conf'
            conf_int = np.asarray(self.conf_ints)
            ax.fill_between(self.X_test.date,
                     conf_int[:, 0], conf_int[:, 1],
                     alpha=0.5, color='orange',
                     label="Confidence Intervals")
        ax.legend(loc='upper left', borderaxespad=0.5, prop={"size":16})
        fig.suptitle(f'{ylabel} Time Series, {self.timeframe}, Freq = {self.freq}: Test vs Predict with Confidence Intervals\n', size=26)
        ax.set_ylabel(ylabel, size=18)
        ax.set_xlabel(ax.get_xlabel(), size=18)
        tick_params = dict(size=4, width=1.5, labelsize=16)
        ax.tick_params(axis='y', **tick_params)
        ax.tick_params(axis='x', **tick_params)
        if func == 'AA':
            ax.set_title(f'AutoARIMA Best Parameters: {self.AA_best_params}', size=24)
            plt.savefig(f'{TOP}/images/AutoArima/{self.ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')
        elif func == 'GS':
            ax.set_title(f'GridSearch Best Parameters: {self.GS_best_params}', size=24)
            plt.savefig(f'{TOP}/images/GridSearch/{self.ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')
        else:
            ax.set_title(f'Parameters: {self.mod_params}', size=24)
            plt.savefig(f'{TOP}/images/{self.ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')

    def plot_forecast_conf(self, ohlc_df=None, ohlc_fc_df=None, hist_df=None, y_hat=None, y_hat_df=None, conf_ints=True,
            lookback=0, ylabel=None, fin=False, all_ohlc=False, func='GS'):
        '''
        Plot forecasts with optional confidence intervals. Can only be run after
        forecasts have been generated.
        '''
        days_fc = self.days_fc                              # number of days to forecast
        dates = self.df_with_fc.index                       # dates indices
        hist_ind = np.arange(self.df.shape[0])              # set numerical indices for historical data
        fc_ind = np.arange(self.df.shape[0], self.df.shape[0]+days_fc) # set numerical indices for forecast data
        if lookback:
            dates = dates[-lookback-days_fc:]               # get subset of dates
            hist_ind = np.arange(lookback)                  # update indices
            fc_ind = np.arange(lookback, lookback+days_fc)  # update indices
            num_months = np.floor(dates.shape[0]/21)        # update num_months
            numticks = int(num_months+1)                    # update numticks
            while numticks>21:                              # cap at 21
                numticks = int(np.ceil(num_months/2)+1)     # halve the number of ticks iteratively
        ylabel=self.data_name                               # get time series name for filename and plot
        df_with_fc = self.df_with_fc.loc[dates]             # truncate (if needed) and assign
        conf_filename = ''                                  # initialize conf string in filename
        conf_title = ''                                     # initialize conf string in fig title

        if fin:                                             # OHLC chart
            ohlc_df = ohlc_df[-lookback:]                   # historical OHLC
            all_ohlc_df = ohlc_df.append(ohlc_fc_df)        # for all OHLC, append forecasts
        else:
            hist_df = hist_df[-lookback:]                   # historical single variable plot

        # fig, ax = plt.subplots(figsize=(24, 16))
        fig, ax = plt.subplots(figsize=(20, 12))            # originally (24,16)
        # hist_ind, fc_ind = equidate_ax(fig, ax, df_with_fc.index.date, days_fc=days_fc)
        ax.set_xlim(0, fc_ind[-1])
        # print(hist_ind)
        # print(fc_ind)
        if fin:
            if not all_ohlc:
                mpl.plot(ohlc_df, type='candle', style="yahoo", ax=ax)
                # ax.plot(range(lookback, lookback+days_fc), y_hat, 'g.', markersize=10, alpha=0.7, label='Forecast')
                ax.plot(fc_ind, y_hat_df, 'g.', markersize=7.5, alpha=0.7, label='Forecast')
            if all_ohlc:
                mpl.plot(all_ohlc_df, type='candle', style="yahoo", ax=ax)
        else: ## not fin
            # ax.plot(hist_df[-lookback:], color='blue', alpha=0.5, label='Historical')
            # ax.plot(self.new_dates, y_hat, 'g.', markersize=10, alpha=0.7, label='Forecast')
            # ax.set_xlim(self.df_with_fc.index[-lookback-days_fc], self.df_with_fc.index[-1]+(lookback//20)*CBD)
            # hist_ind, fc_ind = equidate_ax(fig, ax, df_with_fc.index.date, days_fc=days_fc)
            sns.lineplot(x=hist_ind, y=hist_df, color='blue', alpha=0.5, label='Historical')
            sns.scatterplot(x=fc_ind, y=y_hat_df.iloc[:,0], color='g', s=15, alpha=1, label='Forecast')
            plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation=45, rotation_mode="anchor")
            # equidate_ax(fig, ax, self.df_with_fc[-lookback-days_fc:].index.date)
        ax.set_ylim(get_y_lim(df_with_fc.iloc[:,0]))
        equidate_ax(fig, ax, df_with_fc.index.date, days_fc=days_fc)
        if lookback:
            x_tick_locator = ticker.LinearLocator(numticks=numticks)
            ax.get_xaxis().set_major_locator(x_tick_locator)
        # ax.set_xticklabels(ax.xaxis.get_minorticklabels(), rotation=0)
        if conf_ints:
            conf_filename = '_Conf'
            conf_title = ' with Confidence Intervals'
            conf_int = np.asarray(self.fc_conf_ints)
            if fin:
                # ax.fill_between(range(lookback, lookback+days_fc),
                ax.fill_between(fc_ind,
                conf_int[:, 0], conf_int[:, 1],
                alpha=0.3, color='orange',
                label="Confidence Intervals")
            else:
                # ax.fill_between(self.new_dates,
                ax.fill_between(fc_ind,
                conf_int[:, 0], conf_int[:, 1],
                alpha=0.3, facecolor='orange',
                label="Confidence Intervals")
        ax.set_ylabel(f'{ylabel} (USD)', size=20)
        fig.subplots_adjust(top=0.92)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        # fig.suptitle(f'{ylabel} Time Series, {self.timeframe}, Freq = {self.freq}: Historical vs Forecast with Confidence Intervals\n', size=30)
        fig.suptitle(f'{ylabel} Time Series, {self.timeframe}, Freq = {self.freq}: Historical vs Forecast{conf_title}\n', size=24)
        ax.legend(loc='upper left', borderaxespad=0.5, prop={"size":20})
        if func == 'AA':
            ax.set_title(f'AutoARIMA Best Parameters: {self.AA_best_params}', size=24)
            plt.savefig(f'{TOP}/images/AutoArima/{self.ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf_filename}.png')
        elif func == 'GS':
            ax.set_title(f'GridSearch Best Parameters: {self.GS_best_params}', size=24)
            plt.savefig(f'{TOP}/images/GridSearch/{self.ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf_filename}.png')
        else:
            ax.set_title(f'Parameters: {self.mod_params}', size=24)
            plt.savefig(f'{TOP}/images/{self.ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf_filename}.png')

    def __get_model_scores(self, y_test, y_hat, y_train, model, verbose=0, debug=False):
        try:
            AIC = model.named_steps['arima'].aic()
        except AttributeError:
            AIC = model.named_steps['arima'].model_.aic()
        RMSE = mse(y_test, y_hat, squared=False)
        RMSE_pc = 100*RMSE/y_train.mean()
        SMAPE = smape(y_test, y_hat)
        if verbose:
            print("Test AIC: %.3f" % AIC)
            print("Test RMSE: %.3f" % RMSE)
            print("This is %.3f%% of the avg observed value." % RMSE_pc)
            print("Test SMAPE: %.3f%%\n" % SMAPE)
        if debug:
            print("AIC: %.3f | RMSE: %.3f | RMSE%%=%.3f%% | SMAPE %.3f%%" % (AIC, RMSE, RMSE_pc, SMAPE))
        return AIC, RMSE, RMSE_pc, SMAPE

    def run_stepwise_CV(self, model=None, func='AA', dynamic=False, verbose=1, visualize=True, return_conf_int=True):
        model_str = ''
        '''
        Runs step-wise cross-validation on either an existing AutoARIMA pipeline,
        an existing GridSearch pipeline, or an adhoc pipe passed in as `model`.
        '''
        if func == 'AA':
            if not model:
                model = self.AA_mod_pipe
            mod_params = self.AA_best_params
            mod_params_df = self.AA_best_mod_params_df
        elif func == 'GS':
            if not model:
                model = self.GS_best_mod_pipe
            mod_params = self.GS_best_params
            mod_params_df = self.GS_best_mod_params_df
            # in case not already fit
            try:
                self.model.named_steps['arima'].arroots()
            except Exception as e:
                print('Fitting model... ', end='') if self.verbose else None
                self.fit_model(model)
                print('Done.') if self.verbose else None

        elif func == 'adhoc':
            mod_params_df = self.mod_params_df
            if not model:
                model = self.mod_pipe
                # mod_params = self.mod_params - will not exist
            else:
                # special function, upon loading passed in model pipeline,
                # checking if pipeline contains AutoARIMA and, if so,
                # replace with ARIMA
                if type(model.named_steps['arima'])==pm.AutoARIMA:
                    arima_order = model.named_steps['arima'].model_.order
                    with_intercept = model.named_steps['arima'].model_.with_intercept
                    trend = model.named_steps['arima'].model_.trend
                    # if this was an AutoARIMA pipe, replace AutoARIMA with ARIMA
                    # so that we don't have to do the AA search again
                    arima_params = model.named_steps['arima'].get_params()
                    arima_params['order'] = arima_order
                    arima_params['with_intercept'] = with_intercept
                    arima_params['trend'] = trend
                    arima_model = pm.arima.ARIMA(**arima_params)

                    model.steps[-1]=('arima', arima_model)
                    model = pm.pipeline.Pipeline(model.steps)
                else:
                    arima_order = model.named_steps['arima'].order
                    with_intercept = model.named_steps['arima'].with_intercept
                    trend = model.named_steps['arima'].trend
                mod_params_df['ARIMA_Order'] = str(arima_order)
                mod_params_df['Mod_Order'] = arima_order[0] + arima_order[2]
                mod_params_df['Trend'] = trend
                mod_params_df['Intercept'] = with_intercept
                try:
                    model.named_steps['date']
                except KeyError:
                    mod_params_df['Date'] = False
                else:
                    mod_params_df['Date'] = True

                m_list = []
                k_list = []
                for i, j in model.named_steps.items():
                    if str(j)[:7]=='Fourier':
                        m_list.append(model.named_steps[i].m)
                        k_list.append(model.named_steps[i].k)
                m_str = ', '.join([str(m) for m in m_list])
                k_str = ', '.join([str(k) for k in k_list])
                if m_str:
                    mod_params_df['Fourier'] = True
                    mod_params_df['Fourier_m'] = m_str
                    mod_params_df['Fourier_k'] = k_str
                else:
                    mod_params_df['Fourier'] = False

                try:
                    model.named_steps['box']
                except KeyError:
                    mod_params_df['BoxCox'] = False
                else:
                    mod_params_df['BoxCox'] = True
                try:
                    model.named_steps['log']
                except KeyError:
                    mod_params_df['Log'] = False
                else:
                    mod_params_df['Log'] = True
            if verbose == 1:
                print('Explicit model pipe passed: \n', mod_params_df)
                print('Pipeline: \n', model)
            # in case not already fit
            try:
                self.model.named_steps['arima'].arroots()
            except Exception as e:
                print('Fitting model... ', end='') if self.verbose else None
                self.fit_model(model)
                print('Done.') if self.verbose else None

        if verbose:
            if func == 'AA':
                print(self.AA_best_params)
                print(self.AA_mod_pipe)
                model_str = ' on best model from AutoArima.'
            elif func == 'GS': # for GridSearch or adhoc
                print(self.GS_best_params)
                print(self.GS_best_mod_pipe)
                model_str = ' on best model from GridSearch.'
            elif func == 'adhoc':
                # print(self.mod_params)
                # print(self.mod_pipe)
                model_str = ' on adhoc model.'

            print(f'Starting step-wise cross-validation{model_str}...')
        # X_train, y_train, X_test, y_test, y_hat, conf_ints = self.__run_stepwise_CV(model=model, dynamic=dynamic)

        y_hat, conf_ints = self.__run_stepwise_CV(model=model, dynamic=dynamic, verbose=verbose)
        self.y_hat = y_hat
        self.conf_ints = conf_ints
        self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE = self.__get_model_scores(self.y_test, self.y_hat, self.y_train, model=model, verbose=verbose)
        mod_params_df['Scored'].values[0] = True
        mod_params_df['AIC'].values[0] = '%.4f' % (self.AIC)
        mod_params_df['RMSE'].values[0] = '%.4f' % (self.RMSE)
        mod_params_df['RMSE%'].values[0] = '%.4f' % (self.RMSE_pc)
        mod_params_df['SMAPE'].values[0] = '%.4f' % (self.SMAPE)
        mod_params_df['CV_Time'].values[0] = '%.4f' % (self.end-self.start)

        # update scores
        if func == 'AA':
            self.AA_best_mod_params_df = mod_params_df
            self.__pickle_model(func='AA', verbose=verbose)
        if func == 'GS':
            self.GS_best_mod_params_df = mod_params_df
            self.__pickle_model(func='GS', verbose=verbose)
        if func == 'adhoc':
            self.mod_params_df = mod_params_df
            self.__pickle_model(func='adhoc', verbose=verbose)
        index = csv_write_data(self.mod_CV_filepath, mod_params_df, verbose=verbose)
        print()

        if visualize:
            # self.plot_test_predict(y_hat, ylabel=self.data_name, func=func)
            self.plot_test_predict(self.y_hat, conf_ints=return_conf_int, ylabel=self.data_name, func=func)
        return self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE

    def run_auto_pipeline(self, show_summary=False, visualize=False, return_conf_int=True, verbose=1):
        if verbose:
            print('Starting AutoARIMA...')
            print(f'Data set diffs to use: {self.n_diffs}') if self.n_diffs else None
            print(f'Data set seasonal diffs to use: {self.ns_diffs}') if (self.fit_seas and self.ns_diffs) else None
        self.AA_best_params, self.AA_mod_pipe = self.__reset_mod_params()
        # X_train, y_train, X_test, y_test, y_hat, conf_ints = \
        self.__run_auto_pipeline(show_summary=show_summary, return_conf_int=return_conf_int, verbose=verbose)
        print(f'Best params:\n  {self.AA_best_params}')

        # save best params
        # pipe_params = [(name, transform) for name, transform in pipe.named_steps.items()]
        # self.__pickle_model(func='AA', verbose=verbose)

        self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE = \
            self.__get_model_scores(self.y_test, self.y_hat, self.y_train, model=self.AA_mod_pipe, verbose=verbose)
        if visualize:
            # self.plot_test_predict(y_hat, ylabel=self.data_name, func='AA')
            self.plot_test_predict(self.y_hat, conf_ints=return_conf_int, ylabel=self.data_name, func='AA')

        return self.AA_mod_pipe
        # return df_train, df_test, exog_train, exog_test
        # return X_train, X_test, exog_train, exog_test

    def run_gridsearch_CV(self, min_p=0, max_p=10, min_q=0, max_q=10, min_d=0, max_d=2,
            min_order=0, max_order=6, t_list=['n','c','t','ct'], with_intercept='auto',
            f_m=None, k=None, date=True, fourier=True, box=False, log=False,visualize=True,
            return_conf_int=True, max_models=0, verbose=1, debug=False, parallel=True):
        print('Setting up GridSearchCV...')
        self.GS_best_params, self.GS_best_mod_pipe = self.__reset_mod_params()
        if not f_m:
            f_m = self.f_m
        if not k:
            k = self.k
        best_model, scores = self.__gridsearch_CV(min_p=min_p, max_p=max_p,
            min_q=min_q, max_q=max_q, min_d=min_d, max_d=max_d, min_order=min_order,
            max_order=max_order, t_list=t_list, with_intercept=with_intercept,
            f_m=f_m, k=k, date=date, fourier=fourier, box=box, log=log,
            max_models=max_models, verbose=verbose, debug=debug, parallel=parallel)

        if scores:
            print('GridsearchCV Completed.\n')
            print('Top 10 models from this run:')
            for model, AIC, RMSE, RMSE_pc, SMAPE in scores[:10]:
                print('Model[%s]: AIC=%.3f | RMSE=%.3f | RMSE%%=%.3f%% | SMAPE=%.3f%%' % (model, AIC, RMSE, RMSE_pc, SMAPE))
        else:
            print('No models were scored this run.')

            # load best model from model pickle file
            mod_data = Pmdarima_Model.__unpickle_model(self.ts, self.tf, self.f, func='GS')
            # mod_file = open(pkl_filepath,'rb')
            # mod_data = pkl.load(mod_file)
            # mod_file.close()

            self.GS_best_params = mod_data[0]
            self.GS_best_mod_pipe = mod_data[1]
            self.GS_best_mod_params_df = mod_data[2]

            # run CV on model if not y_hat and conf_ints aren't present
            try:
                self.y_hat = mod_data[4][0]
                self.conf_ints = mod_data[4][1]
            except Exception as e:
                self.fit_model(self.GS_best_mod_pipe, func='GS')
                self.y_hat, self.conf_ints = self.__run_stepwise_CV(self.GS_best_mod_pipe, verbose=verbose)

        if visualize:
            # self.plot_test_predict(self.y_hat, conf_ints=conf_ints, ylabel=self.data_name, func='GS')
            self.plot_test_predict(self.y_hat, conf_ints=return_conf_int, ylabel=self.data_name, func='GS')
        return best_model, scores

    def run_prediction(self, model, days_fc, en_ex, exog_fc_df=None, visualize=True, lookback=0,
                        fin=False, ohlc_df=None, exog_hist_df=None, func='GS', all_ohlc=False,
                        ohlc_fc_df=None, return_conf_int=True, verbose=1):
        '''
        Out of sample predictions. Needs n separate Pmdarima_Model objects, one
        for each variable. Run predictions on Exogenous variables first,
        then run on Endogenous variable.
        If `en_ex` == 'endo', `exog_df` and 'hist_df' must be provided.
        hist_df : historical observations of all variables
        exog_df : exogenous variable for running final predictions
        '''
        try:
            assert(en_ex in ('endo', 'exog')), "Incorrect parameters passed for 'endo'/'exog' switch."
        except AssertionError as e:
            print(e)
            print('Failed to initialize.')
            raise

        self.days_fc = days_fc
        if en_ex == 'exog':
            hist_df = self.df
        else:
            try:
                assert(days_fc == exog_fc_df.shape[0]), "Scalar value `days_fc` and # of rows in `exog_fc_df` do not match."
            except AssertionError as e:
                print(e)
                print(f'Number of rows implies `days_fc` should be {exog_fc_df.shape[0]}.')
                raise
            hist_df = exog_hist_df
            self.exog_df = exog_fc_df
        # try:
        #     assert(type(hist_df) in (pd.Series, pd.DataFrame)), "Historical exog data is not of type Pandas Series or DataFrame."
        #     # assert(type(hist_df) == (pd.DatetimeIndex)), "Historical data index is not of type Pandas DatetimeIndex."
        # except AssertionError as e:
        #     # print(e)
        #     hist_df = pd.DataFrame(hist_df.index, columns=['date'])
        #     print('Using class variable \'df\' - original DataFrame.')

        self.hist_df = hist_df

        # if no model passed, use adhoc model
        if not model:
            model = self.mod_pipe

        if verbose:
            var = None
            if en_ex == 'exog':
                var = 'Exogenous'
            elif en_ex == 'endo':
                var = 'Endogenous'
            print(f'Running Fit and Predict on {var} variable {self.data_name}...')


        today = self.df.index[-1]
        # df_size = self.length
        self.new_dates, self.index_fc, self.new_dates_df = Pmdarima_Model.__get_next_dates(today, self.length, self.days_fc)
        # self.new_dates = new_dates
        # self.index_fc = index_fc
        # self.new_dates_df = new_dates_df

        # run Fit/Predict
        # note that the fc_dt returned for exogenous data does not include date
        self.fc_df, self.y_fc_hat, self.fc_conf_ints = \
            self.__fit_predict(model, self.days_fc, self.new_dates,
            self.index_fc, self.hist_df, self.hist_dates_df, en_ex, self.new_dates_df, self.exog_df, verbose)
        # self.fc_df, self.results = \
        # self.results = \
        # self.fc_df = fc_df
        # self.y_hat = y_hat
        # self.conf_ints = conf_ints

        if visualize:
            if fin:
                self.plot_forecast_conf(ohlc_df=ohlc_df, y_hat_df=self.fc_df,
                conf_ints=return_conf_int, func=func, fin=fin,
                lookback=lookback, all_ohlc=all_ohlc, ohlc_fc_df=ohlc_fc_df)
            else:
                self.plot_forecast_conf(hist_df=self.df.iloc[:,0], y_hat_df=self.fc_df,
                lookback=lookback, conf_ints=return_conf_int, func=func, fin=fin)

        return self.fc_df, self.y_fc_hat, self.new_dates_df, self.fc_conf_ints

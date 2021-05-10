from tqdm.notebook import trange, tqdm
import pandas as pd
import matplotlib
import numpy as np
# import csv
from itertools import product
from functools import reduce
import pickle as pkl

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

from functions import *

from pathlib import Path
TOP = Path(__file__ + '../../..').resolve()

NYSE = mcal.get_calendar('NYSE')
CBD = pd.offsets.CustomBusinessDay(calendar=NYSE)

print(f'Pmdarima_Model.py loaded from {TOP}/data..')

class Pmdarima_Model:
    def __init__(self, df, data_name, n, periods, freq, train_size=80, trend='c', with_intercept='auto',
                order=(0,1,0), s_order=(0,0,0), seas=0, fit_seas=False, f_seas=252, k=4,
                estimate_diffs=False, impute=False, AA_d=None, AA_D=None, verbose=1,
                #max_d=2, max_p=2, max_q=2, max_D=2, max_P=2, max_Q=2,
                date=True, fourier=True, box=False, log=False):

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
        if impute:
            self.df = df.interpolate()
        # else:
        #     self.df = df
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
            # mod_params = 'ARIMA Order()'
            # mod_params = mod_params.replace('()', self.mod_order)
            # if not self.no_intercept:
            #     mod_params += ', Intercept'
            # if self.date:
            #     mod_params += ', DateFeaturizer'
            # if self.fourier:
            #     mod_params += ', FourierFeaturizer'
            # if self.box:
            #     mod_params += ', BoxCoxEndogTransformer'
            # if self.log:
            #     mod_params += ', LogEndogTransformer'

        return mod_params, mod_pipe

    def __pickle_model(self, func='AA', verbose=1):
        '''
        Helper function for pickling a model along with its params as a
        human-readable string.
        '''
        # var = self.data_name.lower()
        pkl_file = f'{TOP}/models/{self.ts}_{self.tf}_{self.f}_{func}_best_model.pkl'
        pkl_out = open(pkl_file, 'wb')
        if func == 'AA':
            func_type = 'AutoARIMA'
            pkl.dump((self.AA_best_params, self.AA_mod_pipe), pkl_out)
        if func == 'GS':
            func_type = 'GridSearchCV'
            pkl.dump((self.GS_best_params, self.GS_best_mod_pipe), pkl_out)
        print(f'Saved best {func_type} model as {pkl_file}') if verbose else None
        pkl_out.close()

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

    def __fit_predict(self, model, days_fc, new_dates, index_fc, hist_dates_df, en_ex, new_dates_df=None, exog_df=None, verbose=1):
        model.fit(self.df, hist_dates_df)
        '''
        Helper function for fitting a model on the full input DataFrame and
        running an out of sample prediction.
        '''
        print('Successfully fit model on historical observations.') if verbose else None

        if en_ex == 'exog':
            y_hat, conf_ints = model.predict(X=new_dates_df, return_conf_int=True)
            fc_df = pd.DataFrame(y_hat, index=index_fc, columns=[f'{self.data_name.lower()}'])
            fc_date_df = pd.DataFrame(zip(new_dates, y_hat), index=index_fc, columns=['date', f'{self.data_name.lower()}'])
            fc_date_df.set_index('date', inplace=True)
        elif en_ex == 'endo':
            y_hat, conf_ints = model.predict(X=exog_df, return_conf_int=True)
            fc_date_df = pd.DataFrame(zip(new_dates, y_hat), index=index_fc, columns=['date', f'{self.data_name.lower()}'])
            fc_date_df.set_index('date', inplace=True)
            fc_df = fc_date_df

        self.df_with_fc = self.df.append(fc_date_df)
        print(f'Successfully forecasted {days_fc} days forward.') if verbose else None

        # fc_df = pd.DataFrame(zip(self.new_dates_df.date.values,y_hat), columns=['date','close'])
        return fc_df, y_hat, conf_ints

    # @classmethod
    # def get_next_dates(cls, today, df_size, days):
    @staticmethod
    def get_next_dates(today, df_size, days_fc):
        '''
        Static method for getting new dates for out of sample predictions.
        Returns a list of Pandas Timestamps, a list of numerical indices extending
        the original numerical indices of the input DataFrame, and a DataFrame consisting
        of the two aforementioned lists.
        '''
        next_day = today + CBD
        new_dates = pd.date_range(start=next_day, periods=days_fc, freq=CBD)
        index_fc = range(df_size, df_size + days_fc)
        new_dates_df = pd.DataFrame(new_dates, index=index_fc, columns=['date'])
        return new_dates, index_fc, new_dates_df

    @classmethod
    def join_exog_data(cls, *args):
        '''
        Takes any number of DataFrames with matching indexes and performs a join.
        First DataFrame must be the dates_df. Number of observations in each must match.
        '''
        try:
            assert(len(set(map(lambda df: df.shape, args))) == 1), "Input DataFrame shapes do not match."
        except AssertionError as e:
            print(e)
            print('Failed to perform join.')
            raise
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
        mod_params_dict['Mod_Order'] = p+q
        mod_params_dict['Trend'] = t
        mod_params_dict['Intercept'] = with_intercept
        mod_params_dict['Date'] = date
        mod_params_dict['Fourier'] = fourier
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
        mod_params_df.index.name = 'Model'
        return mod_params, mod_params_df, pipe

    def __run_stepwise_CV(self, model=None, func='AA', dynamic=False, verbose=1, debug=False):
        '''
        Heavily modified from https://github.com/alkaline-ml/pmdarima/issues/339
        '''
        def __forecast_one_step(date_df):
            fc, conf_int = model.predict(X=date_df, return_conf_int=True)
            return fc.tolist()[0], conf_int[0].tolist()
        if not model:
            model = self.AA_mod_pipe
            print('No model specified, defaulting to AutoARIMA best model.') if verbose else None
        # X_train = self.X_train
        # X_test = self.X_test
        # y_train = self.y_train
        # y_test = self.y_test
        # try:
        #     assert(hasattr(model.named_steps['arima'], 'model_')), "Hmm"
        # except AssertionError:
        #     print('Model has not been fitted. Fitting now.')
            # model.fit(y_train, X_train)
        # model = self.mod_pipe
        # date = pd.DataFrame([X_test.iloc[0].date], columns=['date'])
        date_df = pd.DataFrame([self.X_test.iloc[0].date], index=[self.X_train.size], columns=['date'])
        # date = X_test.iloc[0].date
        forecasts = []
        conf_ints = []
        dynamic_str = ''
        if dynamic:
            dynamic_str = ' dynamically with forecasted values'
        if verbose:
            print(f'Iteratively making predictions on {self.data_name} Time Series and updating model{dynamic_str}, beginning with first index of y_test ...')
        self.start = time.time()
        for n, new_ob in enumerate(tqdm(self.y_test, desc='Step-Wise Prediction Loop')):

            fc, conf = __forecast_one_step(date_df)

                # print(e)
                # print('Fitting on class variables X_train, y_train.')
                # model.fit(self.y_train, self.X_train)
                # self.fit_model(model)
                # fc, conf = __forecast_one_step(date_df)
            forecasts.append(fc)
            conf_ints.append(conf)

            # update the existing model with a small number of MLE steps
            if dynamic:
                model.update([fc], date_df)
            elif not dynamic:
                model.update([new_ob], date_df)

            ## make a little animation
            if n&1:
                print('>_', end='\r')
            else:
                print('> ', end='\r')
            # date = pd.DataFrame([X_test.iloc[0].date + CBD], index=[X_train.size]columns=['date'])
            date_df.iloc[0].date += CBD
            date_df.index += 1
            # date += CBD
        self.end = time.time()
        print('Done.')
        # y_hat = pd.Series(forecasts, index=y_test.index)
        # self.conf_ints = conf_ints
        # self.y_hat = forecasts

        # return X_train, y_train, X_test, y_test, forecasts, conf_ints
        return forecasts, conf_ints

    def __GS_score_model(self, model, verbose=0, debug=False):
        print('________________________________________________________________________')
        print('\nStarting step-wise cross-validation...')
        if verbose:
            print(model[0])
            # print(model[1])
            print(model[2])
        # model[1].fit(self.y_train, self.X_train)
        self.fit_model(model[2])
        result = None
        # convert config to a key
        key = str(model[0])
        # print(key)
        # model[1].fit(y_train, X_train)
        # show all warnings and fail on exception if debugging
        if debug:
            # X_train, X_test, y_train, y_test, y_hat, conf_ints = self.__run_stepwise_CV(model[1])
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
            except:
                error = None
                # check for an interesting result
        if result:
            print('Model[%s]: AIC=%.3f | RMSE=%.3f | RMSE_pc=%.3f%% | SMAPE=%.3f%%' % (key, *result))
            if result[1] < self.RMSE:
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
            model[1]['Scored'].values[0] = True
            model[1]['AIC'].values[0] = '%.4f' % (result[0])
            model[1]['RMSE'].values[0] = '%.4f' % (result[1])
            model[1]['RMSE%'].values[0] = '%.4f' % (result[2])
            model[1]['SMAPE'].values[0] = '%.4f' % (result[3])
            model[1]['CV_Time'].values[0] = '%.4f' % (self.end-self.start)
            # pd.concat([self.GS_all_mod_params_df,model[1]]).drop_duplicates(keep='last').reset_index(drop=True)
            self.GS_all_mod_params_df = self.GS_all_mod_params_df.append(model[1])
            self.GS_all_mod_params_df = self.GS_all_mod_params_df.drop_duplicates(keep='last')
            self.GS_all_mod_params_df = self.GS_all_mod_params_df.reset_index(drop=True)
            self.GS_all_mod_params_df.index.name = 'Model'
            csv_write_data(self.mod_CV_filepath, model[1], verbose=verbose)
            print()

        return (key, *result)

    def __gridsearch_CV(self, min_order=0, max_order=6, max_d=1, t_list=['n','c','t','ct'],
            with_intercept=False, f_m=None, k=None, date=True, fourier=True, box=False,
            log=False, verbose=0, debug=False, parallel=True):
        '''
        Heavily modified from https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
        '''
        def __GS_setup_params(t_list=['n','c','t','ct'], with_intercept='auto',
            f_m=self.f_m, k=self.k, date=True, fourier=True, box=False, log=False, verbose=0):

            print('Building list of models to test.') if verbose else None
            # date_feat = pm.preprocessing.DateFeaturizer(
            # column_name="date",  # the name of the date feature in the X matrix
            # with_day_of_week=True,
            # with_day_of_month=True)
            # _, X_train_feats = date_feat.fit_transform(self.y_train, self.X_train)

            columns = ['ARIMA_Order', 'Mod_Order', 'Trend', 'Intercept', 'Date', 'Fourier',
                        'Fourier_m', 'Fourier_k', 'BoxCox', 'Log',
                        'Scored', 'AIC', 'RMSE', 'RMSE%', 'SMAPE', 'CV_Time']
            # arima_params = dict(
            #     # d = self.n_diffs,
            #     trace=3,
            #     maxiter=200,
            #     # stepwise=True,
            #     # seasonal=False,
            #     suppress_warnings=True)
            # feats_list = 2**sum(box, log)
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
            count = 0
            self.GS_all_mod_params_df = pd.DataFrame(columns=columns)
            self.GS_all_mod_params_df.index.name = 'Model'
            mod_params_dict = {}
            for d in range(0, max_d+1):
                mod_order = 0
                for p in range(7):
                    for q in range(7):
                        mod_order = p+q
                        # print(mod_order)
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
                                # mod_params = 'ARIMA()'
                                # arima_order = (p, d, q)
                                # mod_params_df.ARIMA = [(2,1,0)]
                                # mod_params_df.Trend = [t]
                                # order = f'({p}, {d}, {q})[\'{t}\']'
                                # mod_params = mod_params.replace('()', order)
                                # if verbose:
                                    # print(f'Model Order = {order}', end = '')
                                # if intercept:
                                #     mod_params += ', Intercept'
                                    # arima_params['with_intercept'] = intercept
                                    # print(', Intercept', end='') if verbose else None
                                # print() if verbose == 1 else None
                                # mod_params += ', DateFeaturizer, FourierFeaturizer'
                                for date, fourier, box, log in feats_iter:
                                    f_m_mod = f_m
                                    k_mod = k
                                    if not fourier:
                                        f_m_mod = None
                                        k_mod = None
                                        # print(date, fourier, box, log)
                                    # order = f'({p}, {d}, {q})'
                                    # order_str = str(tuple(order))
                                    # pipe_params = []
                                    # if date == True: # always use DateFeaturizer if True
                                    #     pipe_params.append(('date', date_feat))
                                    #     mod_params += ', DateFeaturizer'
                                    # if fourier == True: # always use FourierFeaturizer if True
                                    #     pipe_params.append(('fourier', pm.preprocessing.FourierFeaturizer(m=self.f_m, k=4)))
                                    #     mod_params += ', FourierFeaturizer'
                                    # if feats == 0:
                                    #     print('', end='')
                                    # elif feats == 1:
                                    # if box == True:
                                    #     pipe_params.append(('box', pm.preprocessing.BoxCoxEndogTransformer()))
                                    #     mod_params += ', BoxCoxEndogTransformer'
                                    # elif feats == 1:
                                    # if log == True:
                                    #     pipe_params.append(('log', pm.preprocessing.LogEndogTransformer()))
                                    #     mod_params += ', LogEndogTransformer'
                                    # elif feats == 3:
                                    #     pipe_params.append(('box', pm.preprocessing.BoxCoxEndogTransformer()))
                                    #     pipe_params.append(('log', pm.preprocessing.LogEndogTransformer()))
                                        # mod_params += ', BoxCoxEndogTransformer'
                                        # mod_params += ', LogEndogTransformer'
                                    # arima_params['order'] = arima_order
                                    # arima_params['trend'] = t
                                    # pipe_params.append(('arima', pm.arima.ARIMA(**arima_params)))
                                    # pipe = pipeline.Pipeline(pipe_params)
                                    # models.append((mod_params, pipe))
                                    # mod_params_dict['ARIMA_Order'] = f'(p, d, q)'
                                    # mod_params_dict['Trend'] = t
                                    # mod_params_dict['Intercept'] = intercept
                                    # mod_params_dict['Date'] = date
                                    # mod_params_dict['Fourier'] = fourier
                                    # mod_params_dict['Fourier_m'] = f_m
                                    # mod_params_dict['Fourier_k'] = k
                                    # mod_params_dict['BoxCox'] = box
                                    # mod_params_dict['Log'] = log
                                    # mod_params_dict['AIC'] = None
                                    # mod_params_dict['RMSE'] = None
                                    # mod_params_dict['RMSE%'] = None # (100*RMSE/y_train.mean()))
                                    # mod_params_dict['SMAPE'] = None
                                    # self.GS_mod_params_df.(f'{TOP}/model_scores/{self.ts}_{self.tf}_{self.f}_GS.csv')
                                    mod_params, mod_params_df, pipe = self.__setup_mod_params(
                                        p=p, d=d, q=q, t=t, with_intercept=intercept,
                                        f_m=f_m_mod, k=k_mod, date=date,
                                        fourier=fourier, box=box, log=log,
                                        func='GS', verbose=verbose)
                                    # print(mod_params)
                                    # index = check_mod_score_exists(csv_filepath, mod_params_df)
                                    # if index == True:  # model found and scored
                                    #     continue
                                    # elif index == False: # model params not have been logged in file
                                    # print(mod_params_df)
                                    index = csv_write_data(self.mod_CV_filepath, mod_params_df, verbose=verbose)
                                    if index == -1:  # model found and scored
                                        print('Model already scored, skipping.') if verbose else None
                                        continue
                                    print('Added a model to test grid.') if verbose else None
                                    # or if model found but not scored, no need to write anything
                                    # otherwise, if model not found, write model params to csv

                                    self.GS_all_mod_params_df = self.GS_all_mod_params_df.append(mod_params_df)
                                    models.append((mod_params, mod_params_df, pipe))
                                    count += 1
            self.GS_mod_count = count
            try:
                assert(count > 0), 'Based on parameters given, 0 models built, terminating.'
            except AssertionError as e:
                print(e)
                raise
            print(f'Finished building list of {count} models.') if verbose else None
            return models
            # print(models) if debug else None

        def __GS_start(models, verbose=0, debug=False, parallel=True):

            print('Starting iterative search through all GridSearch params.') if verbose else None
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
                scores = [self.__GS_score_model(model, debug=debug, verbose=verbose) for model in tqdm(models, desc='Model Loop')]
                # scores = [self.__GS_score_model(model, debug=debug) for model in models]
            # remove empty results
            scores = [score for score in scores if score[1] != None]
            # # sort configs by error, asc
            scores.sort(key=lambda tup: tup[2])
            return scores

        # if __name__ == '__main__':  # <- prevent RuntimeError for 'spawn'
        models = __GS_setup_params(t_list=t_list, with_intercept=with_intercept,
            f_m=f_m, k=k, date=date,
            fourier=fourier, box=box, log=log, verbose=verbose)
        scores = __GS_start(models, debug=debug, verbose=verbose, parallel=parallel)
        print(scores)
        # clear()
        if scores == 1:
            return 1, 1
        else:
            print('GridsearchCV Completed.\n')
            print('Top 10 models:')
            for model, AIC, RMSE, RMSE_pc, SMAPE in scores[:10]:
                print('Model[%s]: AIC=%.3f | RMSE=%.3f | RMSE%%=%.3f%% | SMAPE=%.3f%%' % (model, AIC, RMSE, RMSE_pc, SMAPE))
            self.__pickle_model(func='GS', verbose=verbose)
            return self.GS_best_mod_pipe, scores

    def __run_auto_pipeline(self, show_summary=False, return_conf_int=False, verbose=1):
        # pm.tsdisplay(self.df_train, lag_max=60, title = f'{self.data_name} Time Series Visualization') \
        pm.tsdisplay(self.df, lag_max=60, title = f'{self.data_name} Time Series Visualization') \
            if show_summary else None
        # X_train, y_train, X_test, y_test = self.__split_df_dates(self.df_train, self.df_test)
        # self.X_train, self.X_test, self.y_train, self.y_test = self.__train_test_split_dates(self.df_train, self.df_test)

        # params = []
        # if self.date:
        #     print('Using DateFeaturizer.') if verbose == 1 else None
        #     date_feat = pm.preprocessing.DateFeaturizer(
        #         column_name="date",  # the name of the date feature in the X matrix
        #         with_day_of_week=True,
        #         with_day_of_month=True)
        #     _, X_train_feats = date_feat.fit_transform(self.y_train, self.X_train)
        #     # _, X_train_feats = date_feat.fit_transform(y_train[:,0], X_train)
        #     params.append(('date', date_feat))
        # if self.fourier:
        #     print('Using FourierFeaturizer.') if verbose == 1 else None
        #     params.append(('fourier', pm.preprocessing.FourierFeaturizer(m=self.f_m, k=4)))
        # if self.box:
        #     print('Using BoxCoxEndogTransformer.') if verbose == 1 else None
        #     params.append(('box', pm.preprocessing.BoxCoxEndogTransformer()))
        # if self.log:
        #     print('Using LogEndogTransformer.') if verbose == 1 else None
        #     params.append(('log', pm.preprocessing.LogEndogTransformer()))
        # arima_params = dict(
        #     d = self.n_diffs,
        #     trace=3,
        #     maxiter=200,
        #     stepwise=True,
        #     suppress_warnings=True)
        # if self.fit_seas:
        #     arima_params['seasonal']=True
        #     arima_params['m']=self.m
        #     arima_params['max_p']=0
        #     arima_params['start_p']=0
        #     arima_params['max_q']=0
        #     arima_params['start_q']=0
        #     arima_params['max_P']=0
        #     arima_params['start_P']=0
        #     arima_params['max_Q']=0
        #     arima_params['start_Q']=0
        #     arima_params['start_params']=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
        #     arima_params['D']=self.ns_diffs
        # elif not self.fit_seas:
        #     arima_params['seasonal']=False
        # params.append(('arima', pm.arima.AutoARIMA(**arima_params)))
        AA_mod_params, AA_mod_params_df, AA_pipe = self.__setup_mod_params(d=self.n_diffs, D=self.ns_diffs,
            with_intercept=self.with_intercept, m=self.m, f_m=self.f_m, k=self.k,
            date=self.date, fourier=self.fourier, box=self.box, log=self.log,
            func='AA', verbose=verbose)

        if verbose:
            print(f'Parameters for AutoARIMA Pipeline: \n  {AA_mod_params}') if verbose else None
            print(AA_pipe, '\n')
        # pipe = pipeline.Pipeline(mod_pipe)

        # pipe.fit(y_train, X_train)
        # print(AA_mod_params_df)
        # if True:
        #     raise Exception("Testing.")
        self.fit_model(AA_pipe, func='AA')

        # save best params
        # pipe_params = [(name, transform) for name, transform in pipe.named_steps.items()]
        self.__pickle_model(func='AA', verbose=verbose)
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
        print(f'Best params:\n  {self.AA_best_params}')
        # print(self.AA_mod_pipe)
        # run prediction on test set
        # conf_ints = []
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
        ylabel=self.data_name
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
        ax.legend(loc='upper left', borderaxespad=0.5)
        fig.suptitle(f'{ylabel} Time Series, {self.timeframe}, Freq = {self.freq}: Test vs Predict with Confidence Interals\n', size=24)
        ax.set_ylabel(ylabel, size=14)
        if func == 'AA':
            ax.set_title(f'AutoARIMA Best Parameters: {self.AA_best_params}', size=16)
            plt.savefig(f'{TOP}/images/AutoArima/{self.ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')
        elif func == 'GS':
            ax.set_title(f'GridSearch Best Parameters: {self.GS_best_params}', size=16)
            plt.savefig(f'{TOP}/images/GridSearch/{self.ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')
        else:
            ax.set_title(f'Parameters: {self.mod_params}', size=16)
            plt.savefig(f'{TOP}/images/{self.ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')

    def plot_forecast_conf(self, ohlc_df=None, hist_df=None, y_hat=None, conf_ints=True, days_fc=5,
            lookback=120, ylabel=None, fin=False, func='GS'):
        '''
        Plot forecasts with optional confidence intervals. Can only be run after
        forecasts have been generated.
        '''
        ylabel=self.data_name
        conf = ''
        fig, ax = plt.subplots(figsize=(24, 16))
        if fin:
            mpl.plot(ohlc_df[-lookback:], type='candle', style="yahoo", ax=ax)
            ax.plot(range(lookback, lookback+days_fc), y_hat, 'g.', markersize=10, alpha=0.7, label='Forecast')
            ax.set_xlim(0, lookback+lookback//10)
            equidate_ax(fig, ax, self.df_with_fc[-lookback-days_fc:].index.date)
        else:
            ax.plot(hist_df[-lookback:], color='blue', alpha=0.5, label='Historical')
            ax.plot(self.new_dates, y_hat, 'g.', markersize=10, alpha=0.7, label='Forecast')
            ax.set_xlim(self.df_with_fc.index[-lookback], self.df_with_fc.index[-1]+(lookback//20)*CBD)
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(0, y_max)
        if conf_ints:
            conf = '_Conf'
            conf_int = np.asarray(self.conf_ints)
            if fin:
                ax.fill_between(range(lookback, lookback+days_fc),
                conf_int[:, 0], conf_int[:, 1],
                alpha=0.3, color='orange',
                label="Confidence Intervals")
            else:
                ax.fill_between(self.new_dates,
                conf_int[:, 0], conf_int[:, 1],
                alpha=0.3, facecolor='orange',
                label="Confidence Intervals")
        ax.set_ylabel(f'{ylabel} (USD)', size=14)
        fig.subplots_adjust(top=0.92)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
        fig.suptitle(f'{ylabel} Time Series, {self.timeframe}, Freq = {self.freq}: Historical vs Forecast with Confidence Interals\n', size=24)
        ax.legend(loc='upper left', borderaxespad=0.5)
        if func == 'AA':
            ax.set_title(f'AutoARIMA Best Parameters: {self.AA_best_params}', size=16)
            plt.savefig(f'{TOP}/images/AutoArima/{self.ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf}.png')
        elif func == 'GS':
            ax.set_title(f'GridSearch Best Parameters: {self.GS_best_params}', size=16)
            plt.savefig(f'{TOP}/images/GridSearch/{self.ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf}.png')
        else:
            ax.set_title(f'Parameters: {self.mod_params}', size=16)
            plt.savefig(f'{TOP}/images/{self.ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')



    def __get_model_scores(self, y_test, y_hat, y_train, model, verbose=0, debug=False):
        # pipe = self.mod_pipe
        # if type(pipe.named_steps['arima']=='pm.ARIMA'):
        #     AIC = pipe.named_steps['arima'].aic()
        # elif type(pipe.named_steps['arima']=='pm.AutoARIMA'):
        #     AIC = pipe.named_steps['arima'].model_.aic()
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
        if func == 'AA':
            model = self.AA_mod_pipe
            mod_params_df = self.AA_best_mod_params_df
        elif func == 'GS':
            if not model:
                model = self.GS_best_mod_pipe
            mod_params_df = self.GS_best_mod_params_df
            # in case not already fit
            self.fit_model(model)
        elif func == 'adhoc':
            if not model:
                model = self.mod_pipe
            mod_params_df = self.mod_params_df
            # in case not already fit
            self.fit_model(model)
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
                print(self.mod_params)
                print(self.mod_pipe)
                model_str = ' on adhoc model.'

            print(f'Starting step-wise cross-validation{model_str}...')
        # X_train, y_train, X_test, y_test, y_hat, conf_ints = self.__run_stepwise_CV(model=model, dynamic=dynamic)

        y_hat, conf_ints = self.__run_stepwise_CV(model=model, dynamic=dynamic)
        self.y_hat = y_hat
        self.conf_ints = conf_ints
        self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE = self.__get_model_scores(self.y_test, self.y_hat, self.y_train, model=model, verbose=verbose)
        # self.AIC = AIC
        # self.RMSE = RMSE
        # self.SMAPE = SMAPE
        mod_params_df['Scored'].values[0] = True
        mod_params_df['AIC'].values[0] = '%.4f' % (self.AIC)
        mod_params_df['RMSE'].values[0] = '%.4f' % (self.RMSE)
        mod_params_df['RMSE%'].values[0] = '%.4f' % (self.RMSE_pc)
        mod_params_df['SMAPE'].values[0] = '%.4f' % (self.SMAPE)
        mod_params_df['CV_Time'].values[0] = '%.4f' % (self.end-self.start)
        if func == 'AA':
            self.AA_best_mod_params_df = mod_params_df
        if func == 'GS':
            self.GS_best_mod_params_df = mod_params_df
        if func == 'adhoc':
            self.mod_params_df = mod_params_df
        index = csv_write_data(self.mod_CV_filepath, mod_params_df, verbose=verbose)
        print()

        if visualize:
            # self.plot_test_predict(y_hat, ylabel=self.data_name, func=func)
            self.plot_test_predict(self.y_hat, conf_ints=return_conf_int, ylabel=self.data_name, func=func)
        return self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE

    def run_auto_pipeline(self, show_summary=False, visualize=True, return_conf_int=True, verbose=1):
        if verbose:
            print('Starting AutoARIMA...')
            print(f'Data set diffs to use: {self.n_diffs}') if self.n_diffs else None
            print(f'Data set seasonal diffs to use: {self.ns_diffs}') if (self.fit_seas and self.ns_diffs) else None
        self.AA_best_params, self.AA_mod_pipe = self.__reset_mod_params()
        # X_train, y_train, X_test, y_test, y_hat, conf_ints = \
        self.__run_auto_pipeline(show_summary=show_summary, return_conf_int=return_conf_int, verbose=verbose)
        self.AIC, self.RMSE, self.RMSE_pc, self.SMAPE = \
            self.__get_model_scores(self.y_test, self.y_hat, self.y_train, model=self.AA_mod_pipe, verbose=verbose)
        # self.AIC = AIC
        # self.RMSE = RMSE
        # self.SMAPE = SMAPE
        if visualize:
            # self.plot_test_predict(y_hat, ylabel=self.data_name, func='AA')
            self.plot_test_predict(self.y_hat, conf_ints=return_conf_int, ylabel=self.data_name, func='AA')

        return self.AA_mod_pipe
        # return df_train, df_test, exog_train, exog_test
        # return X_train, X_test, exog_train, exog_test

    def run_gridsearch_CV(self, min_order=0, max_order=6, max_d=1, t_list=['n','c','t','ct'],
            with_intercept='auto', f_m=None, k=None, date=True, fourier=True, box=False, log=False,
            visualize=True, return_conf_int=True, verbose=1, debug=False, parallel=True):
        if verbose:
            print('Starting GridSearchCV...')
        self.GS_best_params, self.GS_best_mod_pipe = self.__reset_mod_params()
        if not f_m:
            f_m = self.f_m
        if not k:
            k = self.k
        best_model, scores = self.__gridsearch_CV(min_order=min_order, max_order=max_order,
            max_d=max_d, t_list=t_list, with_intercept=with_intercept, f_m=f_m, k=k,
            date=date, fourier=fourier, box=box, log=log,
            verbose=verbose, debug=debug, parallel=parallel)
        if best_model == 1:
            return None, None
        if visualize:
            # self.plot_test_predict(self.y_hat, conf_ints=conf_ints, ylabel=self.data_name, func='GS')
            self.plot_test_predict(self.y_hat, conf_ints=return_conf_int, ylabel=self.data_name, func='GS')
        return best_model, scores

    def run_prediction(self, model, days_fc, en_ex, exog_df=None, visualize=True,
                        fin=False, ohlc_df=None, hist_df=None, func='GS',
                        return_conf_int=True, verbose=1):
        '''
        Out of sample predictions. Needs n separate Pmdarima_Model objects, one
        for each variable. Run predictions on Exogenous variables first,
        then run on Endogenous variable.
        '''
        try:
            assert(en_ex in ('endo', 'exog')), "Incorrect parameters passed for 'endo'/'exog' switch."
        except AssertionError as e:
            print(e)
            print('Failed to initialize.')
            raise

        try:
            assert(type(df) in (pd.Series, pd.DataFrame)), "Data is not of type Pandas Series or DataFrame."
            assert(type(df.index) == (pd.DatetimeIndex)), "Data index is not of type Pandas DatetimeIndex."
        except AssertionError as e:
            print(e)
            hist_df = self.df
            print('Using class variable \'df\' - original DataFrame.')

        if not model:
            model = self.AA_mod_pipe

        if verbose:
            var = None
            if en_ex == 'exog':
                var = 'Exogenous'
            elif en_ex == 'endo':
                var = 'Endogenous'
            print(f'Running Fit and Predict on {var} variable {self.data_name}...')

        self.exog_dg = exog_df
        self.days_fc = days_fc

        today = self.df.index[-1]
        # df_size = self.length
        self.new_dates, self.index_fc, self.new_dates_df = Pmdarima_Model.get_next_dates(today, self.length, self.days_fc)
        # self.new_dates = new_dates
        # self.index_fc = index_fc
        # self.new_dates_df = new_dates_df

        # run Fit/Predict
        # note that the fc_dt returned for exogenous data does not include date
        self.fc_df, self.y_fc_hat, self.fc_conf_ints = \
            self.__fit_predict(model, self.days_fc, self.new_dates,
            self.index_fc, self.hist_dates_df, en_ex, self.new_dates_df, self.exog_df, verbose)
        # self.fc_df = fc_df
        # self.y_hat = y_hat
        # self.conf_ints = conf_ints

        if visualize:
            if fin:
                self.plot_forecast_conf(ohlc_df=ohlc_df, y_hat=self.y_fc_hat,
                conf_ints=return_conf_int, func=func, fin=fin, days_fc=self.days_fc)
            else:
                self.plot_forecast_conf(hist_df=hist_df, y_hat=self.y_fc_hat,
                conf_ints=return_conf_int, func=func, fin=fin, days_fc=self.days_fc)

        return self.fc_df, self.y_fc_hat, self.new_dates_df, self.fc_conf_ints

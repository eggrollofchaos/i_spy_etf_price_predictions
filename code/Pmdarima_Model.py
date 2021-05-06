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
top = Path(__file__ + '../../..').resolve()

NYSE = mcal.get_calendar('NYSE')
CBD = pd.offsets.CustomBusinessDay(calendar=NYSE)

print('Pmdarima_Model.py loaded.')

class Pmdarima_Model:
    def __init__(self, df, data_name, n, periods, freq, train_size=80, trend='n',
                start_order=(0,1,0), seas=0, f_seas=0, no_intercept=False, fit_seas=False,
                estimate_diffs=False, impute=False, start_d=1, start_D=None, verbose=1,
                #max_d=2, max_p=2, max_q=2, max_D=2, max_P=2, max_Q=2,
                date=True, fourier=True, box=False, log=False):

        try:
            assert(type(df) in (pd.Series, pd.DataFrame)), "Data is not of type Pandas Series or DataFrame."
            assert(type(df.index) == (pd.DatetimeIndex)), "Data index is not of type Pandas DatetimeIndex."
        except AssertionError as e:
            print(e)
            print('Failed to load data.')
            raise

        try:
            assert(start_order[1] == start_d), "Variables start_d and d start_order conflict."
        except AssertionError as e:
            print(e)
            print('Failed to initialize Class.')
            raise
        if type(df) == pd.Series:
            self.df = pd.DataFrame(df)

        if impute:
            self.df = df.interpolate()
        # else:
        #     self.df = df
        self.train_size = train_size
        self.df_train, self.df_test = pm.model_selection.train_test_split(self.df,
            train_size = self.train_size/100)
        self.dates = df.index
        self.length = df.shape[0]
        self.data_name = data_name
        self.timeframe = f'{n} {periods.title()}'
        self.tf = f'{n}{periods[0].upper()}'
        self.freq = freq
        self.f = freq.split()[0] + freq.split()[1][0].upper()
        self.m = seas
        self.f_m = f_seas
        self.fit_seas = fit_seas
        self.n_diffs = start_d
        self.ns_diffs = start_D
        if estimate_diffs:
            self.__estimate_diffs()
        else:
            self.n_diffs = 1
            self.ns_diffs = 1
        self.arima_order = start_order
        self.p = start_order[0]
        self.d = start_order[1]
        self.q = start_order[2]
        self.t = trend
        self.no_intercept = no_intercept
        self.mod_order = f'({self.p}, {self.d}, {self.q})[\'{self.t}\']'
        self.date = date
        self.fourier = fourier
        self.box = box
        self.log = log
        self.AA_best_params, self.AA_mod_pipe = self.__reset_mod_params()
        self.GS_best_params, self.GS_best_mod_pipe = self.__reset_mod_params()
        self.mod_params, self.mod_pipe = self.__reset_mod_params()
        self.__train_test_split_dates()
        self.y_hat = None
        self.conf_ints = None
        self.RMSE = np.inf
        self.GS_first_mod = True
        print('Successfully created instance of Class Pmdarima_Model.') if verbose else None

    def __estimate_diffs(self):
        # Calculate diffs to use
        kpss_diffs = pm.arima.ndiffs(self.df_train, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = pm.arima.ndiffs(self.df_train, alpha=0.05, test='adf', max_d=6)
        self.n_diffs = max(adf_diffs, kpss_diffs)

        if self.fit_seas:
            ocsb_diffs = pm.arima.nsdiffs(self.df_train, m=self.m, test='ocsb', max_D=6)
            ch_diffs = pm.arima.nsdiffs(self.df_train, m=self.m, test='ch', max_D=6)
            self.ns_diffs = max(ocsb_diffs, ch_diffs)

    def __reset_mod_params(self):
        mod_pipe = None
        mod_params = 'ARIMA Order()'
        mod_params = mod_params.replace('()', self.mod_order)
        if not self.no_intercept:
            mod_params += ', Intercept'
        if self.date:
            mod_params += ', DateFeaturizer'
        if self.fourier:
            mod_params += ', FourierFeaturizer'
        if self.box:
            mod_params += ', BoxCoxEndogTransformer'
        if self.log:
            mod_params += ', LogEndogTransformer'

        return mod_params, mod_pipe

    def __get_model_params(self, p, d, q, t, no_intercept, date, fourier, box, log, verbose=1):
        date_feat = pm.preprocessing.DateFeaturizer(
            column_name="date",  # the name of the date feature in the X matrix
            with_day_of_week=True,
            with_day_of_month=True)
        _, X_train_feats = date_feat.fit_transform(self.y_train, self.X_train)

        arima_params = dict(
            trace=3,
            maxiter=200,
            suppress_warnings=True)

        arima_order = (p, d, q)
        order = f'({p}, {d}, {q})[\'{t}\']'
        mod_params = 'ARIMA Order()'
        mod_params = mod_params.replace('()', order)
        arima_params['order'] = arima_order
        arima_params['trend'] = t
        pipe_params = []
        if not no_intercept:
            arima_params['with_intercept'] = True
            mod_params += ', Intercept'
        if date:
            pipe_params.append(('date', date_feat))
            mod_params += ', DateFeaturizer'
        if fourier:
            pipe_params.append(('fourier', pm.preprocessing.FourierFeaturizer(m=self.f_m, k=4)))
            mod_params += ', FourierFeaturizer'
        if box:
            pipe_params.append(('box', pm.preprocessing.BoxCoxEndogTransformer()))
            mod_params += ', BoxCoxEndogTransformer'
        if log:
            pipe_params.append(('log', pm.preprocessing.LogEndogTransformer()))
            mod_params += ', LogEndogTransformer'

        pipe_params.append(('arima', pm.arima.ARIMA(**arima_params)))
        print(mod_params) if verbose >= 1 else None
        print(pipe_params) if verbose >= 2 else None
        pipe = pipeline.Pipeline(pipe_params)
        return mod_params, pipe

    def __train_test_split_dates(self):
        self.X_train, self.y_train, self.X_test, self.y_test = self.__split_df_dates(self.df_train, self.df_test)

    def __run_stepwise_CV(self, model=None, dynamic=False, verbose=1, debug=False):
        def __forecast_one_step(date_df):
            fc, conf_int = model.predict(X=date_df, return_conf_int=True)
            return (
                # fc.tolist()[0],
                fc.tolist()[0],
                # np.asarray(conf_int).tolist()[0]
                conf_int[0].tolist()
                )
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        # try:
        #     assert(hasattr(model.named_steps['arima'], 'model_')), "Hmm"
        # except AssertionError:
        #     print('Model has not been fitted. Fitting now.')
            # model.fit(y_train, X_train)
        # model = self.mod_pipe
        # date = pd.DataFrame([X_test.iloc[0].date], columns=['date'])
        date_df = pd.DataFrame([X_test.iloc[0].date], index=[X_train.size], columns=['date'])
        # date = X_test.iloc[0].date
        forecasts = []
        conf_ints = []
        dynamic_str = ''
        if dynamic:
            dynamic_str = ' dynamically with forecasted values'
        if verbose:
            print(f'Iteratively making predictions on {self.data_name} Time Series and updating model{dynamic_str}, beginning with first index of y_test ...')
        for n, new_ob in enumerate(tqdm(y_test, desc='Step-Wise Prediction Loop')):
            fc, conf = __forecast_one_step(date_df)
            forecasts.append(fc)
            conf_ints.append(conf)

            # Updates the existing model with a small number of MLE steps
            if dynamic:
                model.update([fc], date_df)
            elif not dynamic:
                model.update([new_ob], date_df)
            if n&1:
                print('>_', end='\r')
            else:
                print('> ', end='\r')
            # date = pd.DataFrame([X_test.iloc[0].date + CBD], index=[X_train.size]columns=['date'])
            date_df.iloc[0].date += CBD
            date_df.index += 1
            # date += CBD
        print('Done.')
        # y_hat = pd.Series(forecasts, index=y_test.index)
        self.conf_ints = conf_ints
        self.y_hat = forecasts

        return X_train, y_train, X_test, y_test, forecasts, conf_ints

    def __GS_score_model(self, model, verbose=0, debug=False):
        print('_____________________________________________________________________')
        print('\nStarting step-wise cross-validation...')
        if verbose:
            print(model[0])
            print(model[1])
        model[1].fit(self.y_train, self.X_train)
        result = None
        # convert config to a key
        key = str(model[0])
        # print(key)
        # model[1].fit(y_train, X_train)
        # show all warnings and fail on exception if debugging
        if debug:
            X_train, X_test, y_train, y_test, y_hat, conf_ints = self.__run_stepwise_CV(model[1])
            result = self.__get_model_scores(y_test, y_hat, y_train, model=model[1], debug=debug)
        else:
            # one failure during model validation suggests an unstable config
            try:
                # never show warnings when grid searching, too noisy
                with catch_warnings():
                    filterwarnings("ignore")
                    X_train, X_test, y_train, y_test, y_hat, conf_ints = self.__run_stepwise_CV(model[1])
                    result = self.__get_model_scores(y_test, y_hat, y_train, model=model[1], debug=debug)
            except:
                error = None
                # check for an interesting result
        if result:
            print('Model[%s]: AIC=%.3f | RMSE=%.3f | SMAPE=%.3f%%' % (key, *result))
            if result[1] < self.RMSE:
                self.RMSE = result[1]
                self.y_hat = y_hat
                self.conf_ints = conf_ints
                self.GS_best_mod_pipe = model[1]
                self.GS_best_params = key
                if self.GS_first_mod:
                    print('First viable model found, RMSE=%.3f' % result[1])
                    self.GS_first_mod = False
                else:
                    print('Next best model found, RMSE=%.3f' % result[1])

        return (key, *result)

    def __gridsearch_CV(self, max_order=6, t_list=['n','c','t','ct'], no_intercept=False, date=True, fourier=True, box=False,
        log=False, verbose=0, debug=False, parallel=True):
        def __GS_setup_params(t_list=['n','c','t','ct'], no_intercept=False, date=True, fourier=True, box=False, log=False, verbose=0):

            # date_feat = pm.preprocessing.DateFeaturizer(
            # column_name="date",  # the name of the date feature in the X matrix
            # with_day_of_week=True,
            # with_day_of_month=True)
            # _, X_train_feats = date_feat.fit_transform(self.y_train, self.X_train)

            # columns = ['ARIMA', 'Trend', 'Intercept', 'DateFeaturizer', 'FourierFeaturizer',
                        # 'BoxCoxEndogTransformer', 'LogEndogTransformer']

            # arima_params = dict(
            #     # d = self.n_diffs,
            #     trace=3,
            #     maxiter=200,
            #     # stepwise=True,
            #     # seasonal=False,
            #     suppress_warnings=True)
            # feats_list = 2**sum(box, log)
            inter_iter = list(set([True, not no_intercept]))
            box = list(set([box, False]))[::-1]
            log = list(set([log, False]))[::-1]
            feats_iter = list(itertools.product(box, log))
            models = []
            count = 0
            for d in range(1,self.n_diffs+1):
                mod_order = 0
                # mod_params_df = pd.DataFrame(columns=columns)
                for p in range(6):
                    for q in range(6):
                        mod_order = p+q
                        # print(mod_order)
                        if mod_order > max_order:
                            continue
                        for t in t_list:
                            for intercept in inter_iter:
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
                                for box, log in feats_iter:
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
                                    # print(pipe_params)
                                    # print('Added one more model')
                                    # pipe = pipeline.Pipeline(pipe_params)
                                    # models.append((mod_params, pipe))
                                    mod_params, pipe = self.__get_model_params(
                                        p, d, q, t, no_intercept, date, fourier, box, log, verbose)
                                    models.append((mod_params, pipe))
                                    count += 1
            print(f'Finished building list of {count} models.') if verbose else None
            # print(models) if debug else None
            return models

        def __GS_start(models, verbose=0, debug=False, parallel=True):

            print('Starting iterative search through all params.') if verbose else None
            scores = None
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
                scores = [self.__GS_score_model(model, debug=debug) for model in tqdm(models, desc='Model Loop')]
                # scores = [self.__GS_score_model(model, debug=debug) for model in models]
            # remove empty results
            scores = [score for score in scores if score[1] != None]
            # sort configs by error, asc
            scores.sort(key=lambda tup: tup[2])
            return scores

        # if __name__ == '__main__':  # <- prevent RuntimeError for 'spawn'
        models = __GS_setup_params(t_list=t_list, no_intercept=no_intercept, date=date,
            fourier=fourier, box=box, log=log, verbose=verbose)
        scores = __GS_start(models, debug=debug, parallel=parallel)
        # clear()
        print('GridsearchCV Completed.\n')
        print('Top 10 models:')
        for model, AIC, RMSE, SMAPE in scores[:10]:
            print('Model[%s]: AIC=%.3f | RMSE=%.3f | SMAPE=%.3f%%' % (model, AIC, RMSE, SMAPE))

        return self.GS_best_mod_pipe, scores

    def __run_manual_pipeline(self, train, test, n_diffs):
        return

    def __split_df_dates(self, train, test):
        X_train = pd.DataFrame(train.index)
        y_train = train.values
        X_test = pd.DataFrame(test.index, index=range(X_train.size, self.length))
        y_test = test.values
        return X_train, y_train, X_test, y_test

    def __run_auto_pipeline(self, show_summary=False, return_conf_int=False, verbose=1):
        pm.tsdisplay(self.df_train, lag_max=60, title = f'{self.data_name} Time Series Visualization') \
            if show_summary else None
        X_train, y_train, X_test, y_test = self.__split_df_dates(self.df_train, self.df_test)

        params = []
        if self.date:
            print('Using DateFeaturizer.') if verbose == 1 else None
            date_feat = pm.preprocessing.DateFeaturizer(
                column_name="date",  # the name of the date feature in the X matrix
                with_day_of_week=True,
                with_day_of_month=True)
            _, X_train_feats = date_feat.fit_transform(y_train, X_train)
            # _, X_train_feats = date_feat.fit_transform(y_train[:,0], X_train)
            params.append(('date', date_feat))
        if self.fourier:
            print('Using FourierFeaturizer.') if verbose == 1 else None
            params.append(('fourier', pm.preprocessing.FourierFeaturizer(m=self.f_m, k=4)))
        if self.box:
            print('Using BoxCoxEndogTransformer.') if verbose == 1 else None
            params.append(('box', pm.preprocessing.BoxCoxEndogTransformer()))
        if self.log:
            print('Using LogEndogTransformer.') if verbose == 1 else None
            params.append(('log', pm.preprocessing.LogEndogTransformer()))
        arima_params = dict(
            d = self.n_diffs,
            trace=3,
            maxiter=200,
            stepwise=True,
            suppress_warnings=True)
        if self.fit_seas:
            arima_params['seasonal']=True
            arima_params['m']=self.m
            arima_params['max_p']=0
            arima_params['start_p']=0
            arima_params['max_q']=0
            arima_params['start_q']=0
            arima_params['max_P']=0
            arima_params['start_P']=0
            arima_params['max_Q']=0
            arima_params['start_Q']=0
            arima_params['start_params']=np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
            arima_params['D']=self.ns_diffs
        elif not self.fit_seas:
            arima_params['seasonal']=False
        params.append(('arima', pm.arima.AutoARIMA(**arima_params)))

        print(f'Parameters for Pipeline: \n{params}\n') if verbose == 1 else None
        pipe = pipeline.Pipeline(params)
        pipe.fit(y_train, X_train)

        # save best params
        pipe_params = [(name, transform) for name, transform in pipe.named_steps.items()]
        pkl_out = open(f'{top}/models/3Y_AA_best_model.pkl', 'wb')
        pkl.dump(pipe_params, pkl_out)
        pkl_out.close()
        best_arima = pipe.named_steps['arima'].model_
        self.AA_best_params = self.AA_best_params.replace('()', str(best_arima.order))
        self.AA_mod_pipe = pipe

        # run prediction on test set
        # conf_ints = []
        if return_conf_int:
            self.y_hat, self.conf_ints = pipe.predict(X=X_test, return_conf_int=return_conf_int)
        elif not return_conf_int:
            self.y_hat = pipe.predict(X=X_test)
        # print("Test RMSE: %.3f" % mse(y_test, y_hat, squared=False))

        return X_train, y_train, X_test, y_test, self.y_hat, self.conf_ints

    def plot_test_predict(self, y_hat=None, conf_ints=True, ylabel=None, fin=True, func='AA'):
        '''
        Plot Test vs Predict with optional confidence intervals
        '''
        conf = ''
        ylabel=self.data_name
        ts = ylabel.replace(' ', '_')
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
            plt.savefig(f'{top}/images/AutoArima/{ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')
        elif func == 'GS':
            ax.set_title(f'GridSearch Best Parameters: {self.GS_best_params}', size=16)
            plt.savefig(f'{top}/images/GridSearch/{ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')
        else:
            ax.set_title(f'Parameters: {self.mod_params}', size=16)
            plt.savefig(f'{top}/images/{ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')

    def plot_forecast_conf(self, ohlc_df=None, hist_df=None, y_hat=None, conf_ints=True, days_fc=5,
            lookback=120, ylabel=None, fin=False, func='GS'):
        '''
        Plot forecasts with optional confidence intervals. Can only be run after
        forecasts have been generated.
        '''
        ylabel=self.data_name
        ts = ylabel.replace(' ', '_')
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
            plt.savefig(f'{top}/images/AutoArima/{ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf}.png')
        elif func == 'GS':
            ax.set_title(f'GridSearch Best Parameters: {self.GS_best_params}', size=16)
            plt.savefig(f'{top}/images/GridSearch/{ts}_{self.tf}_{self.f}_Hist_vs_Forecast{conf}.png')
        else:
            ax.set_title(f'Parameters: {self.mod_params}', size=16)
            plt.savefig(f'{top}/images/{ts}_{self.tf}_{self.f}_Test_vs_Predict{conf}.png')

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
        SMAPE = smape(y_test, y_hat)
        if verbose:
            print("Test AIC: %.3f" % AIC)
            print("Test RMSE: %.3f" % RMSE)
            print("This is %.3f%% of the avg observed value." % (100*RMSE/y_train.mean()))
            print("Test SMAPE: %.3f%%\n" % SMAPE)
        if debug:
            print("AIC: %.3f | RMSE: %.3f | SMAPE %.3f%%%" % (AIC, RMSE, SMAPE))
        return AIC, RMSE, SMAPE

    def run_stepwise_CV(self, model=None, func=None, dynamic=False, verbose=1, visualize=True, conf_ints=True):
        model_str = ''
        if not model:
            model = self.AA_mod_pipe
            model_str = ' on best model from AutoArima.'
            func = 'AA'
        if verbose:
            print(model)
            print(f'Starting step-wise cross-validation{model_str}...')
        X_train, y_train, X_test, y_test, y_hat, conf_ints = self.__run_stepwise_CV(model=model, dynamic=dynamic)
        AIC, RMSE, SMAPE = self.__get_model_scores(y_test, y_hat, y_train, model=model, verbose=verbose)
        if visualize:
            # self.plot_test_predict(y_hat, ylabel=self.data_name, func=func)
            self.plot_test_predict(y_hat, conf_ints=conf_ints, ylabel=self.data_name, func=func)
        return AIC, RMSE, SMAPE

    def run_auto_pipeline(self, show_summary=False, verbose=1, visualize=True, conf_ints=True):
        if verbose:
            print('Starting AutoARIMA...')
            print(f'Data set diffs to use: {self.n_diffs}')
            if self.fit_seas:
                print(f'Data set seasonal diffs to use: {self.ns_diffs}')
        self.AA_best_params, self.AA_mod_pipe = self.__reset_mod_params()
        X_train, y_train, X_test, y_test, y_hat, conf_ints = \
            self.__run_auto_pipeline(show_summary=show_summary, return_conf_int=True)
        self.__get_model_scores(y_test, y_hat, y_train, model=self.AA_mod_pipe, verbose=verbose)
        if visualize:
            # self.plot_test_predict(y_hat, ylabel=self.data_name, func='AA')
            self.plot_test_predict(y_hat, conf_ints=conf_ints, ylabel=self.data_name, func='AA')

        return self.AA_mod_pipe
        # return df_train, df_test, exog_train, exog_test
        # return X_train, X_test, exog_train, exog_test

    def run_gridsearch_CV(self, max_order=6, t_list=['n','c','t','ct'], no_intercept=False,
                            date=True, fourier=True, box=False, log=False, visualize=True,
                            conf_ints=True, verbose=1, debug=False, parallel=True):
        if verbose:
            print('Starting GridSearchCV...')
        self.GS_best_params, self.GS_best_mod_pipe = self.__reset_mod_params()
        best_model, scores = self.__gridsearch_CV(max_order=max_order, t_list=t_list,
            no_intercept=no_intercept, date=date, fourier=fourier, box=box, log=log,
            verbose=verbose, debug=debug, parallel=parallel)
        if visualize:
            # self.plot_test_predict(self.y_hat, conf_ints=conf_ints, ylabel=self.data_name, func='GS')
            self.plot_test_predict(self.y_hat, conf_ints=conf_ints, ylabel=self.data_name, func='GS')
        return best_model, scores

    def __fit_predict(self, model, days_fc, new_dates, index_fc, hist_dates_df, en_ex, new_dates_df=None, exog_df=None, verbose=1):
        model.fit(self.df, hist_dates_df)
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

    def run_prediction(self, model, days_fc, en_ex, exog_df=None, visualize=True,
                        fin=False, ohlc_df=None, hist_df=None, func='GS', verbose=1):
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

        if not model:
            model = self.AA_mod_pipe

        if verbose:
            var = None
            if en_ex == 'exog':
                var = 'Exogenous'
            elif en_ex == 'endo':
                var = 'Endogenous'
            print(f'Running Fit and Predict on {var} variable {self.data_name}...')

        self.days_fc = days_fc
        hist_dates_df = pd.DataFrame(self.df.index, columns=['date'])
        today = self.df.index[-1]
        df_size = self.length
        new_dates, index_fc, new_dates_df = Pmdarima_Model.get_next_dates(today, df_size, days_fc)
        self.new_dates = new_dates
        self.index_fc = index_fc
        self.new_dates_df = new_dates_df

        # run Fit/Predict
        # note that the fc_dt returned for exogenous data does not include date
        fc_df, y_hat, conf_ints = self.__fit_predict(model, days_fc, new_dates,
            index_fc, hist_dates_df, en_ex, new_dates_df, exog_df, verbose)
        self.fc_df = fc_df
        self.y_hat = y_hat
        self.conf_ints = conf_ints

        if visualize:
            if fin:
                self.plot_forecast_conf(ohlc_df=ohlc_df, y_hat=y_hat, conf_ints=True, func=func, fin=fin, days_fc=days_fc)
            else:
                self.plot_forecast_conf(hist_df=hist_df, y_hat=y_hat, conf_ints=True, func=func, fin=fin, days_fc=days_fc)
        return fc_df, y_hat, new_dates_df, conf_ints

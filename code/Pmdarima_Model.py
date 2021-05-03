

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
print('Pmdarima_Model.py loaded.')

from pathlib import Path
top = Path(__file__ + '../../..').resolve()

class Pmdarima_Model:
    def __init__(self, endo, exog, endo_name, exog_name, train_size,
                        n, period, freq, seas, impute=0, verbose=1,
                        # max_d=2, max_p=2, max_q=2, max_D=2, max_P=2, max_Q=2,
                        date=1, fourier=0, box=0, log=0, gridsearch=0):

# def pdarima_fit_predict(endo, exog, endo_name, exog_name, train_size,
#                         n, period, freq, seas, impute=1, verbose=1, extra=0,
#                         gridsearch=0):
        try:
            assert(type(endo)==pd.Series), 'Endogenous variable is not of type Pandas Series.'
            assert(type(exog)==pd.Series), 'Exogenous variable is not of type Pandas Series.'
        except AssertionError:
            raise

        self.dates = endo.index
        self.length = endo.index.size
        if impute==1:
            self.endo = endo.interpolate()
            self.exog = exog.interpolate()
        else:
            self.endo = endo
            self.exog = exog
        self.endo_name = endo_name
        self.exog_name = exog_name
        self.train_size = train_size
        # self.n = n
        # self.period = period
        self.timeframe = f'{n} {period.title()}'
        self.tf = f'{n}{period[0].upper()}'
        self.freq = freq
        self.f = freq.split()[0] + freq.split()[1][0].upper()
        self.seas = seas
        self.verbose = verbose
        self.date = date
        self.fourier = fourier
        self.box = box
        self.log = log
        self.__reset_best_params()
        # if gridsearch==0:
        # self.endo_train, self.endo_test = train_test_split_data(self.endo, self.train_size, verbose=self.verbose)
        # self.exog_train, self.exog_test = train_test_split_data(self.exog, self.train_size, verbose=self.verbose)

        self.endo_train, self.endo_test = pm.model_selection.train_test_split(self.endo,
            train_size = self.train_size/100)
        self.exog_train, self.exog_test = pm.model_selection.train_test_split(self.exog,
            train_size = self.train_size/100)

        # self.gridsearch = gridsearch
        kpss_diffs = ndiffs(self.endo_train, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = ndiffs(self.endo_train, alpha=0.05, test='adf', max_d=6)
        self.n_diffs_en = max(adf_diffs, kpss_diffs)

        kpss_diffs = ndiffs(self.exog_train, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = ndiffs(self.exog_train, alpha=0.05, test='adf', max_d=6)
        self.n_diffs_ex = max(adf_diffs, kpss_diffs)
        print('Successfully created instance of Class Pmdarima_Model.') if verbose==1 else None

    def __reset_best_params(self):
        self.endo_best_params = 'ARIMA Order()'
        self.exog_best_params = 'ARIMA Order()'
        if self.date == 1:
            self.endo_best_params += ', DateFeaturizer'
            self.exog_best_params += ', DateFeaturizer'
        if self.fourier == 1:
            self.endo_best_params += ', FourierFeaturizer'
            self.exog_best_params += ', FourierFeaturizer'
        if self.box == 1:
            self.endo_best_params += ', BoxCoxEndogTransformer'
            self.exog_best_params += ', BoxCoxEndogTransformer'
        if self.log == 1:
            self.endo_best_params += ', LogEndogTransformer'
            self.exog_best_params += ', LogEndogTransformer'
        self.endo_pipe = None
        self.exog_pipe = None

    def __fit_predict(self, endo, exog, endo_name, exog_name, train_size,
                            n, period, freq, seas, impute=1, verbose=1, extra=0,
                            gridsearch=0):
        '''
        Run auto_arima full pipe to fit and predict in-sample values.
        Only defined for EOD prices.
        '''

        # dates =endo.index

    def __run_stepwise_CV(self, model, y_train):
        def forecast_one_step(model):
            fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
            return (
                fc.tolist()[0],
                np.asarray(conf_int).tolist()[0])

        forecasts = []
        conf_ints = []

        for new_ob in y_test:
            fc, conf = forecast_one_step()
            forecasts.append(fc)
            conf_ints.append(conf)

            # Updates the existing model with a small number of MLE steps
            model.update(new_ob)
            print(new_ob)

        # y_hat = pd.Series(forecasts, index=y_test.index)
        y_hat = forecasts
        self.__calc_RMSE(y_test, y_hat, y_train)
        print(f"SMAPE: {smape(y_test, forecasts)}%")

        return

    def __gridsearch_rmse(self, train, test, n_diffs):
        for d in range(n_diffs):
            for p in range(3):
                for q in range(3):
                    print(f'Order = ({p}, {d}, {q}') if verbose==1 else None


        return None

    def __run_manual_pipeline(self, train, test, n_diffs):
        pass

    def __run_auto_pipeline(self, train, test, n_diffs, en_ex, return_conf_int):
        pm.tsdisplay(train, lag_max=60)

        X_train = pd.DataFrame(train.index)
        y_train = train.values
        X_test = pd.DataFrame(test.index)
        y_test = test.values

        params = []
        if self.date == 1:
            print('Using DateFeaturizer.') if self.verbose == 1 else None
            date_feat = pm.preprocessing.DateFeaturizer(
                column_name="date",  # the name of the date feature in the X matrix
                with_day_of_week=True,
                with_day_of_month=True)
            _, X_train_feats = date_feat.fit_transform(y_train, X_train)
            params.append(('date', date_feat))
        if self.fourier == 1:
            print('Using FourierFeaturizer.') if self.verbose == 1 else None
            params.append(('fourier', pm.preprocessing.FourierFeaturizer(m=self.seas, k=4)))
        if self.box == 1:
            print('Using BoxCoxEndogTransformer.') if self.verbose == 1 else None
            params.append(('box', pm.preprocessing.BoxCoxEndogTransformer()))
        if self.log == 1:
            print('Using LogEndogTransformer.') if self.verbose == 1 else None
            params(('log', pm.preprocessing.LogEndogTransformer()))
        params.append(('arima', pm.arima.AutoARIMA(d=n_diffs,
                trace=3,
                stepwise=True,
                suppress_warnings=True,
                seasonal=False
                )))
        print(f'Parameters for Pipeline: \n{params}\n') if self.verbose == 1 else None
        pipe = pipeline.Pipeline(params)
        # pipe = pipeline.Pipeline([
        #     ('date', date_feat),
        #     # ("fourier", pm.preprocessing.FourierFeaturizer(m=self.seas, k=4)),
        #     ('arima', pm.arima.AutoARIMA(d=n_diffs,
        #         trace=3,
        #         stepwise=True,
        #         suppress_warnings=True,
        #         seasonal=False))
        #         ])
        pipe.fit(y_train, X_train)

        # save best params
        pipe_params = [(name, transform) for name, transform in pipe.named_steps.items()]
        pkl_out = open(f'{top}/models/3Y_best_arima.pkl', 'wb')
        pkl.dump(pipe_params, pkl_out)
        pkl_out.close()
        best_arima = pipe.named_steps['arima'].model_
        if en_ex == 'endo':
            self.endo_best_params = self.endo_best_params.replace('()', str(best_arima.order))
        elif en_ex == 'exog':
            self.exog_best_params = self.exog_best_params.replace('()', str(best_arima.order))

        # run prediction
        conf_ints = []
        if return_conf_int == True:
            y_hat, conf_ints = pipe.predict(X=X_test, return_conf_int=return_conf_int)
        elif return_conf_int == False:
            y_hat = pipe.predict(X=X_test)
        # print("Test RMSE: %.3f" % mse(y_test, y_hat, squared=False))

        return X_train, y_train, X_test, y_test, y_hat, pipe, conf_ints

    def __plot_test_predict(self, X_train, y_train, X_test, y_test, y_hat, ylabel, auto=False):

        # n_train = y_train.shape[0]
        # = np.arange('2005-02', '2005-03', dtype='datetime64[D]')
        # params = f'ARIMA Order{best_arima.order}'
        # if self.date == 1:
        #     params += ', DateFeaturizer'
        # if self.fourier == 1:
        #     params += ', FourierFeaturizer'
        # if self.box == 1:
        #     params += ', BoxCoxEndogTransformer'
        # if self.log == 1:
        #     params += ', LogEndogTransformer'
        ts = ylabel.replace(' ', '_')
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
        ax.plot(X_test, y_hat, color='green', marker=',', alpha=0.5, label='Predicted')
        ax.plot(X_test, y_test, color='red', alpha=0.5, label='Actual')
        ax.legend(loc='lower left', borderaxespad=0.5)
        fig.suptitle(f'{ylabel} Time Series, {self.timeframe}, Freq = {self.freq}: Test vs Predict\n', size=18)
        ax.set_title(f'Parameters: {self.best_params}', size=16)
        ax.set_ylabel(ylabel, size=14)
        if auto == True:
            plt.savefig(f'{top}/images/AutoArima/{ts}_{self.tf}_{self.f}_Test_vs_Predict.png')
        else:
            plt.savefig(f'{top}/images/{ts}_{self.tf}_{self.f}_Test_vs_Predict.png')

    def __plot_test_predict_conf(self, X_train, y_train, X_test, y_test, y_hat, ylabel, en_ex, auto=False, conf_ints=None):
        '''
        Plot Test vs Predict with optional confidence intervals
        '''
        print(X_test)
        print(type(X_test))
        ts = ylabel.replace(' ', '_')
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
        ax.plot(X_test, y_hat, color='green', marker=',', alpha=0.5, label='Predicted')
        if conf_ints is None:
            ax.plot(X_test, y_test, color='red', alpha=0.5, label='Actual')
        else:
            conf_int = np.asarray(conf_ints)
            ax.fill_between(X_test.date,
                     conf_int[:, 0], conf_int[:, 1],
                     alpha=0.5, color='orange',
                     label="Confidence Intervals")
        ax.legend(loc='lower left', borderaxespad=0.5)
        fig.suptitle(f'{ylabel} Time Series, {self.timeframe}, Freq = {self.freq}: Test vs Predict with Confidence Interals\n', size=18)
        if en_ex == 'endo':
            ax.set_title(f'Parameters: {self.endo_best_params}', size=16)
        elif en_ex == 'exog':
            ax.set_title(f'Parameters: {self.exog_best_params}', size=16)
        ax.set_ylabel(ylabel, size=14)
        if auto == True:
            plt.savefig(f'{top}/images/AutoArima/{ts}_{self.tf}_{self.f}_Test_vs_Predict_Conf.png')
        else:
            plt.savefig(f'{top}/images/{ts}_{self.tf}_{self.f}_Test_vs_Predict_Conf.png')

    def __calc_RMSE(self, y_test, y_hat, y_train):
        RMSE = mse(y_test, y_hat, squared=False)
        print("Test RMSE: %.3f" % RMSE)
        print("This is %.2f%% of the avg observed value.\n" % (100*RMSE/y_train.mean()))

    def run_stepwise_cv(self):
        print('Starting Step-Wise Cross-Validation Sequence...')
        # X_train, y_train, X_test, y_test, y_hat = \
        y_hat, conf_ints = self.__run_stepwise_CV(self.endo_train, self.endo_test, self.n_diffs_en)
        self.__plot_test_predict_conf(X_train, y_train, X_test, y_test, y_hat,
                ylabel=self.endo_name, en_ex='endo', auto=False)
        self.__plot_test_predict_conf(X_train, y_train, X_test, y_test, y_hat,
                ylabel=self.endo_name, en_ex='endo', auto=False, conf_ints=conf_ints)

        # X_train, y_train, X_test, y_test, y_hat = \
        y_hat, conf_ints = self.__run_stepwise_CV(self.exog_train, self.exog_test, self.n_diffs_ex)
        self.__plot_test_predict_conf(X_train, y_train, X_test, y_test, y_hat,
                ylabel=self.exog_name, en_ex='exog', auto=False)
        self.__plot_test_predict_conf(X_train, y_train, X_test, y_test, y_hat,
                ylabel=self.exog_name, en_ex='exog', auto=False, conf_ints=conf_ints)

    def run_auto(self):
        if self.verbose == 1:
            print('Starting AutoARIMA sequence...')
            print(f'Endogenous data set diffs to use: {self.n_diffs_en}')
            print(f'Exogenous data set diffs to use:  {self.n_diffs_ex}')
        self.__reset_best_params()
        X_train, y_train, X_test, y_test, y_hat, conf_ints, self.endo_pipe = \
            self.__run_auto_pipeline(self.endo_train, self.endo_test, self.n_diffs_en, en_ex='endo', return_conf_int=True)
        self.__calc_RMSE(y_test, y_hat, y_train)
        self.__plot_test_predict_conf(X_train, y_train, X_test, y_test, y_hat,
                ylabel=self.endo_name, en_ex='endo', auto=True)
        self.__plot_test_predict_conf(X_train, y_train, X_test, y_test, y_hat,
                ylabel=self.endo_name, en_ex='endo', auto=True, conf_ints=conf_ints)

        X_train, y_train, X_test, y_test, y_hat, conf_ints, self.exog_pipe = \
            self.__run_auto_pipeline(self.exog_train, self.exog_test, self.n_diffs_ex, en_ex='exog', return_conf_int=True)
        self.__calc_RMSE(y_test, y_hat, y_train)
        self.__plot_test_predict_conf(X_train, y_train, X_test, y_test, y_hat,
                ylabel=self.exog_name, en_ex='exog', auto=True)
        self.__plot_test_predict_conf(X_train, y_train, X_test, y_test, y_hat,
                ylabel=self.exog_name, en_ex='exog', auto=True, conf_ints=conf_ints)

        return pipe
        # return endo_train, endo_test, exog_train, exog_test
        # return X_train, X_test, exog_train, exog_test

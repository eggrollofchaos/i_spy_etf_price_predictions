from tqdm.notebook import trange, tqdm
import pandas as pd
import matplotlib
import numpy as np
from itertools import product
from functools import reduce
import pickle as pkl

import time
import datetime
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

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV,\
    cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, plot_confusion_matrix,\
    precision_score, recall_score, accuracy_score, f1_score, log_loss,\
    roc_curve, roc_auc_score, classification_report
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRFClassifier

import matplotlib.pyplot as plt
import seaborn as sns
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

from functions import *
from pathlib import Path
top = Path(__file__ + '../../..').resolve()

font = {'size'   : 12}
matplotlib.rc('font', **font)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',25)
NYSE = mcal.get_calendar('NYSE')
CBD = NYSE.holidays()

print(f'regressions.py loaded from {top}/data.')

################################################################################

def create_data_sets(left, right, verbose=True):
    cat_df = left.join(right)
    cat_df['close_diff'] = cat_df.close.diff()
    cat_df['change'] = np.where(cat_df.close_diff >= 0, 1, 0)

    X = cat_df.drop(['close', 'adj_close', 'open', 'high', 'low', 'volume', 'close_diff'], axis=1)

    # y = cat_df.drop(y_setup.index[0]).change
    y = cat_df.pop('change')

    if verbose:
        print('X:\n', X)
        print('y:\n', y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = .20,
        random_state = 729)

    return X, y

def build_pipeline(X_train, verbose=1):
    numerical_pipeline = Pipeline(steps=[
        ('rs', RobustScaler())
    ])

    # categorical_pipeline = Pipeline(steps=[
    #     ('ohe', OneHotEncoder( #drop='first',
    #                          sparse=False,
    #                          handle_unknown='ignore'))
    # ])

    trans = ColumnTransformer(transformers=[
        ('numerical', numerical_pipeline, X_train.columns),
    #     ('categorical', categorical_pipeline, X_train.columns)
    ])

    pipe = Pipeline(steps=[
        ('trans', trans),
        ('lr', LogisticRegression(random_state=1, max_iter=500))
    ])
    print(pipe) if verbose else None
    return pipe


def run_model_gridsearch_CV(X_y_train_test, clf_type='log', verbose=1):
    pipe = build_pipeline(X_y_train_test[0])
    if clf_type=='log':
        class_str = 'LogisticRegression'
        params_grid = {'lr__penalty' : ['l1', 'l2','elasticnet'],
                          'lr__class_weight' : ['balanced', 'none'],
                          'lr__dual' : [True, False],
                          'lr__solver' : ['lbfgs', 'liblinear'],
                          'lr__C' : np.logspace(-4, 4, 5),
                          'lr__l1_ratio' : np.logspace(-4, 4, 5) # only needed for elasticnet
                         }

    clf = GridSearchCV(pipe, param_grid = params_grid, cv = 3, verbose=4, n_jobs=-1)

    # clf.fit(X_train, y_train)
    clf.fit(X_y_train_test[0], X_y_train_test[2])
    print(clf.best_params_) if verbose else None
    best_clf = clf.best_estimator_
    # model_stats(X.columns, best_clf, class_str, X_test, y_test, binary=True)
    model_stats(X_y_train_test[0].columns, best_clf, class_str, X_y_train_test[1],
        X_y_train_test[3], binary=True)

    return best_clf, y_pred

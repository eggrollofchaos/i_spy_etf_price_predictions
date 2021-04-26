import pandas as pd
import matplotlib
import numpy as np
from sklearn import metrics

import itertools
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
plt.style.use('ggplot')
sns.set_theme(style="darkgrid")

font = {'size'   : 12}
matplotlib.rc('font', **font)

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',25)

# only display whole years in figures
years = mdates.YearLocator()
years_fmt = mdates.DateFormatter('%Y')
print('Functions loaded.')

def melt_data(df):
    '''
    Takes in a Zillow Housing Data File (ZHVI) as a DataFrame in wide format
    and returns a melted DataFrame
    '''
    melted = pd.melt(df, id_vars=['RegionID', 'RegionName', 'City', 'State', 'StateName', 'Metro', 'CountyName', 'SizeRank', 'RegionType'], var_name='date')
    melted['date'] = pd.to_datetime(melted['date'], infer_datetime_format=True)
    melted = melted.dropna(subset=['value'])
    return melted

def visualize_data(df):
    pass

def create_df_dict(df):
    zipcodes = list(set(df.zipcode))
    keys = [zipcode for zipcode in map(str,zipcodes)]
    data_list = []

    for key in keys:
        new_df = df[df.zipcode == int(key)]
        new_df.drop('zipcode', inplace=True, axis=1)
        new_df.columns = ['date', 'value']
        new_df.date = pd.to_datetime(new_df.date)
        new_df.set_index('date', inplace=True)
        new_df = new_df.asfreq('M')
        data_list.append(new_df)

    df_dict = dict(zip(keys, data_list))

    return df_dict

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

def test_stationarity_all_zips(df_dict, diffs=0):
    for zipcode, df in df_dict.items():
        if diffs == 2:
            dftest = adfuller(df.diff().diff().dropna())
        elif diffs == 1:
            dftest = adfuller(df.diff().dropna())
        else:
            dftest = adfuller(df)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value', '#Lags Used',
                                             'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' %key] = value
        print(dfoutput[1])

def plot_pacf_housing(df_all, bedrooms):
    pacf_fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    pacf_fig.suptitle(f'Partial Autocorrelations of {bedrooms}-Bedroom Time Series for Entire San Francisco Data Set', fontsize=18)
    plot_pacf(df_all, ax=ax[0])
    ax[0].set_title('Undifferenced PACF', size=14)
    ax[0].set_xlabel('Lags', size=14)
    ax[0].set_ylabel('PACF', size=14)
    plot_pacf(df_all.diff().dropna(), ax=ax[1])
    ax[1].set_title('Differenced PACF', size=14)
    ax[1].set_xlabel('Lags', size=14)
    ax[1].set_ylabel('PACF', size=14)
    pacf_fig.tight_layout()
    pacf_fig.subplots_adjust(top=0.9)
    plt.savefig(f'images/{bedrooms}_bdrm_PACF.png')

def plot_acf_housing(df_all, bedrooms):
    acf_fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    acf_fig.suptitle(f'Autocorrelations of {bedrooms}-Bedroom Time Series for Entire San Francisco Data Set', fontsize=18)
    plot_acf(df_all, ax=ax[0])
    ax[0].set_title('Undifferenced ACF', size=14)
    ax[0].set_xlabel('Lags', size=14)
    ax[0].set_ylabel('ACF', size=14)
    plot_acf(df_all.diff().dropna(), ax=ax[1])
    ax[1].set_title('Once-Differenced ACF', size=14)
    ax[1].set_xlabel('Lags', size=14)
    ax[1].set_ylabel('ACF', size=14)
    plot_acf(df_all.diff().diff().dropna(), ax=ax[2])
    ax[2].set_title('Twice-Differenced ACF', size=14)
    ax[2].set_xlabel('Lags', size=14)
    ax[2].set_ylabel('ACF', size=14)
    acf_fig.tight_layout()
    acf_fig.subplots_adjust(top=0.9)
    plt.savefig(f'images/{bedrooms}_bdrm_PACF.png')

def plot_seasonal_decomposition(df_all, bedrooms):
    decomp = seasonal_decompose(df_all, freq=12)
    dc_obs = decomp.observed
    dc_trend = decomp.trend
    dc_seas = decomp.seasonal
    dc_resid = decomp.resid
    dc_df = pd.DataFrame({"observed": dc_obs, "trend": dc_trend,
                            "seasonal": dc_seas, "residual": dc_resid})
    start = dc_df.iloc[:, 0].index[0]
    end = dc_df.iloc[:, 0].index[-1] + relativedelta(months=+15) + relativedelta(day=31)

    decomp_fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    for i, ax in enumerate(axes):
        ax.plot(dc_df.iloc[:, i])
        ax.set_xlim(start, end)
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        ax.set_ylabel(dc_df.iloc[:, i].name)
        if i != 2:
            ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.setp(ax.xaxis.get_majorticklabels(), ha="right", rotation=45, rotation_mode="anchor")

    decomp_fig.suptitle(
        f'Seasonal Decomposition of {bedrooms}-Bedroom Time Series of San Francisco Home Values (Mean)', fontsize=24)
    decomp_fig.tight_layout()
    decomp_fig.subplots_adjust(top=0.94)
    plt.savefig(f'images/{bedrooms}_bdrm_seasonal_decomp.png')

def train_test_split_housing(df_dict, split=84):
    print(f'Using a {split}/{100-split} train-test split...')
    cutoff = [round((split/100)*len(df)) for zipcode, df in df_dict.items()]
    train_dict_list = [df_dict[i][:cutoff[count]] for count, i in enumerate(list(df_dict.keys()))]
    train_dict = dict(zip(list(df_dict.keys()), train_dict_list))
    test_dict_list = [df_dict[i][cutoff[count]:] for count, i in enumerate(list(df_dict.keys()))]
    test_dict = dict(zip(list(df_dict.keys()), test_dict_list))
    return train_dict, test_dict

def gridsearch_SARIMAX(train_dict, seas = 12, p_min=2, p_max=2, q_min=0, q_max=0, d_min=1, d_max=1,
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

    zipcodes = []
    param_list = []
    param_seasonal_list = []
    aic_list = []

    for zipcode, train in train_dict.items():
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                mod = SARIMAX(train,
                              order=param,
                              seasonal_order=param_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                zipcodes.append(zipcode[-5:])
                param_list.append(param)
                param_seasonal_list.append(param_seasonal)
                aic = mod.fit().aic
                aic_list.append(aic)
                print(f'Zip code {zipcode}: {aic}')
    return zipcodes, param_list, param_seasonal_list, aic_list

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
        plt.savefig(f'images/{bedrooms}_bdrm_test_predict{zipcode}.png')

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
    plt.savefig(f'images/{bedrooms}_bdrm_RMSE.png')

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
        plt.savefig(f'images/1_bdrm_forecast_{zipcode}.png')
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
    plt.savefig(f'images/{bedrooms}_bdrm_home_values_forecast.png')

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
    plt.savefig(f'images/final_forecasts.png')

def best_3_zipcodes(sorted_df, bedrooms):
    print(f'The zipcodes with the greatest projected growth in mid-tier {bedrooms}-bedroom home values are:\n{sorted_df.iloc[-3]}\n {sorted_df.iloc[-2]}\n {sorted_df.iloc[-1]}')

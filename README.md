# I SPY: Time Series Analysis for SPY ETF for Flatiron School Data Science Immersive (Capstone Project)
  
## Overview  
In this project, we examine and perform time series analysis on the [SPDR® S&P 500® ETF Trust](https://www.ssga.com/us/en/institutional/etfs/funds/spdr-sp-500-etf-trust-spy) to forecast future prices and inform trading decisions.  

![finance](images/finance.jpeg)  
  
## Motivation  
While institutional investors leverage enterprise-grade data feeds and models to plan and execute trades in the sub-second timeframe, amateur investors are limited primarily to free data and tools to make investment decisions. The goal is to take freely available public data and, leveraging machine learning modalities to predict the movement of financial instruments, help the little guy get on the right side of the trade, more often than not. The ticker 'SPY' (aka "Spiders") was chosen because it is well-established, highly liquid, and popular.

## Data  
Our primary SPY time series data comes the [yfinance API](https://github.com/ranaroussi/yfinance). We focused on the Close price marked at EOD of each trading data as well as the Volume traded as our first exogenous variable. Other exogenous variables -- the US Treasury 10 Year Note yield, the Federal Funds Target Rate, and the Federal Funds Effective rate -- come from the [Quandl API](https://www.quandl.com/tools/python). Our last exogenous variable is a binary indicator on whether a date is a SPY option expiry date; this was an engineered feature.

### Historical Data  
The SPY ETF, since it tracks the S&P 500 Index, show positive growth over time, and has tripled in price in the last 10 years. There is also some cyclical trends and seasonality present.  
 
### Target Variable  
We will forecast SPY closing price up through the end of the year.
![SPY Chart](images/SPY_Chart.png)  

## Methods  
Our methodology implements the CRISP-DM model for exploratory data analysis, cleaning, modeling, and evaluation.  
We leveraged ARIMA modeling from [pmdarima](http://alkaline-ml.com/pmdarima/) to analyze and forecast SPY closing prices. We leveraged the Date Featurizer, Fourier Featurizer, and the Logarithmic Featurizer. The quality of our modeling was inferred from [AIC value](https://en.wikipedia.org/wiki/Akaike_information_criterion) and evaluated with [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) and [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error). We also utilized statistical tools via [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html) to adjust confidence intervals on the model predictions.  
Other tools used include NumPy, Pandas. Visualizations were created with MatPlotLib and Seaborn.  

## Validation
We wrote and implemented a Step-Wise Cross-Validation tool to make iterative in-sample predictions to determine model fit and accuracy. We built a custom GridSearch function to test out different model paremeters, scoring each model via the aforementioned StepWiseCV. Best models were found for each variable.  
To further validate our best-scoring model, we implemented a historical simulation profit calculator to evaluate how the model would perform if tested against the market (i.e. against buying and holding).  
Using another custom GridSearch function, we tuned strategy parameters to determine the highest possible profit to ensure the model outputs compare favorably against historical data.
  
## Conclusion
After running ARIMA analysis on various combinations of timeframes, observation frequencies, and seasonalities, we chose daily frequencies with a yearly seasonality, for a 10 Year timeframe.  
Using our custom GridSearchCV function, we found ideal parameters for our model for the SPY Close as well as SPY Volume and each of our exogenous variables that minimize RMSE as a percent of the mean of the time series when compared against observed values.  We also added a quarterly seasonality and a 4-year seasonality.
Via our historical simulation GridSearch, we found ideal strategy parameters for our model to maximize profit from trading using predicted prices vs observed SPY Close.
Lastly, we forecasts 52 days ahead as well as until year-end for our exogenous variables and, using those as inputs, forecasted SPY Close 52 days and until year-end.

Here is a summary of our best models for the 10 Year time series:
Variable            | Parameters                                  |  AIC      | RMSE (% of mean) |  SMAPE
:-------------------|:--------------------------------------------|----------:|-----------------:|---------:  
Close               | Arima(4,1,6), Intercept, Trend='t', Fourier | 11939.92  |       4.59%      |  0.96%
Volume              | Arima(1,0,1), No Intercept, Fourier, Log    |  849.94   |      24.86%      | 24.82%
TSY 10Y             | Arima(0,1,0), Intercept, Trend='c'          | -8287.71  |       2.17%      |  3.22%
Fed Funds Target    | Arima(1,0,0), No Intercept                  | -10827.33 |       7.71%      |  0.66%
Fed Funds Effective | Arima(1,0,0), No Intercept                  | -10467.00 |       8.86%      |  4.37%

We found our model was accurate enough such that trading with it was able to beat the market by **25%** over the span of the time series.

Our results so far are encouraging: our model predicted a year-end price of **$418.34**, representing an S&P Index level of 4183, which compares well against the average projection on Wall Street of 4207.

## Further Actions  
- We will build a Buy/Hold/Sell indicator to recommend **trading decisions**, as well as implement a _profit calculator_ using historical data to evaluate _model performance_ in the real world compared against simpling buy and holding.  **DONE**
- Using aditional **exogenous variables** such as SPY Futures data, recession dates, options trade data, options expiry date, jobs data, and Treasury yield curve data, we will be able to implement a more robust model.  **Done for options expiry dates, Treasury yield curve data, Federal Funds Target and Effective rates.
- Additional _time series_ **modeling** can be built using TBATS, Facebook Prophet. We can also explore _neural networks_ such as LSTM, RNN, CNN to model and predict on SPY closing price. Initial results on _logistic regression_ has shown some promise, and we can delve into that as well.  
- We can build a **web-scraping tool** to gather and anayze _market sentiment_ on Twitter, Reddit, etc.
- Using available libraries as well as creating our own, we can plot and incorporate **technical indicators** such as MACD, RSI, Bollinger Bands, Support/Resistance, Stochastic, CCI, OBV, and Elliot Wave Analysis into our model.  
- Finally, we can create a **real-time model** that will automatically fetch and run models on each new observation. We can then build a _web app_ to allow us to access these predictions on the fly.  
     
## Index  
- **code/** — directory containing python code files
  - **functions.py** — helper functions
  - **Gridsearch_Calc_Profit.py** - class for running GridSearch on historical simulation parameters
  - **Model_Historical_Simuluation.py** - class for running historical simulations
  - **Pmdarima_Model.py** - class for running pmdarima models and tools
- **crisp_dm_process/** — directory for initial EDA and model notebook files
  - **av_eod_eda_modeling.ipynb** — notebook file for data exploration and modeling on AlphaVantage data using SARIMAX
  - **yf_eda_modeling_colab.ipynb** — notebook file for data exploration and modeling using Google Colab
  - **yf_eda_modeing.ipynb** — notebook file for data exploration and modeling on yfinance SPY data using pmdarima
  - **yf_fut_eda_modeling.ipynb** — notebook file for exploration and modeling using yfinance SPY Futures data
- **data/** — directory of project data sets
- **I SPY - ETF Price Predicitons.pdf** - presentation slides PDF
- **images/** — directory containing all image files:
  - **AutoArima/** - AutoArima in-sample test predictions
  - **GridSearch/** - AutoArima in-sample test predictions
  - **model_profit/** - visualizations of historical simulation model profit and GridSearch heatmaps
  - various EDA visualizations
  - additional images for README
- **model_CV_scores/** - directoring containing CSV files of Step-Wise Cross-Validation scores during time-series modeling
- **model_profit_GS/** - directoring containing CSV and pickled files of historical simulation GridSearch
- **models/** - directoring containing pickled model files
- **README.md** - this file
- **sound/** - directory containing audio file to provide alert after code execution
- **Wei Alexander Xin - Data Science Final Presentation.mp4** - recording of presentation
- **spy_final_notebook.ipynb** — final project notebook
  
## Bibliography  
1. Dataset:  
  
       Yahoo! Finance Stock Data  
            Yahoo! Group  
   Last Access:    Wednesday May 12, 2021.  
   Web Source:  https://finance.yahoo.com/quote/SPY            
2. Exogenous Data:
    1. US Treasury Bond Yield Data

           US Treasury Bond Yield Data
           Treasury Direct
           U.S. Department of the Treasury, Bureau of the Fiscal Service. 
    Last Access:    Wednesday May 12, 2021.  
    Web Source:  https://www.treasurydirect.gov/indiv/products/prod_tnotes_glance.htm  
    
    2. Federal Funds Target Rate Data

           Federal Reserve Economica Data
           Federal Reserve Bank of St. Louis
    Last Access:    Wednesday May 12, 2021.  
    Web Source:  https://fred.stlouisfed.org/series/DFEDTARU
    
    3. Federal Funds Effective Rate Data

           Federal Reserve Economica Data
           Federal Reserve Bank of St. Louis
    Last Access:    Wednesday May 12, 2021.  
    Web Source:  https://fred.stlouisfed.org/series/FEDFUNDS


<div align="center";>Author 
  <div align="center";>Wei Alexander Xin  
    
[Alexander's GitHub](https://github.com/eggrollofchaos)  
[Alexander's LinkedIn](https://www.linkedin.com/in/waximus)

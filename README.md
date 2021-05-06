# I SPY: Time Series Analysis for SPY ETF for Flatiron School Data Science Immersive (Capstone Project)
  
## Overview  
In this project, we examine and perform time series analysis on the [SPDR® S&P 500® ETF Trust](https://www.ssga.com/us/en/institutional/etfs/funds/spdr-sp-500-etf-trust-spy) to forecast future prices and inform trading decisions.  

![finance](data/finance.jpeg)  
  
## Motivation  
While institutional investors leverage enterprise-grade data feeds and models to plan execute trades in the sub-second time-frame, amateur investors are limited primarily to free data and tools and make investment decisions. The goal is to take freely available public data and, leveraging machine learning modalities to predict the movement of financial instruments to help the little guy get on the right side of the trade, more often than not. The tick 'SPY' (aka "Spiders") was chosen because it is well established, highly liquid, and popular.

## Data  
Our primary SPY time series data comes the [yfinance API](https://github.com/ranaroussi/yfinance). As our MVP, we focused on the Close price marked at EOD of each trading data as well as the Volume traded as an exogenous variable.

### Historical Data  
The SPY ETF, since it tracks the S&P 500 Index, show positive growth over time, and has tripled in price in the last 10 years. There is also some cyclical and seasonality present.  

### Target Variable  
We will forecast SPY closing price up to 15 trading days into the future, ending on a Friday.

## Methods  
Our methodology implements the CRISP-DM model for exploratory data analysis, cleaning, modeling, and evaluation.  
We leveraged ARIMA modeling from [pmdarima](http://alkaline-ml.com/pmdarima/) to analyze and forecast SPY closing prices. The quality of our modeling was inferred from [AIC value](https://en.wikipedia.org/wiki/Akaike_information_criterion) and evaluated with [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) and [SMAPE](https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error). We also performed statistical analysis via [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html) to further make inferences on the data.  
Other tools used include NumPy, Pandas. Visualizations were created with MatPlotLib and Seaborn.  
  
## Conclusion
After running ARIMA analysis on various combinations of timeframes, observation frequencies, and seasonalities, we chose daily frequencies with a yearly seasonality, for 3 Year, 5 Year, and 10 Year timeframes.  Using a custom GridSearchCV function, we found ideal parameters for model, separately, the Volume and Close time series. We fit and predicted on Volume on 12 days ahead, then used the results as exogenous variables to fit and predict on Close.  

Here is a summary of our best models, for the 10 Year time series:
Variable  |    AIC   |     RMSE    |  SMAPE
:---------|:---------|-------------|--------:  
Volume    | 269.89   |     4.47    |  0.931% 
Close     | 12244.82 | 23971769.85 | 22.826% 

Our results so far are encouraging; out of three trading days, our predictions were right (within 50 cents) of the actual price.

## Further Actions  
- We will build a Buy/Hold/Sell indicator to recommend **trading decisions**, as well as implement a _profit calculator_ using historical data to evaluate _model performance_ in the real world compared against simpling buy and holding.  
- Using aditional **exogenous variables** such as SPY Futures data, recession dates, options trade data, options expiry data, jobs data, and Treasury yield curve data, we will be able to implement a more robust model.  
- Additional _time series_ **modeling** can be built using TBATS, Facebook Prophet. We can also explore _neural networks_ such as LSTM, RNN, CNN to model and predict on SPY closing price. Initial results on _logistic regression_ has shown some promise, and we can delve into that as well.  
- We can build a **web-scraping tool** to gather and anayze _market sentiment_ on Twitter, Reddit, etc.
- Using available libraries as well as creating our own, we can plot and incorporate **technical indicators** such as MACD, RSI, Bollinger Bands, Support/Resistance, Stochastic, CCI, OBV, and Elliot Wave Analysis into our model.  
- Finally, we can create a **real-time model** that will automatically fetch and run models on each new observation. We can then build a _web app_ to allow us to access these predictions on the fly.  
     
## Index  
- **code/** — directory containing python code files
  - **functions.py** — helper functions
  - **Pmdarima_Model.py** - custom class for running pmdarima models and tools
- **crisp_dm_process/** — directory for initial EDA and model notebook files  
  - **spy_av_eda.ipynb** — notebook file for data exploration and modeling on AlphaVantage data using SARIMAX
  - **spy_eda_colab.ipynb** — notebook file for data exploration and modeling using Google Colab
  - **spy_yf_eda.ipynb** — notebook file for data exploration and modeling on yfinance SPY data using pmdarima - primary notebook for MVP
  - **spy_yf_fut_eda.ipynb** — notebook file for exploration and modeling using yfinance SPY Futures data
- **data/** — directory of project data sets
- **images/** — directory containing all image files:  
  - exports of data visualizations  
  - additional images for README
- **spy_final_notebook.ipynb** — final project notebook  
- **I SPY MVP.pdf** - presentation slides PDF
- **README.md** - this file
  
## Bibliography  
1. Dataset Origin:  
  
       Yahoo! Finance Stock Data  
            Yahoo! Group  
2. Last Access:    Wednesday May 5, 2021. 
3. Web Source:  https://finance.yahoo.com/quote/SPY            
  
<div align="center";>Author 
  <div align="center";>Wei Alexander Xin  
    
[Alexander's GitHub](https://github.com/eggrollofchaos)  
[Alexander's LinkedIn](https://www.linkedin.com/in/waximus)

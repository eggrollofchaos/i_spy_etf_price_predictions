from tqdm.notebook import tqdm
import pandas as pd
import matplotlib
import numpy as np

try:
    from code.functions import *
except Exception as e:
    from functions import *

# print(f'Model_Historical_Simuluation.py loaded from {TOP}/data.')

class Model_Historical_Simuluation():
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

    def __init__(self, model, ohlc_df, exog_hist_df, shares=1000, z=1,
        limit_offset_pc=0.5, start_date=None, verbose=0):
        self.model = model
        self.ohlc_df = ohlc_df
        self.exog_hist_df = exog_hist_df
        self.shares = shares
        self.z = z
        self.limit_offset_pc = limit_offset_pc
        self.start_date = start_date
        if self.start_date == None:
            self.start_date = self.ohlc_df.index[0]
        self.cost_basis = self.shares*self.ohlc_df.close[self.start_date]
        self.verbose = verbose
        self.cash = 0
        self.portfolio_value = self.cash + self.cost_basis
        self.trades = 0
        self.predictions = []
        self.conf_ints = []

    def __buy_shares(self, date, stop_price, strat='all', verbose=0):
        limit_price = stop_price - self.limit_offset_pc*0.01*stop_price
        if self.ohlc_df.low[date] < limit_price:       # limit order will be filled
            if strat=='all':
                shares_to_buy = self.cash/limit_price
            cost = shares_to_buy*limit_price
            self.trades += 1
            if verbose>1:
                print('***********TRADE***********')
                print(f'   Buying max number of shares - {shares_to_buy:,.4f}.')
                print(f'   Cost = ${cost:,.2f}')
            return shares_to_buy, cost
        else:
            return 0, 0

    def __sell_shares(self, date, stop_price, strat='all', verbose=0):
        limit_price = stop_price + self.limit_offset_pc*0.01*stop_price
        if self.ohlc_df.high[date] > limit_price:      # limit order will be filled
            if strat=='all':
                shares_to_sell = self.shares
            returns = shares_to_sell*limit_price
            self.trades += 1
            if verbose>1:
                print('***********TRADE***********')
                print(f'   Selling all {shares_to_sell:,.4f} shares.')
                print(f'   Revenue = ${returns:,.2f}')
            return shares_to_sell, returns
        else:
            return 0, 0

    def __update_portfolio_value(self, date, shares_delta, cash_delta):
        self.shares += shares_delta
        self.cash += cash_delta
        self.portfolio_value = self.shares*self.ohlc_df.low[date] + self.cash

    def __calc_profit(self, index, current_date, verbose=0):
        '''
        Iteratively calculate profit using in-sample predictions one day
        at a time, with dynamic = False.
        '''
        print(f'Analyzing Day {index+1} - {current_date.date()}:') if verbose>1 else None
        # a little animation to pass the time
        if verbose:
            if index&1:
                print('>_', end='\r')
            else:
                print('> ', end='\r')
        traded = False
        if index == 0:
            # initialize as equal to the starting price to match dimensions
            self.predictions.append(self.ohlc_df.close[current_date])
            self.conf_ints.append([self.ohlc_df.close[current_date], self.ohlc_df.close[current_date]])
            print('Nothing to do! We only just bought all the dang shares.') if verbose>1 else None
        else:
            # pred, conf_int = self.model.predict_in_sample(X=self.exog_hist_df[0:index+1],
            pred, conf_int = self.model.predict_in_sample(X=self.exog_hist_df,
                start=index, end=index, return_conf_int=True, alpha=(2-stats.norm.cdf(self.z)*2))
            self.predictions.append(pred[0])
            self.conf_ints.append(conf_int.tolist()[0])
            # print(len(y_hat))
            fcast = pred[0]
            fcast_low = conf_int[0][0]
            fcast_high = conf_int[0][1]
            if verbose>1:
                print('High   |    Low |  Close | Predicted Close | Confidence Interval')
                print('%.2f | %.2f | %.2f |          %.2f | %.2f - %.2f' % (self.ohlc_df.high[current_date], self.ohlc_df.high[current_date], self.ohlc_df.close[current_date], fcast, fcast_low, fcast_high))
            high_diff = self.ohlc_df.high[current_date] - fcast_high
            low_diff = self.ohlc_df.low[current_date] - fcast_low
            if self.shares>0 and high_diff>0:
                print('SPY high of day greater than top of confidence interval by %.2f' % high_diff) if verbose>1 else None
                shares_to_sell, cash_received = self.__sell_shares(
                    current_date, stop_price=fcast_high,
                    strat='all', verbose=verbose)
                self.__update_portfolio_value(current_date, -shares_to_sell, cash_received)
                traded=True

            elif self.cash>0 and low_diff<0:
                print('SPY low of day less than bottom of confidence interval by %.2f' % low_diff) if verbose>1 else None
                shares_to_buy, cash_spent = self.__buy_shares(
                    current_date, stop_price=fcast_low,
                    strat='all', verbose=verbose)
                self.__update_portfolio_value(current_date, shares_to_buy, -cash_spent)
                traded=True

            else:
                print('Price movement for today is within confidence interval.\n-----------Holding-----------') if verbose>1 else None

        self.mod_profit_df.loc[current_date,'shares'] = self.shares
        self.mod_profit_df.loc[current_date,'cash'] = self.cash
        self.mod_profit_df.loc[current_date,'trade'] = traded
        self.mod_profit_df.loc[current_date,'portfolio_value'] = self.portfolio_value
        self.mod_profit_df.loc[current_date,'eod_profit'] = self.portfolio_value - self.cost_basis
        self.mod_profit_df.loc[current_date,'eod_profit_pc'] = 100*(self.portfolio_value - self.cost_basis)/self.cost_basis

        if verbose>1:
            print('EOD portfolio snapshot:')
            print('Shares | Cash | Portfolio Value')
            print(f'{self.shares:,.4f} | ${self.cash:,.2f} | ${self.portfolio_value:,.2f} ')
            print('__________________________________________________________________')

    def main(self):
        print('Running `I SPY` model historical simulation...\n') if self.verbose else None

        # check if already fit
        try:
            self.model.named_steps['arima'].arroots()
        except Exception as e:
            print('Fitting model to all exogenous variables... ', end='') if self.verbose>1 else None
            self.model.fit(self.ohlc_df.close, self.exog_hist_df)
            print('Done.') if self.verbose>1 else None

        N = self.ohlc_df.shape[0]
        self.mod_profit_df = pd.DataFrame(index=self.ohlc_df.index)
        self.mod_profit_df['shares'] = np.zeros(N)
        self.mod_profit_df['cash'] = np.zeros(N)
        self.mod_profit_df['trade'] = np.zeros(N)
        self.mod_profit_df['portfolio_value'] = np.zeros(N)
        self.mod_profit_df['eod_profit'] = np.zeros(N)
        self.mod_profit_df['eod_profit_pc'] = np.zeros(N)

        print('Performing step-wise calculation of profit using time iterative model forecasts...\n') if self.verbose>1 else None
        if self.verbose:
            [self.__calc_profit(index, current_date, self.verbose) for index, current_date in enumerate(tqdm(self.ohlc_df.index, desc='Profit Calculation Loop'))]
        else:
            [self.__calc_profit(index, current_date, self.verbose) for index, current_date in enumerate(self.ohlc_df.index)]

        self.total_profit = self.portfolio_value-self.cost_basis
        self.total_profit_pc = 100*(self.portfolio_value-self.cost_basis)/self.cost_basis
        if self.verbose>1:
            print('Done.')
            print(f'Made {self.trades} trades out of {N} trading days.')
            print(f'Starting portfolio value was ${self.cost_basis:,.2f}.')
            print(f'Final portfolio value is ${self.portfolio_value:,.2f}.')
        if self.verbose:
            print(f'Total profit is ${self.total_profit:,.2f}, which is a {self.total_profit_pc:,.2f}% profit.')

        return self.predictions, self.conf_ints, self.total_profit, self.total_profit_pc, self.mod_profit_df

    if __name__ == "__main__":
        main()

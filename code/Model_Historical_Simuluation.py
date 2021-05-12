from tqdm.notebook import tqdm
import pandas as pd
import matplotlib
import numpy as np

from functions import *

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
        limit_offset_pc=0.5, start_date=None, freq=CBD, verbose=1):
        self.self.model = model
        self.ohlc_df = ohlc_df
        self.exog_hist_df = ohlc_df
        self.shares = shares
        self.z = z
        self.limit_offset_pc = limit_offset_pc
        self.start_date = start_date
        self.freq = freq
        self.verbose = verbose
        self.cash = 0
        self.portfolio_value = cash + cost_basis
        self.trades = 0
        self.predictions = []
        self.conf_ints = []

    def __buy_shares(self, shares, cash, date, trades, ohlc_df, stop_price,
            limit_offset_pc, strat='all', verbose=0):
        limit_price = stop_price - limit_offset_pc*0.01*stop_price
        if ohlc_df.low[date] < limit_price:       # limit order will be filled
            if strat=='all':
                shares_to_buy = cash/limit_price
            cost = shares_to_buy*limit_price
            trades += 1
            if verbose>1:
                print('***********TRADE***********')
                print(f'   Buying max number of shares - {shares_to_sell:,.4f}.')
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


    def __calc_profit(index, current_date):
        '''
        Iteratively calculate profit using in-sample predictions one day
        at a time, with dynamic = False.
        '''
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
            pred, conf_int = self.model.predict_in_sample(X=exog_hist_df[0:index+1],
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
                shares, cash, port

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

    def main():
        print('Running `I SPY` model historical simulation...\n') if verbose else None
        if start_date == None:
            start_date = ohlc_df.index[0]
        cost_basis = shares*ohlc_df.close[start_date]

        # check if already fit
        try:
            self.model.named_steps['arima'].arroots()
        except Exception as e:
            if verbose>1:
                print('Fitting model to all exogenous variables... ', end='')
                print('Done.')
            self.model.fit(ohlc_df.close, exog_hist_df)

        N = ohlc_df.shape[0]
        mod_profit_df = pd.DataFrame(index=ohlc_df.index)
        mod_profit_df['shares'] = np.zeros(N)
        mod_profit_df['cash'] = np.zeros(N)
        mod_profit_df['trade'] = np.zeros(N)
        mod_profit_df['portfolio_value'] = np.zeros(N)
        mod_profit_df['eod_profit'] = np.zeros(N)
        mod_profit_df['eod_profit_pc'] = np.zeros(N)

        print('Performing step-wise calculation of profit using time iterative model forecasts...\n') if verbose>1 else None
        if verbose:
            [__calc_profit for index, current_date in enumerate(tqdm(ohlc_df.index, desc='Profit Calculation Loop'))]
        else:
            [__calc_profit for index, current_date in enumerate(ohlc_df.index, desc='Profit Calculation Loop')]

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

        # return

        # for index, current_date in enumerate(tqdm(ohlc_df.index, desc='Profit Calculation Loop')):

            # print(f'Analyzing Day {index+1} - {current_date.date()}:') if verbose>1 else None
            # # a little animation to pass the time
            # if not verbose:
            #     if index&1:
            #         print('>_', end='\r')
            #     else:
            #         print('> ', end='\r')
            # traded = False
            # if index == 0:
            #     # initialize as equal to the starting price to match dimensions
            #     predictions.append(ohlc_df.close[current_date])
            #     conf_ints.append([ohlc_df.close[current_date], ohlc_df.close[current_date]])
            #     print('Nothing to do! We only just bought all the dang shares.') if verbose>1 else None
            # else:
            #     pred, conf_int = model.predict_in_sample(X=exog_hist_df[0:index+1],
            #         start=index, end=index, return_conf_int=True, alpha=(2-stats.norm.cdf(z)*2))
            #     predictions.append(pred[0])
            #     conf_ints.append(conf_int.tolist()[0])
            #     # print(len(y_hat))
            #     fcast = pred[0]
            #     fcast_low = conf_int[0][0]
            #     fcast_high = conf_int[0][1]
            #     if verbose>1:
            #         print('High  |   Low  |  Close | Predicted Close | Confidence Interval')
            #         print('%.2f  | %.2f | %.2f   | %.2f | %.2f - %.2f' % (ohlc_df.high[current_date], ohlc_df.high[current_date], ohlc_df.close[current_date], fcast, fcast_low, fcast_high))
            #         # print(f'{ohlc_df.high[current_date]:.2f}   {ohlc_df.low[current_date]:.2f}  {ohlc_df.close[current_date]:.2f}   | {fcast:.2f} | {fcast_low:.2f} - {fcast_high:.2f}')
            #     high_diff = ohlc_df.high[current_date] - fcast_high
            #     low_diff = ohlc_df.low[current_date] - fcast_low
            #     if shares>0 and high_diff>0:
            #         print('SPY high of day greater than top of confidence interval by %.2f' % high_diff) if verbose>1 else None
            #         shares_to_sell, cash_received, trades = __sell_shares(shares, cash,
            #             current_date, trades, ohlc_df, stop_price=fcast_high,
            #             limit_offset_pc=limit_offset_pc, strat='all', verbose=verbose)
            #         shares, cash, portfolio_value = __update_portfolio_value(current_date,
            #             shares, cash, -shares_to_sell, cash_received)
            #         traded=True
            #
            #     elif cash>0 and low_diff<0:
            #         print('SPY low of day less than bottom of confidence interval by %.2f' % low_diff) if verbose>1 else None
            #         shares_to_buy, cash_spent, trades = __buy_shares(shares, cash,
            #             current_date, trades, ohlc_df, stop_price=fcast_low,
            #             limit_offset_pc=limit_offset_pc, strat='all', verbose=verbose)
            #         shares, cash, portfolio_value = __update_portfolio_value(current_date,
            #             shares, cash, shares_to_buy, -cash_spent)
            #         traded=True
            #
            #     else:
            #         print('Price movement for today is within confidence interval.\n-----------Holding-----------') if verbose>1 else None
            #         shares, cash, portfolio_value = __update_portfolio_value(current_date, shares, cash, 0, 0)

            # mod_profit_df.loc[current_date,'shares'] = shares
            # mod_profit_df.loc[current_date,'cash'] = cash
            # mod_profit_df.loc[current_date,'trade'] = traded
            # mod_profit_df.loc[current_date,'portfolio_value'] = portfolio_value
            # mod_profit_df.loc[current_date,'eod_profit'] = portfolio_value - cost_basis
            # mod_profit_df.loc[current_date,'eod_profit_pc'] = 100*(portfolio_value - cost_basis)/cost_basis
            #
            # if verbose>1:
            #     print('EOD portfolio snapshot:')
            #     print('Shares | Cash | Portfolio Value')
            #     print(f'{shares:,.4f} | ${cash:,.2f} | ${portfolio_value:,.2f} ')
            #     print('__________________________________________________________________')
            # time.sleep(5)


    if __name__ == "__main__":
        main()

from tqdm.notebook import tqdm
import pandas as pd
import matplotlib
import numpy as np

try:
    from code.functions import *
except Exception as e:
    from functions import *

 # -*- coding: utf-8 -*-
try:
    from code.Model_Historical_Simuluation import Model_Historical_Simuluation
except Exception as e:
    from Model_Historical_Simuluation import Model_Historical_Simuluation


# print(f'Gridsearch_Calc_Profit.py loaded from {TOP}/data.')

class Gridsearch_Calc_Profit:
    '''
    Run GridSearch on model profit historical simulation by tuning parameters
    standard deviation `z` and limit price offset percent `lim`.
    '''
    def __init__(self, ts, model, ohlc_df, exog_hist_df, shares=1000,
        steps=10, z_min=0.5, z_max=2.5, lim_min=0, lim_max=1,
        start_date=None, visualize=True, verbose=2):
        self.ts = ts
        self.model = model
        self.ohlc_df = ohlc_df
        self.exog_hist_df = exog_hist_df
        self.shares = shares
        self.steps = steps
        self.z_min = z_min
        self.z_max = z_max
        self.lim_min = lim_min
        self.lim_max = lim_max
        self.start_date = start_date
        self.verbose = verbose

        columns = ['z', 'Limit_Offset_pc', 'Final_Market_Value', 'Total_Profit', 'Total_Profit_pc']
        self.GS_mod_profit_df = pd.DataFrame(columns=columns)
        self.GS_mod_profit_df.index.name = 'Strategy'
        self.z_list = np.linspace(z_min, z_max, steps).round(3)
        self.lim_list = np.linspace(lim_min, lim_max, steps).round(3)
        self.num_sims = len(self.z_list)*len(self.lim_list)

    def __GS_calc_z_loop(self, z, verbose=0):
        # mod_profit_dict = {}
        if verbose:
            [self.__GS_calc_lim_loop(z, lim, verbose) for lim in tqdm(self.lim_list, desc='GridSearch `lim` Loop')]
        else:
            [self.__GS_calc_lim_loop(z, lim, verbose) for lim in self.lim_list]
        # print(mod_profit_dict)
        # return mod_profit_dict

    def __GS_calc_lim_loop(self, z, lim, verbose=0):
        # for z in tqdm(np.arange(z_min, z_max, 0.2), desc='GridSearch Loop: z'):
        #     for lim in tqdm(np.arange(lim_min, lim_max, 0.1), desc='GridSearch Loop: lim'):
        # if abs(z/2-lim)<0.4 or abs(z/2-lim)>1:
        #     return
        # if (z/2+lim)<0.5 or (z/2+lim)>1.5:
        if (3*z/5+lim)<0.4 or (3*z/5+lim)>2.75:
            return

        mod_profit_dict = {}
        if verbose:
            print(f'Parameter standard deviations (z) = {z}, Limit price offset percent = {lim}')

        sim = Model_Historical_Simuluation(self.model, self.ohlc_df, self.exog_hist_df, self.shares,
            z, lim, self.start_date, verbose)
        y_hat, conf_ints, mod_profit, mod_profit_pc, mod_profit_df = \
            sim.main()
        # mod_profit_dict['Time Series'] = ts
        # mod_profit_dict['Model_Pipeline'] = model
        # mod_profit_dict['Exogenous_Variables'] = exog_hist_df.columns[1:].tolist()
        # mod_profit_dict['Initial_Shares'] = shares
        # mod_profit_dict['Cost_Basis'] = cost_basis
        mod_profit_dict['z'] = z
        mod_profit_dict['Limit_Offset_pc'] = lim
        mod_profit_dict['Final_Market_Value'] = mod_profit_df.eod_profit[-1]
        mod_profit_dict['Total_Profit'] = mod_profit
        mod_profit_dict['Total_Profit_pc'] = mod_profit_pc
        print('__________________________________________________________________') if verbose else None

        self.GS_mod_profit_df = self.GS_mod_profit_df.append(mod_profit_dict, ignore_index=True)
        # return
        # GS_mod_profit_df.append(mod_profit_dict, ignore_index=True)
        # return mod_profit_dict

    def get_max_profit(self):
        self.GS_best_z_lim = self.get_best_z_lim()
        self.GS_max_profit = self.GS_mod_profit_df.sort_values(by='Total_Profit_pc', ascending=False).head(1)['Total_Profit'].values[0]
        self.GS_max_profit_pc = self.GS_mod_profit_df.sort_values(by='Total_Profit_pc', ascending=False).head(1)['Total_Profit_pc'].values[0]
        print(f'Best profit achieved with `z` = {self.GS_best_z_lim[0]} and `lim` = {self.GS_best_z_lim[1]}:')
        print(f'${self.GS_max_profit:,.2f} | {self.GS_max_profit_pc:,.2f}%')
        return self.GS_max_profit, self.GS_best_z_lim

    def get_best_z_lim(self):
        best_z = self.GS_mod_profit_df.sort_values(by='Total_Profit_pc', ascending=False).head(1)['z'].values[0]
        best_lim = self.GS_mod_profit_df.sort_values(by='Total_Profit_pc', ascending=False).head(1)['Limit_Offset_pc'].values[0]
        self.GS_best_z_lim = (best_z, best_lim)
        return self.GS_best_z_lim

    def plot_profit_heatmap(self):
        tick_params = dict(size=4, width=1.5, labelsize=16)
        self.GS_mod_profit_pivot = self.GS_mod_profit_df.pivot(index='Limit_Offset_pc', columns='z', values='Total_Profit_pc')

        fig, ax = plt.subplots(figsize=(15,12))
        sns.heatmap(self.GS_mod_profit_pivot, ax=ax, cbar_kws={"label": "Profit %"}, cmap="YlGn")
        ax.invert_yaxis()
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(**tick_params)
        cbar.ax.yaxis.label.set_size(20)
        ax.set_xlabel('Confidence Interval Standard Deviation (z)', size=20)
        ax.set_ylabel('Limit Price Offset %', size=20)
        # ax.set_xticks(np.linspace(0.5, self.steps-0.5, 8))
        # ax.set_yticks(np.linspace(0.5, self.steps-0.5, 8))
        # ax.set_xticklabels(np.linspace(self.z_min, self.z_max, self.steps).round(3), rotation=0)
        # ax.set_yticklabels(np.linspace(self.lim_min, self.lim_max, self.steps).round(3)[::-1], va="center")
        # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
        # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
        ax.tick_params(axis='x', **tick_params)
        ax.tick_params(axis='y', **tick_params)

        fig.suptitle('Comparison of `z` vs `lim` Parameters on Profit %', size=26)
        fig.subplots_adjust(top=0.93)
        plt.savefig(f'{TOP}/images/model_profit/SPY_10Y_Model_Profit_Heatmap.png')

    def main(self):
        print(f'Running GridSearchCV on {self.ts} model trading strategy...')

        print(f'Running {self.num_sims} simulations using `z` in ({self.z_min}, {self.z_max}) and `lim` in ({self.lim_min}, {self.lim_max}).')
        # mod_profit_dict = []
        if self.verbose:
            [self.__GS_calc_z_loop(z, self.verbose) for z in tqdm(self.z_list, desc='GridSearch `z` Loop')]
        else:
            [self.__GS_calc_z_loop(z, self.verbose) for z in self.z_list]

        N = self.GS_mod_profit_df.shape[0]
        self.GS_mod_profit_df.insert(0, 'Cost_Basis', [self.shares*self.ohlc_df.close[0]]*N)
        self.GS_mod_profit_df.insert(0, 'Initial_Shares', [self.shares]*N)
        self.GS_mod_profit_df.insert(0, 'Exogenous_Variables', [self.exog_hist_df.columns[1:].tolist()]*N)
        self.GS_mod_profit_df.insert(0, 'Model_Pipeline', [self.model]*N)
        self.GS_mod_profit_df.insert(0, 'Time Series', [self.ts]*N)

        ts_str = self.ts.replace(' ', '_').upper()
        self.GS_mod_profit_df.to_csv(f'{TOP}/model_profit_GS/{ts_str}_Profit_CV.csv')
        pickle_data(self.GS_mod_profit_df, f'{TOP}/model_profit_GS/{ts_str}_Profit_CV.pkl')

        if visualize:
            self.plot_profit_heatmap()

        return self.GS_mod_profit_df

    if __name__ == "__main__":
        main()

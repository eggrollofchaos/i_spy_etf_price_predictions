import nium

class Gridsearch_Calc_Profit:
    '''
    Run GridSearchCV on model profit historical simulation by tuning parameters
    standard deviation `z` and limit price offset percent `lim`.
    '''
    def __init__(self, ts, model, ohlc_df, exog_hist_df, shares=1000,
        steps=10, z_min=0.5, z_max=2.5, lim_min=0, lim_max=1,
        start_date=None, verbose=2):
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
        mod_profit_dict = {}
        if verbose:
            print(f'Parameter standard deviations (z) = {z}, Limit price offset percent = {lim}')
        y_hat, conf_ints, mod_profit, mod_profit_pc, mod_profit_df = \
            Model_Historical_Simuluation(self.model, self.ohlc_df, self.exog_hist_df, shares=self.shares,
                z=z, limit_offset_pc=lim, verbose=verbose)
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
        return mod_profit_dict

    # def run_gridsearch_calc_profit():
    def main(self):
        print(f'Running GridSearchCV on {self.ts} model trading strategy...')

        # columns = ['Time Series', 'Model_Pipeline', 'Exogenous_Variables', 'z',
        #     'Limit_Offset_pc', 'Initial_Shares', 'Cost_Basis', 'Final_Market_Value', 'Total_Profit', 'Total_Profit_pc']
        # columns = ['z', 'Limit_Offset_pc', 'Final_Market_Value', 'Total_Profit', 'Total_Profit_pc']
        # GS_mod_profit_df = pd.DataFrame(columns=columns)
        # GS_mod_profit_df.index.name = 'Strategy'
        # z_list = np.linspace(z_min, z_max, steps).round(3)
        # lim_list = np.linspace(lim_min, lim_max, steps).round(3)
        # num_sims = len(z_list)*len(lim_list)
        print(f'Running {self.num_sims} simulations using `z` in ({self.z_min}, {self.z_max}) and `lim` in ({self.lim_min}, {self.lim_max}).')
        # mod_profit_dict = []
        if self.verbose:
            [self.__GS_calc_z_loop(z, self.verbose) for z in tqdm(self.z_list, desc='GridSearch `z` Loop')]
        else:
            [self.__GS_calc_z_loop(z, self.verbose) for z in self.z_list]
        # print(mod_profit_dict)
        # GS_mod_profit_df = pd.DataFrame.from_records(mod_profit_dict)
        N = self.GS_mod_profit_df.shape[0]
        self.GS_mod_profit_df.insert(0, 'Cost_Basis', [self.shares*self.ohlc_df.close[0]]*N)
        self.GS_mod_profit_df.insert(0, 'Initial_Shares', [self.shares]*N)
        self.GS_mod_profit_df.insert(0, 'Exogenous_Variables', [self.exog_hist_df.columns[1:].tolist()]*N)
        self.GS_mod_profit_df.insert(0, 'Model_Pipeline', [self.model]*N)
        self.GS_mod_profit_df.insert(0, 'Time Series', [self.ts]*N)
            # for lim in tqdm(np.arange(lim_min, lim_max, 0.1), desc='GridSearch Loop: lim'):
            #     print(f'Parameter `z` = {z}')
            #     print(f'Limit price offset percent = {lim}')
            #     y_hat, conf_ints, mod_profit, mod_profit_pc, mod_profit_df = \
            #         calc_model_profit(model, ohlc_df, exog_hist_df, shares=shares,
            #             z=z, limit_offset_pc=lim, verbose=verbose)
            #     mod_profit_dict['Time Series'] = ts
            #     mod_profit_dict['Model_Pipeline'] = model
            #     mod_profit_dict['Exogenous_Variables'] = exog_hist_df.columns[1:].tolist()
            #     mod_profit_dict['z'] = z
            #     mod_profit_dict['Limit_Offset_pc'] = lim
            #     mod_profit_dict['Initial_Shares'] = shares
            #     mod_profit_dict['Cost_Basis'] = cost_basis
            #     mod_profit_dict['Final_Market_Value'] = mod_profit_df.eod_profit[-1]
            #     mod_profit_dict['Total_Profit'] = mod_profit
            #     mod_profit_dict['Total_Profit_pc'] = mod_profit_pc
            #     GS_mod_profit_df = GS_mod_profit_df.append(mod_profit_dict, ignore_index=True)
            #     print('__________________________________________________________________')
        ts_str = self.ts.replace(' ', '_').title()
        self.GS_mod_profit_df.to_csv(f'{TOP}/model_profit_CV/{ts_str}_Profit_CV.csv')
        pickle_data(self.GS_mod_profit_df, f'{TOP}/model_profit_CV/{ts_str}_Profit_CV.pkl')

        return self.GS_mod_profit_df

    if __name__ == "__main__":
        main()

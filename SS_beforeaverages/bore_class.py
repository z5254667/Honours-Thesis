#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:43:55 2022

@author: johnsalvaris
"""

import numpy as np
import pandas as pd
import import_functions

class bore():
    def __init__(self, bore_id):
        self.id = bore_id
    
    def add_location(self):
        self.bore_latitude = self.bore_df['Latitude'][0]
        self.bore_longitude = self.bore_df['Longitude'][0]
        
        self.silo_latitude = self.silo_df['latitude'].to_numpy()[0]
        self.silo_longitude = self.silo_df['longitude'].to_numpy()[0]
        
        """
        Categorise as coastal or inland
        """
        
    def add_dfs(self):
        self.bore_df = import_functions.get_bore_df(self.id)
        self.gwl_df = import_functions.get_gwl_df(self.id)
        self.silo_df = import_functions.get_silo_df(self.id)
        

    def add_gwl_data(self, leads):
        self.gwl_dates = self.gwl_df['Date'].to_numpy()
        self.swl = self.gwl_df['Result (m)'].to_numpy()
        
        # Delta swl as an output
        self.output_delta_swl = []
        for i in range(leads, len(self.swl)):
            self.output_delta_swl.append(self.swl[i]-self.swl[i-leads])
        self.output_delta_swl = np.array(self.output_delta_swl) # length is = length of swl - number of leads
        
        # Daily delta swl as an input
        self.daily_delta_swl = []
        for i in range(1, len(self.swl)):
            self.daily_delta_swl.append(self.swl[i]-self.swl[i-1])
        self.daily_delta_swl = np.array(self.daily_delta_swl)

    def add_silo_data(self):
        self.silo_dates = self.silo_df['YYYY-MM-DD'].to_numpy()
        
        self.daily_rain = self.silo_df['daily_rain'].to_numpy()
        self.max_temp = self.silo_df['max_temp'].to_numpy()
        self.min_temp = self.silo_df['min_temp'].to_numpy()
        self.vp = self.silo_df['vp'].to_numpy()
        self.vp_deficit = self.silo_df['vp_deficit'].to_numpy()
        self.evap_pan = self.silo_df['evap_pan'].to_numpy()
        self.evap_syn = self.silo_df['evap_syn'].to_numpy()
        self.evap_comb = self.silo_df['evap_comb'].to_numpy()
        self.evap_morton_lake = self.silo_df['evap_morton_lake'].to_numpy()
        self.radiation = self.silo_df['radiation'].to_numpy()
        self.rh_tmax = self.silo_df['rh_tmax'].to_numpy()
        self.rh_tmin = self.silo_df['rh_tmin'].to_numpy()
        self.et_short_crop = self.silo_df['et_short_crop'].to_numpy()
        self.et_tall_crop = self.silo_df['et_tall_crop'].to_numpy()
        self.et_morton_actual = self.silo_df['et_morton_actual'].to_numpy()
        self.et_morton_potential = self.silo_df['et_morton_potential'].to_numpy()
        self.et_morton_wet = self.silo_df['et_morton_wet'].to_numpy()
        self.mslp = self.silo_df['mslp'].to_numpy()
        
    def add_data_dict(self):
        self.data_dict = {'swl': self.swl,
                          'daily_rain': self.daily_rain, 
                          'max_temp': self.max_temp,
                          'min_temp': self.min_temp, 
                          'vp': self.vp, 
                          'vp_deficit': self.vp_deficit,
                          'evap_pan': self.evap_pan, 
                          'evap_syn': self.evap_syn,
                          'evap_comb': self.evap_comb,
                          'evap_morton_lake': self.evap_morton_lake,
                          'radiation': self.radiation,
                          'rh_tmax': self.rh_tmax, 
                          'rh_tmin': self.rh_tmin,
                          'et_short_crop': self.et_short_crop, 
                          'et_tall_crop': self.et_tall_crop,
                          'et_morton_actual': self.et_morton_actual,
                          'et_morton_potential': self.et_morton_potential,
                          'et_morton_wet': self.et_morton_wet,
                          'mslp': self.mslp,
                          'daily_delta_swl': self.daily_delta_swl,
                          'delta_swl': self.output_delta_swl}
        self.dates = self.gwl_dates
    
    def remove_null_dates(self):
        self.gwl_df, self.silo_df = import_functions.remove_null_dates(self.gwl_df, self.silo_df)

    


        
        
        
        # self.change_df = pd.DataFrame({'delta_swl': self.delta_swl})
        # for i in range(len(list(self.data_dict.keys()))):
        #     append_df = pd.DataFrame({list(self.data_dict.keys())[i]: list(self.data_dict.values())[i][1:]})
        #     self.change_df = pd.concat([self.change_df, append_df], axis=1)

        # corr = change_corr_df.corr()
        # sns.heatmap(corr)
        # plt.show()
    
    
    
    
    
        
        

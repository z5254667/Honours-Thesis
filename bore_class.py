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
    
        coastal_ids = ['GW036872.1.1', 'GW075025.1.1', 'GW080079.1.1', 'GW080980.1.1', 'GW081101.1.2']
        inland_ids = ['GW075405.1.1', 'GW080415.1.1', 'GW273314.1.1', 'GW403746.3.3']
        
        if self.id in coastal_ids:
            self.region = 'Coastal'
        elif self.id in inland_ids:
            self.region = 'Inland'
        else:
            self.region = 'Unknown'
 
    def add_dfs(self):
        self.bore_df = import_functions.get_bore_df(self.id)
        self.gwl_df = import_functions.get_gwl_df(self.id)
        self.silo_df = import_functions.get_silo_df(self.id)   
 
    def add_location(self):
        self.bore_latitude = self.bore_df['Latitude'][0]
        self.bore_longitude = self.bore_df['Longitude'][0]
        
        self.silo_latitude = self.silo_df['latitude'].to_numpy()[0]
        self.silo_longitude = self.silo_df['longitude'].to_numpy()[0]
        
    def remove_null_dates(self):
        self.gwl_df, self.silo_df = import_functions.remove_null_dates(self.gwl_df, self.silo_df)

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

    def add_gwl_data(self, leads):
        self.leads = leads
        self.gwl_dates = self.gwl_df['Date'].to_numpy()
        self.swl = self.gwl_df['Result (m)'].to_numpy()
        
        # Delta swl as an output
        self.output_delta_swl = []
        for i in range(self.leads, len(self.swl)):
            self.output_delta_swl.append(self.swl[i]-self.swl[i-self.leads])
        self.output_delta_swl = np.array(self.output_delta_swl) # length is = length of swl - number of leads
        
        # Daily delta swl as an input
        self.daily_delta_swl = []
        for i in range(1, len(self.swl)):
            self.daily_delta_swl.append(self.swl[i]-self.swl[i-1])
        self.daily_delta_swl = np.array(self.daily_delta_swl)
        
    def average_data(self, av_period=30):
        self.av_period = av_period
        self.dates = self.gwl_dates
        
        def convert_to_period_avs(data, time_period):
            #data must be one dimensional
            num_periods = len(data)//time_period
            groups = np.array([])
            for i in range(num_periods):
                groups = np.append(groups, data[i*time_period:time_period*(i+1)])
            groups = groups.reshape((num_periods,time_period))
            group_avs = np.array([])
            for group in groups:
                group_avs = np.append(group_avs, np.mean(group))
            return group_avs

        self.av_swl = convert_to_period_avs(self.swl, self.av_period)
        self.av_daily_rain = convert_to_period_avs(self.daily_rain, self.av_period)
        self.av_max_temp = convert_to_period_avs(self.max_temp, self.av_period)
        self.av_min_temp = convert_to_period_avs(self.min_temp, self.av_period)
        self.av_vp = convert_to_period_avs(self.vp, self.av_period)
        self.av_vp_deficit = convert_to_period_avs(self.vp_deficit, self.av_period)
        self.av_evap_pan = convert_to_period_avs(self.evap_pan, self.av_period)
        self.av_evap_syn = convert_to_period_avs(self.evap_syn, self.av_period)
        self.av_evap_comb = convert_to_period_avs(self.evap_comb, self.av_period)
        self.av_evap_morton_lake = convert_to_period_avs(self.evap_morton_lake, self.av_period)
        self.av_radiation = convert_to_period_avs(self.radiation, self.av_period)
        self.av_rh_tmax = convert_to_period_avs(self.rh_tmax, self.av_period)
        self.av_rh_tmin = convert_to_period_avs(self.rh_tmin, self.av_period)
        self.av_et_short_crop = convert_to_period_avs(self.et_short_crop, self.av_period)
        self.av_et_tall_crop = convert_to_period_avs(self.et_tall_crop, self.av_period)
        self.av_et_morton_actual = convert_to_period_avs(self.et_morton_actual, self.av_period)
        self.av_et_morton_potential = convert_to_period_avs(self.et_morton_potential, self.av_period)
        self.av_et_morton_wet = convert_to_period_avs(self.et_morton_wet, self.av_period)
        self.av_mslp = convert_to_period_avs(self.mslp, self.av_period)
        
        # calculate change in average swl (change calculated after average - not average calculated after change) --> FOR OUTPUTS
        self.av_output_delta_swl = np.array([])
        for i in range(self.leads, len(self.av_swl)):
            self.av_output_delta_swl = np.append(self.av_output_delta_swl, self.av_swl[i]-self.av_swl[i-self.leads]) # length is = length of av_swl - number of leads
        
        # calculate average period delta swl (change calculated after average - not average calculated after change) --> FOR INPUTS
        self.av_period_delta_swl = np.array([])
        for i in range(1, len(self.av_swl)):
            self.av_period_delta_swl = np.append(self.av_period_delta_swl, self.av_swl[i]-self.av_swl[i-1])
        
        # self.av_daily_delta_swl = convert_to_period_avs(self.daily_delta_swl, self.av_period)
        # self.av_output_delta_swl = convert_to_period_avs(self.output_delta_swl, self.av_period)
        
        self.av_dates = np.array([self.dates[i] for i in range(len(self.dates)) if i%self.av_period==self.av_period-1]) # Average date taken as the last date in the period being averaged
        
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
                          'delta_swl': self.output_delta_swl,
                          'av_swl': self.av_swl,
                          'av_daily_rain': self.av_daily_rain, 
                          'av_max_temp': self.av_max_temp,
                          'av_min_temp': self.av_min_temp, 
                          'av_vp': self.av_vp, 
                          'av_vp_deficit': self.av_vp_deficit,
                          'av_evap_pan': self.av_evap_pan, 
                          'av_evap_syn': self.av_evap_syn,
                          'av_evap_comb': self.av_evap_comb,
                          'av_evap_morton_lake': self.av_evap_morton_lake,
                          'av_radiation': self.av_radiation,
                          'av_rh_tmax': self.av_rh_tmax, 
                          'av_rh_tmin': self.av_rh_tmin,
                          'av_et_short_crop': self.av_et_short_crop, 
                          'av_et_tall_crop': self.av_et_tall_crop,
                          'av_et_morton_actual': self.av_et_morton_actual,
                          'av_et_morton_potential': self.av_et_morton_potential,
                          'av_et_morton_wet': self.av_et_morton_wet,
                          'av_mslp': self.av_mslp,
                          'av_period_delta_swl': self.av_period_delta_swl,
                          'av_delta_swl': self.av_output_delta_swl}
        # self.dates = self.gwl_dates
    

        
        # self.change_df = pd.DataFrame({'delta_swl': self.delta_swl})
        # for i in range(len(list(self.data_dict.keys()))):
        #     append_df = pd.DataFrame({list(self.data_dict.keys())[i]: list(self.data_dict.values())[i][1:]})
        #     self.change_df = pd.concat([self.change_df, append_df], axis=1)

        # corr = change_corr_df.corr()
        # sns.heatmap(corr)
        # plt.show()
    
    
    
    
    
        
        
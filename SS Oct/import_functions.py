#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:41:23 2022

@author: johnsalvaris
"""

import pandas as pd
import os
    
"""
NEED FUNCTION TO HANDLE MISSING DAYS --> possibly add back in dates
"""

def path(subfolders, file_name):
    main_folder = os.getcwd() # '/Users/johnsalvaris/OneDrive - UNSW/Thesis-John’s MacBook Pro 2020/'
    
    return main_folder + subfolders + file_name

def get_bore_df(bore_id):
    subfolders = '/' + bore_id + '/'
    bore_file = 'Bore Detailed.csv'    
    bore_path = path(subfolders, bore_file)
    bore_df = pd.read_csv(bore_path)
    
    return bore_df

def get_gwl_df(bore_id):
    subfolders = '/' + bore_id + '/'
    gwl_file = 'Water Level.csv'    
    gwl_path = path(subfolders, gwl_file)

    gwl_df = pd.read_csv(gwl_path, parse_dates=['Date'])
    gwl_df = gwl_df[gwl_df['Variable'] == 'SWL'] # Only use standing water level measurments
    gwl_df = gwl_df.sort_values(by='Date', ascending=True) # Force to be ascending date order
    
    return gwl_df
    
def get_silo_df(bore_id):
    subfolders = '/' + bore_id + '/'
    silo_file = 'SILO.csv'
    silo_path = path(subfolders, silo_file)
    
    silo_df = pd.read_csv(silo_path, parse_dates=['YYYY-MM-DD'])
    silo_df = silo_df.sort_values(by='YYYY-MM-DD', ascending=True) # Force to be ascending date order
    
    return silo_df

def remove_null_dates(gwl_df, silo_df):
    gwl_dates = gwl_df['Date'].to_numpy()
    silo_df = silo_df[silo_df['YYYY-MM-DD'].isin(gwl_dates)] # remove dates from silo data not in gwl data
    
    return gwl_df, silo_df



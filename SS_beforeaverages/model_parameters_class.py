#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:45:30 2022

@author: johnsalvaris
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler # min-max to 0-1
from sklearn.preprocessing import StandardScaler # z-score

from sklearn.model_selection import train_test_split

from torch import Tensor
from torch.autograd import Variable


class model_parameters():
    def __init__(self, silo_variables, gwl_input='standard', gwl_output='standard'): # define the input variable names
        self.input_variables = silo_variables.copy()
        self.gwl_input = gwl_input
        self.gwl_output = gwl_output
        
        if self.gwl_input == 'standard':
            self.input_variables.insert(0, 'swl')
        elif self.gwl_input == 'delta':
            self.input_variables.insert(0, 'daily_delta_swl')
        elif self.gwl_input != 'none':
            raise Exception("gwl_input should be 'standard', 'delta' or 'none'.")
            
        if self.gwl_output != 'standard' and self.gwl_output != 'delta':
            raise Exception("gwl_output should be 'standard' or 'delta'")

    def add_data(self, bore):
        if len(self.input_variables) == 0:
            raise Exception("No inputs assigned.")
        
        # # Define all dates corresponding to data
        # self.dates = bore.gwl_df['Date']
        # self.dates.reset_index(drop=True, inplace=True)
        # self.dates = self.dates.rename('date')
        
        # Define all dates corresponding to data - does not include dates before first predicted value
        self.output_dates = bore.gwl_df['Date']
        self.output_dates.reset_index(drop=True, inplace=True)
        self.output_dates = self.output_dates.rename('date')
        
        self.output_dates = self.output_dates.drop(index=self.output_dates.index[:self.times_in+self.out_after-1])
        self.output_dates.reset_index(drop=True, inplace=True)
        
        # Define swl data 
        self.swl_df = bore.gwl_df['Result (m)']
        self.swl_df.reset_index(drop=True, inplace=True)
        
        # Real swl values
        self.output_swl = self.swl_df
        self.output_swl = self.output_swl.drop(index=self.output_swl.index[:self.times_in+self.out_after-1])
        self.output_swl.reset_index(drop=True, inplace=True)
        
        
        # current time swl to calculate future level
        self.output_current_swl = self.swl_df
        self.output_current_swl = self.output_current_swl.drop(index=self.output_current_swl.index[:self.times_in-1])
        self.output_current_swl.reset_index(drop=True, inplace=True)
        self.output_current_swl = self.output_current_swl.drop(index=self.output_current_swl.index[len(self.output_current_swl)-self.out_after:])
        self.output_current_swl.reset_index(drop=True, inplace=True)
        
        
        # Output handling
        
        if self.gwl_output == 'standard':
            self.output_data = bore.gwl_df['Result (m)']
            self.output_data.reset_index(drop=True, inplace=True)
            self.output_data = self.output_data.rename('swl')
            
            # remove the first lag + lead -1 rows with no corresponding input
            self.output_data = self.output_data.drop(index=self.output_data.index[:self.times_in+self.out_after-1])
            self.output_data.reset_index(drop=True, inplace=True)
            
            
        elif self.gwl_output == 'delta':
            self.output_data = pd.DataFrame({'delta_swl': bore.output_delta_swl})

            # Drop the first lag - 1 rows (first lead rows already removed) with no corresponding input
            self.output_data = self.output_data.drop(index=self.output_data.index[:self.times_in-1])
            self.output_data.reset_index(drop=True, inplace=True)


        
        
        # Input handling
        
        # Isolates climate variables chosen
        self.input_data = bore.silo_df.drop(bore.silo_df.columns[~bore.silo_df.columns.isin(self.input_variables)], axis=1)
        self.input_data.reset_index(drop=True, inplace=True) 
        
        if self.gwl_input == 'standard':
            self.input_data = pd.concat((self.swl_df, self.input_data), axis=1)
            self.input_data = self.input_data.rename(columns={'Result (m)': 'swl'})
            
        elif self.gwl_input == 'delta':    
            # Remove first row of input data, output data and dates (no associated change in gwl)
            self.input_data = self.input_data.drop(index=self.input_data.index[0])
            self.input_data.reset_index(drop=True, inplace=True)
            
            self.output_dates = self.output_dates.drop(index=self.output_dates.index[0])
            self.output_dates.reset_index(drop=True, inplace=True)
            
            self.output_data = self.output_data.drop(index=self.output_data.index[0])
            self.output_data.reset_index(drop=True, inplace=True)
            
            self.output_swl = self.output_swl.drop(index=self.output_swl.index[0])
            self.output_swl.reset_index(drop=True, inplace=True)
            
            self.output_current_swl = self.output_current_swl.drop(index=self.output_current_swl.index[0])
            self.output_current_swl.reset_index(drop=True, inplace=True)
                  
            self.input_data = pd.concat((pd.DataFrame({'daily_delta_swl': bore.daily_delta_swl}), self.input_data), axis=1)


        # # Remove last rows equalling the lead time (no corresponding output)
        # seld.input_data = self.input_data.drop(index=self.input_data.index[-self.out_after:], axis=0, inplace=True) 
        # self.input_data.reset_index(drop=True, inplace=True)            


    def scale_data(self, scaler_type='mm'):        
        if scaler_type == 'mm':
            self.input_scaler = MinMaxScaler()
            self.output_scaler = MinMaxScaler()
        elif scaler_type == 'ss':
            self.input_scaler = StandardScaler()
            self.output_scaler = StandardScaler()
        else:
            raise Exception("Invalid scaler_type choice. Must be 'mm' or 'ss'")
        
        self.input_scaler = self.input_scaler.fit(self.input_data[self.input_data.columns])
        self.output_scaler = self.output_scaler.fit(self.output_data.values.reshape(-1,1))

        self.scaled_input = pd.DataFrame(self.input_scaler.transform(self.input_data[self.input_data.columns]))
        self.scaled_output = pd.DataFrame(self.output_scaler.transform(self.output_data.values.reshape(-1,1)))

    def add_in_out_params(self, epochs, times_in, out_after, times_out=1, between_outs=1):
        if times_out < 1:
            raise Exception("At least one output day required. Change times_out")
        if between_outs < 1:
            raise Exception("Minimum one day between outputs. Change between_outs")
        if out_after < 1:
            raise Exception("Output must be at least one day after final input. Change out_after")
        
        self.times_in = times_in # number of input days
        self.out_after = out_after # time to first predicition e.g. 7 for one week in the future
        self.times_out = times_out # number of output days
        self.between_outs = between_outs # consistent spacing between each output e.g. 7 for weekly
        self.num_epochs = epochs

    def format_inputs_outputs(self):        
        # entries = self.scaled_input.shape[0] - (self.times_out - 1) * self.between_outs - self.out_after - self.times_in + 1
        self.num_samples = self.scaled_output.shape[0]
        
        # format inputs
        input_list = []
        for i in range(self.num_samples):
            input_list.append(self.scaled_input.shift(-i, axis=0)[:self.times_in].to_numpy())
        self.formatted_inputs = np.array(input_list)
        
        # format outputs
        self.formatted_outputs = self.scaled_output.to_numpy()
        
        """
        Add back multiple outputs? Old code:
        
        # format outputs
        output_list = []
        out_index = [i for i in range(0, (self.times_out) * self.between_outs, self.between_outs)]
        for i in range(entries):
            output_list.append(self.scaled_output.shift(-self.out_after - self.times_in + 1 - i, axis=0)[0][out_index])
        self.formatted_outputs = np.array(output_list)

        self.used_dates = self.dates[self.times_in + self.out_after - 1:]
        self.unscaled_output = self.output_data[self.times_in + self.out_after - 1:]
        
        USED DATES DOES NOT MATCH REAL WHEN times_out >1
        """
        
    def divide_data(self, test_size=0.2, shuffle=False, keras_validation_split=0):
        self.test_size = test_size # % of dataset for testing
        self.shuffle = shuffle
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.formatted_inputs, self.formatted_outputs, test_size=self.test_size, shuffle=self.shuffle)
        self.train_dates, self.test_dates = train_test_split(self.output_dates, test_size=self.test_size, shuffle=self.shuffle)
        self.train_output_swl, self.test_output_swl = train_test_split(self.output_swl, test_size=self.test_size, shuffle=self.shuffle)
        self.train_output_current_swl, self.test_output_current_swl = train_test_split(self.output_current_swl, test_size=self.test_size, shuffle=self.shuffle)
        
        self.all_dates = np.concatenate([self.train_dates, self.test_dates])
        
        self.unscaled_y_train, self.unscaled_y_test = train_test_split(self.output_data, test_size=self.test_size, shuffle=self.shuffle)
        
        """
        USED DATES DOES NOT MATCH REAL WHEN times_out >1
        """

    def format_for_torch_LSTM(self):
        self.X_train_tensor = Variable(Tensor(self.X_train))
        self.X_test_tensor = Variable(Tensor(self.X_test))
        self.y_train_tensor = Variable(Tensor(self.y_train))
        self.y_test_tensor = Variable(Tensor(self.y_test))

    def format_for_sklearn_SVR(self):
        if self.X_train.ndim == 3:
            self.X_train_svr = self.X_train.reshape((self.X_train.shape[0],self.X_train.shape[1]*self.X_train.shape[2]))
            self.X_test_svr = self.X_test.reshape((self.X_test.shape[0],self.X_test.shape[1]*self.X_test.shape[2]))
        else:
            raise Exception('X_train not in format [samples, timesteps, features]')
        if self.y_train.shape[1] >= 2:
            raise Exception('SVR only capable of making one prediction at a time')
        else:
            self.y_train_svr = self.y_train.reshape(self.y_train.shape[0])
            self.y_test_svr = self.y_test.reshape(self.y_test.shape[0])


    # def add_train_prediction(self, y_hat_train):
    #     self.y_hat_train = y_hat_train
        
    # def add_test_prediction(self, y_hat_test):
    #     self.y_hat_test = y_hat_test

    def add_scores_summary(self, scores_dict):
        self.scores_dict = scores_dict

    def create_graph(self, bore_class, package='keras', model='lstm'): # Move to Bore class?
        model = model.upper()    
    
        """    
        fix when multiple outputs 
        
        if self.times_out > 1:
            # real_y_test --> SINGLE LINE
            np.concatenate(self.y_test[:,0],self.y_test[-1,-1])
        
        """
        # if self.y_hat_test is None:
        #     raise Exception('No prediction for testing period')
        # if self.y_hat_train is None:
        #     raise Exception('No prediction for training period')
        
        self.unscaled_y_hat_train = self.output_scaler.inverse_transform(self.y_hat_train)
        self.unscaled_y_hat_test = self.output_scaler.inverse_transform(self.y_hat_test)
        
        
        
        #Plot levels x 2
        #Plot change
        
        
        
        #TITLE 
        title = bore_class.id + ' - ' + package + ' ' + model + ' (Full Set)'
        # Include loss + metrics
        
        plt.figure()
        plt.plot(self.test_dates, self.unscaled_y_hat_test, label='Test Predictions')
        plt.plot(self.test_dates, self.unscaled_y_test, label='Test Actuals')
        plt.plot(self.train_dates, self.unscaled_y_hat_train, label='Training Predictions')
        plt.plot(self.train_dates, self.unscaled_y_train, label='Training Actuals')

        plt.title(title)
        plt.xlabel('Date')
        if self.gwl_output == 'standard':            
            plt.ylabel('SWL (m)')
        else:
            plt.ylabel('Delta SWL (m)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()
        
        plt.figure()
        
        title = bore_class.id + ' - ' + package + ' ' + model + ' (Test Set)'
        
        plt.plot(self.test_dates, self.unscaled_y_hat_test, label='Test Predictions')
        plt.plot(self.test_dates, self.unscaled_y_test, label='Test Actuals')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('SWL (m)')
        plt.legend()
        plt.tight_layout()
        plt.close()
    """ 
    
    FOR VALIDATION SPLIT FUNCTION
      
    def val(x,y,val_split):
        split_at = int(x.shape[0] * (1. - val_split))
        X_validate = x[split_at:]
        X_train = x[:split_at]
        y_validate = y[split_at: ]
        y_train = y[:split_at]
        return

    """

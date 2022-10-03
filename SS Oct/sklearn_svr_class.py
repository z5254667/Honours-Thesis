#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:48:08 2022

@author: johnsalvaris
"""

import matplotlib.pyplot as plt
import numpy as np
import score_functions
from sklearn.svm import SVR

class sklearn_SVR():
    def __init__(self, kernel='rbf', gamma='scale', epsilon='0.1', C=1.0, tolerance=0.001, degree=3, coef0=0.0, shrinking=True, verbose=0, shuffle_order=False):
        self.kernel = kernel
        self.gamma = gamma # 'scale' , 'auto'
        self.epsilon = epsilon # epsilon tube for no penalty to training loss function
        self.verbose = verbose
        self.tol = tolerance # tolerance for stopping criteria 
        self.C = C #Regularisation Parameter
        self.degree = degree
        self.coef0 = coef0 
        self.shuffle = shuffle_order
        self.shrinking = shrinking
        
        
        if self.verbose == 0:
            self.verbose = False
        elif self.verbose == 1 or self.verbose == 2:
            self.verbose = True
        else:
            raise Exception("Invalid verbose. Redefine as 0 (off), 1 or 2 (on)")
        
        """
        check correct gamma
        
        include parameter for tol --> tolerance for stopping criteria
     
        """
     
    def add_sets(self, model_parameters):
        self.X_train = model_parameters.X_train
        self.y_train = model_parameters.y_train        
        self.unscaled_y_train = model_parameters.unscaled_y_train.to_numpy()
        self.X_test = model_parameters.X_test
        self.y_test = model_parameters.y_test
        self.unscaled_y_test = model_parameters.unscaled_y_test.to_numpy()
        self.train_dates = model_parameters.train_dates.to_numpy()
        self.test_dates = model_parameters.test_dates.to_numpy()
        
        self.train_output_swl = model_parameters.train_output_swl.to_numpy()
        self.test_output_swl = model_parameters.test_output_swl.to_numpy()
        self.train_output_current_swl = model_parameters.train_output_current_swl.to_numpy()
        self.test_output_current_swl = model_parameters.test_output_current_swl.to_numpy()
        
        self.gwl_input = model_parameters.gwl_input
        self.gwl_output = model_parameters.gwl_output
        
    def create_model(self):
        self.model = SVR(kernel=self.kernel, gamma=self.gamma, epsilon=self.epsilon, C=self.C, tol=self.tol, degree=self.degree, coef0=self.coef0, verbose=self.verbose)
    
    def train_model(self, X, y):
        self.model.fit(X, y)
        
        """
        Add sample weights to force more empahsis on certain features??
        """
    
    def predict(self, X, scaler, dataset='test'):
        if dataset == 'test':
            self.y_hat_test = self.model.predict(X).reshape(X.shape[0], 1)
            self.unscaled_y_hat_test = scaler.inverse_transform(self.y_hat_test)
            if self.gwl_output == 'delta':
                self.y_hat_level_test = np.array(list(map(lambda current, change: current + change, self.test_output_current_swl, self.y_hat_test)))
        elif dataset == 'train':
            self.y_hat_train = self.model.predict(X).reshape(X.shape[0], 1)
            self.unscaled_y_hat_train = scaler.inverse_transform(self.y_hat_train)
            if self.gwl_output == 'delta':
                self.y_hat_level_train = np.array(list(map(lambda current, change: current + change, self.train_output_current_swl, self.y_hat_train)))

            
    def scores(self):
        self.train_rmse = score_functions.rmse(self.y_train, self.y_hat_train)
        self.train_mse = score_functions.mse(self.y_train, self.y_hat_train)
        self.train_r_squared = score_functions.r_squared(self.y_train, self.y_hat_train)
        self.train_mape = score_functions.mape(self.unscaled_y_train, self.unscaled_y_hat_train)
        self.train_mae = score_functions.mae(self.y_train, self.y_hat_train)
        #self.train_nse = score_functions.nse(self.y_train, self.y_hat_train)
        
        self.test_rmse = score_functions.rmse(self.y_test, self.y_hat_test)
        self.test_mse = score_functions.mse(self.y_test, self.y_hat_test)
        self.test_r_squared = score_functions.r_squared(self.y_test, self.y_hat_test)
        self.test_mape = score_functions.mape(self.unscaled_y_test, self.unscaled_y_hat_test)
        self.test_mae = score_functions.mae(self.y_test, self.y_hat_test)
        #self.train_nse = score_functions.nse(self.y_test, self.y_hat_test)
        
        self.train_scores_dict = {'Train Root Mean Squared Error': self.train_rmse,
                                'Train Mean Squared Error': self.train_mse,
                                'Train Coefficient of Determination': self.train_r_squared,
                                'Train Mean Absolute Percentage Error': self.train_mape,
                                'Train Mean Absolute Error': self.train_mae}
        
        self.test_scores_dict = {'Test Root Mean Squared Error': self.test_rmse,
                                 'Test Mean Squared Error': self.test_mse,
                                 'Test Coefficient of Determination': self.test_r_squared,
                                 'Test Mean Absolute Percentage Error': self.test_mape,
                                 'Test Mean Abolsute Error': self.test_mae}

    
    def plot_results(self, file_name): 
        model = 'SVR'
        package = 'Scikit Learn'
    
        """    
        fix when multiple outputs 
        
        if self.times_out > 1:
            # real_y_test --> SINGLE LINE
            np.concatenate(self.y_test[:,0],self.y_test[-1,-1])
        
        """
        
        if self.gwl_output == 'delta':
            self.y_axis_label = 'Change in SWL (m)'
        else:
            self.y_axis_label = 'SWL (m)'
        
        #TITLE 
        title = package + ' ' + model 
        # Include loss + metrics
        
        figure, axis = plt.subplots(2, 1, figsize=(8.27, 11.69))

        axis[0].axvline(x=self.train_dates[-1], c='r', linestyle='--') #size of the training set
        axis[0].plot(np.concatenate([self.train_dates, self.test_dates]), np.concatenate([self.unscaled_y_train, self.unscaled_y_test]), label='Actual')
        axis[0].plot(np.concatenate([self.train_dates, self.test_dates]), np.concatenate([self.unscaled_y_hat_train, self.unscaled_y_hat_test]), label='Predicted')


        axis[0].set_ylabel(self.y_axis_label)
        axis[0].set_title(title + ': Training and Testing Sets')
        axis[0].legend()
        
        axis[1].plot(self.test_dates, self.unscaled_y_test, label='Actual') 
        axis[1].plot(self.test_dates, self.unscaled_y_hat_test, label='Predicted')
        axis[1].set_ylabel(self.y_axis_label)
        axis[1].set_title(title + ': Testing Set')
        axis[1].legend()
        
        figure.tight_layout()

        # plt.savefig(predicted_data, format='pdf')
        plt.savefig(file_name, format='pdf')
        
        
        figure, axis = plt.subplots(2, 1, figsize=(8.27, 11.69))

        axis[0].axvline(x=self.train_dates[-1], c='r', linestyle='--') #size of the training set
        axis[0].plot(np.concatenate([self.train_dates, self.test_dates]), np.concatenate([self.unscaled_y_train, self.unscaled_y_test]), label='Actual')
        axis[0].plot(np.concatenate([self.train_dates, self.test_dates]), np.concatenate([self.unscaled_y_hat_train, self.unscaled_y_hat_test]), label='Predicted')
    
        axis[0].invert_yaxis()
        axis[0].set_ylabel('Inverted ' + self.y_axis_label)
        axis[0].set_title(title + ': Training and Testing Sets')
        axis[0].legend()
        
        axis[1].plot(self.test_dates, self.unscaled_y_test, label='Actual') 
        axis[1].plot(self.test_dates, self.unscaled_y_hat_test, label='Predicted')
        axis[1].invert_yaxis()
        axis[1].set_ylabel('Inverted ' + self.y_axis_label)
        axis[1].set_title(title + ': Testing Set')
        axis[1].legend()
        
        figure.tight_layout()
        plt.savefig(file_name, format='pdf')
        
        
        
        # add extra graph if needed
        
        if self.gwl_output == 'delta':
            self.y_axis_label = 'SWL (m)'
            
            figure, axis = plt.subplots(2, 1, figsize=(8.27, 11.69))

            axis[0].axvline(x=self.train_dates[-1], c='r', linestyle='--') #size of the training set
            axis[0].plot(np.concatenate([self.train_dates, self.test_dates]), np.concatenate([self.train_output_swl, self.test_output_swl]), label='Actual')
            axis[0].plot(np.concatenate([self.train_dates, self.test_dates]), np.concatenate([self.y_hat_level_train, self.y_hat_level_test]), label='Predicted')


            axis[0].set_ylabel(self.y_axis_label)
            axis[0].set_title(title + ': Training and Testing Sets')
            axis[0].legend()
            
            axis[1].plot(self.test_dates, self.test_output_swl, label='Actual') 
            axis[1].plot(self.test_dates, self.y_hat_level_test, label='Predicted')
            axis[1].set_ylabel(self.y_axis_label)
            axis[1].set_title(title + ': Testing Set')
            axis[1].legend()
            
            figure.tight_layout()

            # plt.savefig(predicted_data, format='pdf')
            plt.savefig(file_name, format='pdf')
            
            
            figure, axis = plt.subplots(2, 1, figsize=(8.27, 11.69))

            axis[0].axvline(x=self.train_dates[-1], c='r', linestyle='--') #size of the training set
            axis[0].plot(np.concatenate([self.train_dates, self.test_dates]), np.concatenate([self.train_output_swl, self.test_output_swl]), label='Actual')
            axis[0].plot(np.concatenate([self.train_dates, self.test_dates]), np.concatenate([self.y_hat_level_train, self.y_hat_level_test]), label='Predicted')
        
            axis[0].invert_yaxis()
            axis[0].set_ylabel('Inverted ' + self.y_axis_label)
            axis[0].set_title(title + ': Training and Testing Sets')
            axis[0].legend()
            
            axis[1].plot(self.test_dates, self.test_output_swl, label='Actual') 
            axis[1].plot(self.test_dates, self.y_hat_level_test, label='Predicted')
            axis[1].invert_yaxis()
            axis[1].set_ylabel('Inverted ' + self.y_axis_label)
            axis[1].set_title(title + ': Testing Set')
            axis[1].legend()
            
            figure.tight_layout()
            plt.savefig(file_name, format='pdf')
            
        plt.close()
        
        

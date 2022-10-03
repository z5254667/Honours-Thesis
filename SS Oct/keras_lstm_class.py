#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:39:13 2022

@author: johnsalvaris
"""

# import tensorflow as tf

import score_functions
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
# from tensorflow.keras.metrics import RootMeanSquaredError as RMSE


class keras_LSTM():
    def __init__(self, optimiser='adam', loss='mse', metric='accuracy', epochs=200, keras_validation_split=0.2, shuffle_order=False, verbose=0):
      
        self.optimiser = optimiser 
        self.loss = loss #mae, mse, rmse
        self.metric = metric  #MAKE THIS SO IT WORKS AS A LIST
        
        #used in train method
        self.epochs = epochs
        self.verbose = verbose
        self.val_split = keras_validation_split
        self.shuffle_order = shuffle_order
        
        # if self.val_split > 0 and self.val_split < 1:
        #     self.validation_data_used = True
        # else:
        #     self.validation_data_used = False
        
        self.metric_keys = [] # Change name to scores_keys?
        self.metric_values = [] # Change name to scores_values?
     
    def add_sets(self, model_parameters):
        self.X_train = model_parameters.X_train
        self.y_train = model_parameters.y_train  
        self.train_dates = model_parameters.train_dates.to_numpy()
        self.test_dates = model_parameters.test_dates.to_numpy()
        self.unscaled_y_train = model_parameters.unscaled_y_train.to_numpy()
        self.X_test = model_parameters.X_test
        self.y_test = model_parameters.y_test
        self.unscaled_y_test = model_parameters.unscaled_y_test.to_numpy()
        
        self.train_output_swl = model_parameters.train_output_swl.to_numpy()
        self.test_output_swl = model_parameters.test_output_swl.to_numpy()
        self.train_output_current_swl = model_parameters.train_output_current_swl.to_numpy()
        self.test_output_current_swl = model_parameters.test_output_current_swl.to_numpy()

        self.gwl_input = model_parameters.gwl_input
        self.gwl_output = model_parameters.gwl_output
        
    
    def create_network(self, input_shape, num_outputs=1, num_lstm_cells=64, num_fc_neurons=8, lstm_dropout_rate=0, lstm_recurrent_dropout_rate=0):
        self.full_input_shape = input_shape
        self.input_shape = input_shape[1:] # (lags, features)
        self.num_outputs = num_outputs # number of outputs 
        self.lstm_cells = num_lstm_cells
        self.fc_neurons = num_fc_neurons
        self.lstm_dropout = lstm_dropout_rate
        self.lstm_recurrent_dropout = lstm_recurrent_dropout_rate

        self.model = Sequential()
        
        # Input layer
        self.model.add(InputLayer(input_shape=self.input_shape)) 
        
        """
        Add a FC first?? --> num nodes = lags * features? --> Linear activation
        """
        
        #LSTM Layer
        # self.model.add(LSTM(self.lstm_cells, dropout=self.lstm_dropout, recurrent_dropout=self.lstm_recurrent_dropout))
        # self.model.add(LSTM(self.lstm_cells, dropout=self.lstm_dropout, recurrent_dropout=self.lstm_recurrent_dropout, return_sequences=True))
        self.model.add(LSTM(self.lstm_cells, dropout=self.lstm_dropout, recurrent_dropout=self.lstm_recurrent_dropout))
        #Fully Connected Hidden Layer
        self.model.add(Dense(self.fc_neurons, activation='relu')) # Linear Activation
        #Fully Connected Output Layer
        self.model.add(Dense(self.num_outputs)) #linear Activtion
        
        #compile
        self.model.compile(optimizer=self.optimiser, loss=self.loss, metrics=[self.metric])
    
    def train_model(self, X, y):                
        self.history = self.model.fit(X, y, epochs=self.epochs, verbose=self.verbose, validation_split=self.val_split, shuffle=self.shuffle_order)
        self.train_loss, self.train_metric = self.model.evaluate(X, y, verbose=self.verbose)

    # def predict(self, X, y, data_set='test'):
    #     # Change references to 'metric' to 'scores'?
    #     self.metric_keys.extend(list(map(lambda x: data_set + '_' + x, self.model.metrics_names)))
    #     self.metric_values.extend(self.model.evaluate(X, y, verbose=self.verbose))
        
    #     return self.model.predict(X, verbose=self.verbose) # Return to input into model_parameters class methods
    
    
    def predict(self, X, scaler, dataset='test'):        
        if dataset == 'test':
            self.y_hat_test = self.model.predict(X, verbose=self.verbose)
            self.unscaled_y_hat_test = scaler.inverse_transform(self.y_hat_test)
            if self.gwl_output == 'delta':
                self.y_hat_level_test = np.array(list(map(lambda current, change: current + change, self.test_output_current_swl, self.unscaled_y_hat_test)))
        elif dataset == 'train':
            self.y_hat_train = self.model.predict(X, verbose=self.verbose)  
            self.unscaled_y_hat_train = scaler.inverse_transform(self.y_hat_train)
            if self.gwl_output == 'delta':
                self.y_hat_level_train = np.array(list(map(lambda current, change: current + change, self.train_output_current_swl, self.unscaled_y_hat_train)))

        else:
            raise Exception("Specify if dataset is 'train' or 'test'")
        
    # def scores_dict(self):
    #     self.scores = dict(zip(self.metric_keys, self.metric_values))
    #     return self.scores

    def scores(self):
        self.train_rmse = score_functions.rmse(self.y_train, self.y_hat_train)
        self.train_mse = score_functions.mse(self.y_train, self.y_hat_train)
        self.train_r_squared = score_functions.r_squared(self.y_train, self.y_hat_train)
        self.train_mape = score_functions.mape(self.unscaled_y_train, self.unscaled_y_hat_train) # Uses unscaled to avoid division by 0
        self.train_mae = score_functions.mae(self.y_train, self.y_hat_train)
        #self.train_nse = score_functions.nse(self.y_train, self.y_hat_train)
        
        self.test_rmse = score_functions.rmse(self.y_test, self.y_hat_test)
        self.test_mse = score_functions.mse(self.y_test, self.y_hat_test)
        self.test_r_squared = score_functions.r_squared(self.y_test, self.y_hat_test)
        self.test_mape = score_functions.mape(self.unscaled_y_test, self.unscaled_y_hat_test) # Uses unscaled to avoid division by 0
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


    """
    function in model_parameters to make y_train in single line
    """
    
    def plot_results(self, file_name): # Move to Bore class?
        model = 'LSTM'
        package = 'TensorFlow Keras'
    
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
    
    def plot_loss(self, file_name):
        plt.figure(figsize=(8.27, 11.69))
        plt.plot(self.history.history['loss']) 
        if self.val_split > 0:
            plt.plot(self.history.history['val_loss'])
            plt.title('TensorFlow Keras Learning Curves')
            plt.legend(['Training Loss', 'Validation Loss'])
        else:
            plt.title('TensorFlow Keras Learning Curve')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.tight_layout()
        plt.savefig(file_name, format='pdf')
        plt.close()
        # figure, axis = plt.subplots(1, 1, figsize=(8.27, 11.69))
        # axis[0].plot(self.history.history['accuracy']) #['root_mean_squared_error'])
        # # axis[0].plot(self.history.history['val_accuracy'])
        # axis[0].set_title('TensorFlow Keras Training Accuracy')
        # axis[0].set_ylabel('Accuracy')
        # axis[0].set_xlabel('Epoch')
        # # axis[0].legend(['train', 'validation'], loc='upper left')
        
        # # axis[1].plot(self.history.history['loss'])
        # # # axis[1].plot(self.history.history['val_loss'])
        # # axis[1].set_title('TensorFlow Keras Training Loss')
        # # axis[1].set_ylabel('Loss')
        # # axis[1].set_xlabel('Epoch')
        # # # axis[1].legend(['train', 'validation'], loc='upper left')
        # figure.tight_layout()
        # plt.show()
        

         

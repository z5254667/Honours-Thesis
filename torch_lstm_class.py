#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:37:28 2022

@author: johnsalvaris
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import score_functions

from torch.autograd import Variable

class torch_LSTM(torch.nn.Module):
    def __init__(self, optimiser='adam', loss='mse', metric='accuracy', epochs=200, learning_rate=0.001, shuffle_order=False, verbose=0):
        super(torch_LSTM, self).__init__()
        self.opt_key = optimiser
        self.loss_key = loss
        # self.metric = metric
        self.num_epochs = epochs
        self.lr = learning_rate
        self.verbose = verbose  # 0 for silent, 1 for progress
        self.shuffle = shuffle_order
                
        """
        Determine how to implement:
            validation_split
            shuffle_order
            metric
        """
        
        def RMSELoss(y_hat, y):
            return torch.sqrt(torch.mean((y_hat - y)**2))
        
        self.loss_dict = {'mse': torch.nn.MSELoss(), 'mae': torch.nn.L1Loss(), 'rmse': RMSELoss}
        self.opt_dict = {'adam': torch.optim.Adam}
        
        if self.loss_key not in list(self.loss_dict.keys()):
            raise Exception("Non-default loss chosen, use: 'mse', 'mae' or 'rmse' instead.")
            
        if self.opt_key not in list(self.opt_dict.keys()):
            raise Exception("Non-default loss chosen, use: 'adam' instead.")
        
        self.criterion = self.loss_dict[self.loss_key]
        self.optimizer = self.opt_dict[self.opt_key]
        
        
    def create_network(self, input_shape, num_outputs, num_fc_neurons, lstm_dropout_rate=0, stacked_lstm=1):
        self.num_classes = num_outputs # number of output neurons 
        self.num_layers = stacked_lstm # number of stacked lstm layers 
        self.input_size = input_shape[-1] # (features) number of input features (variables) 
        self.hidden_size = num_fc_neurons # number of output neurons from lstm (neurons in subsequent layer)
        self.seq_length = input_shape[-2] # sequence length (number of lags)
        self.dropout = lstm_dropout_rate
        
        self.lstm_cells = self.seq_length * self.input_size
        
        """
        Is a FC input layer needed first?
        """
        
        # LSTM layer
        self.lstm_layer = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        # Fully Connected Hidden Layer
        self.fc_hidden_layer =  torch.nn.Linear(self.hidden_size, self.hidden_size) #self.hidden_size) # Linear activation
        # Fully Connected Output Layer
        self.fc_output_layer =  torch.nn.Linear(self.hidden_size, self.num_classes) #(self.hidden_size # Linear activation
    
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) # Initialise hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) # Initialise cell state
        
        # Propagate input through LSTM
        lstm_out, (hn, cn) = self.lstm_layer(x, (h_0, c_0)) 
        hn = hn.view(-1, self.hidden_size) # Reshape for fully connected layer
        fc_hidden_out = self.fc_hidden_layer(hn)
        fc_output_out = self.fc_output_layer(fc_hidden_out)
        
        return fc_output_out
    
    def train_model(self, X_train_tensor, y_train_tensor):    
        self.X_train_tensor = X_train_tensor
        self.y_train_tensor = y_train_tensor
        self.optimizer = self.optimizer(self.parameters(), lr=self.lr) 

        self.train() # Open model in training mode
        self.print_history = []
        self.history = []
        
        for epoch in range(self.num_epochs):
            outputs = self.forward(self.X_train_tensor) 
            self.optimizer.zero_grad() # Manually set gradient to 0
         
            self.loss = self.criterion(outputs, self.y_train_tensor)
            self.loss.backward() # Calculate the loss of the loss function             
            self.optimizer.step() # Backpropagation
            
            self.history.append((epoch, self.loss.item()))
            
            if self.verbose == 1:
                if epoch % 100 == 0:
                    self.print_history.append("Epoch: %d, Loss: %1.5f" % (epoch, self.loss.item()))
                    print(str(np.round(epoch/self.num_epochs * 100, 0))+ "% trained")
                elif epoch == self.num_epochs - 1:
                    # history.append("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
                    print("100% trained")
                    
            self.final_loss = self.loss.item()
            
    def predict(self, X_tensor, scaler, dataset='test'):        
        self.eval() # Open model in prediction mode
        if dataset == 'test':
            self.y_hat_test = self(X_tensor).data.numpy()
            self.unscaled_y_hat_test = scaler.inverse_transform(self.y_hat_test)
            if self.gwl_output == 'delta':
                self.y_hat_level_test = np.array(list(map(lambda current, change: current + change, self.test_output_current_swl, self.y_hat_test)))
        elif dataset == 'train':
            self.y_hat_train = self(X_tensor).data.numpy()   
            self.unscaled_y_hat_train = scaler.inverse_transform(self.y_hat_train)
            if self.gwl_output == 'delta':
                self.y_hat_level_train = np.array(list(map(lambda current, change: current + change, self.train_output_current_swl, self.y_hat_train)))

        else:
            raise Exception("Specify if dataset is 'train' or 'test'")

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

    # def add_sets(self, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_dates, test_dates, unscaled_y_train, unscaled_y_test):
    #     self.X_train = X_train_tensor.data.numpy()
    #     self.y_train = y_train_tensor.data.numpy()        
    #     self.unscaled_y_train = unscaled_y_train

    #     self.X_test = X_test_tensor.data.numpy()
    #     self.y_test = y_test_tensor.data.numpy()
    #     self.unscaled_y_test = unscaled_y_test
    
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

    def plot_results(self, file_name): # Move to Bore class?
        model = 'LSTM'
        package = 'PyTorch'
    
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
        his_array = np.array(self.history)
        epochs = his_array[:,0]+1
        epoch_loss = his_array[:,1]
        
        plt.figure(figsize=(8.27, 11.69))
        plt.title('PyTorch Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.ylabel(self.loss_key)
        plt.plot(epochs, epoch_loss)
        plt.tight_layout()
        plt.savefig(file_name, format='pdf')
        plt.close()
        
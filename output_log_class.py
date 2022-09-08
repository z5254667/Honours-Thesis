#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:49:11 2022

@author: johnsalvaris
"""

#plot summary
#plot variables in
#plot model graphs
#plot model summary + model scores + training logs

import import_functions
import matplotlib.pyplot as plt
import numpy as np
import PyPDF2

from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf

"""
Output predictions to a xlsx??
"""

class output_log():
    def __init__(self, start_date_time, start_time, end_time, bore, model_parameters, keras_model, sklearn_model):
        self.start_time = start_time
        self.end_time = end_time
        self.start_date_time = start_date_time
        self.bore = bore
        self.model_parameters = model_parameters
        self.keras_model = keras_model
        self.sklearn_model = sklearn_model
        self.bore_id = bore.id
    
        # Set up file paths
        self.temp_input_graph_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_input_graphs.pdf')
        self.temp_general_text_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_general_text.pdf')
        self.temp_lstm_text_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_lstm_text.pdf')
        self.temp_keras_graph_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_keras_graphs.pdf')
        self.temp_section_4_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_section_4.pdf')
        self.temp_sklearn_graph_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_sklearn_graphs.pdf')

        
        def print_scores(scores_dict, output_text):
            for key, value in scores_dict.items():
                output_text.append(key + str(': ') + str(scores_dict[key]))
                
        self.inputs_dict = {'swl': 'Standing Water Level (m)', 
                            'daily_rain': 'Rainfall (mm)', 
                            'max_temp': 'Maximum Temperature (°C)',
                            'min_temp': 'Minimum Temperature (°C)', 
                            'vp': 'Vapour Pressure (hPa)', 
                            'vp_deficit': 'Vapour Pressure Deficit (hPa)',
                            'evap_pan': 'Evaporation - Class A Pan (mm)', 
                            'evap_syn': 'Evaporation - Synthetic Estimate (mm)',
                            'evap_comb': 'Evaporation - Combination (Synthetic Estimate pre-1970, Class A Pan 1970 Onwards) (mm)',
                            'evap_morton_lake': "Evaporation - Morton's Shallow Lake Evaporation (mm)",
                            'radiation': 'Solar Radiation - Total Incoming Downward Shortwave Radiation on a Horizontal Surface (MJ/m^2)',
                            'rh_tmax': 'Relative Humidity at Time of Maximum Temperature (%)', 
                            'rh_tmin': 'Relative Humidity at Time of Minimum Temperature (%)',
                            'et_short_crop': 'Evapotranspiration - FAO56 Short Crop (mm)', 
                            'et_tall_crop': 'Evapotranspiration - ASCE Tall Crop (mm)',
                            'et_morton_actual': "Evapotranspiration - Morton's Areal Actual Evapotranspiration (mm)",
                            'et_morton_potential': "Evapotranspiration - Morton's Potential Evapotranspiration", 
                            'et_morton_wet': "Evapotranspiration - Wet-Environment Areal Evapotranspiration Over Land (mm)",
                            'mslp': 'Mean Sea Level Pressure (hPa)',
                            'delta_swl': 'Change in Standing Water Level (m)',
                            'daily_delta_swl': 'Daily Change in Standing Water Level (m)',
                            #averages
                            'av_swl': 'Average Standing Water Level (m)', 
                            'av_daily_rain': 'Average Rainfall (mm)', 
                            'av_max_temp': 'Average Maximum Temperature (°C)',
                            'av_min_temp': 'Average Minimum Temperature (°C)', 
                            'av_vp': 'Average Vapour Pressure (hPa)', 
                            'av_vp_deficit': 'Average Vapour Pressure Deficit (hPa)',
                            'av_evap_pan': 'Average Evaporation - Class A Pan (mm)', 
                            'av_evap_syn': 'Average Evaporation - Synthetic Estimate (mm)',
                            'av_evap_comb': 'Average Evaporation - Combination (Synthetic Estimate pre-1970, Class A Pan 1970 Onwards) (mm)',
                            'av_evap_morton_lake': "Average Evaporation - Morton's Shallow Lake Evaporation (mm)",
                            'av_radiation': 'Average Solar Radiation - Total Incoming Downward Shortwave Radiation on a Horizontal Surface (MJ/m^2)',
                            'av_rh_tmax': 'Average Relative Humidity at Time of Maximum Temperature (%)', 
                            'av_rh_tmin': 'Average Relative Humidity at Time of Minimum Temperature (%)',
                            'av_et_short_crop': 'Average Evapotranspiration - FAO56 Short Crop (mm)', 
                            'av_et_tall_crop': 'Average Evapotranspiration - ASCE Tall Crop (mm)',
                            'av_et_morton_actual': "Average Evapotranspiration - Morton's Areal Actual Evapotranspiration (mm)",
                            'av_et_morton_potential': "Average Evapotranspiration - Morton's Potential Evapotranspiration", 
                            'av_et_morton_wet': "Average Evapotranspiration - Wet-Environment Areal Evapotranspiration Over Land (mm)",
                            'av_mslp': 'Average Mean Sea Level Pressure (hPa)',
                            'av_delta_swl': 'Average Change in Standing Water Level (m)',
                            'av_period_delta_swl': 'Average Period Change in Standing Water Level (m)'}
        
        self.run_time = end_time - start_time
        
        self.num_inputs = len(self.model_parameters.input_variables)
        
        self.used_inputs = []
        self.used_outputs = self.inputs_dict[model_parameters.output_variables]
        for input_var in model_parameters.input_variables:
            self.used_inputs.append(self.inputs_dict[input_var])
        
        
    
        # log file name strings
        self.file_name_date_time = self.start_date_time.strftime("%Y_%m_%d__%H_%M_%S")
        
        # time script run string
        self.start_date_time_string = self.start_date_time.strftime("%d/%m/%Y - %H:%M:%S")
    
        # Data range strings
        start_date_year = str(bore.gwl_dates[0])[:4]
        start_date_month = str(bore.gwl_dates[0])[5:7]
        start_date_day = str(bore.gwl_dates[0])[8:10]                
        end_date_year = str(bore.gwl_dates[-1])[:4]
        end_date_month = str(bore.gwl_dates[-1])[5:7]
        end_date_day = str(bore.gwl_dates[-1])[8:10]
        self.data_range_start_date = start_date_day +  '/' + start_date_month + '/' + start_date_year
        self.data_range_end_date = end_date_day +  '/' + end_date_month + '/' + end_date_year
    
        # Location
        self.bore_coordinates = '(' + str(bore.bore_latitude) + ', ' + str(bore.bore_longitude) + ')'
        self.silo_coordinates = '(' + str(bore.silo_latitude) + ', ' + str(bore.silo_longitude) + ')'
    
#%%    
        ## Section 1 (General text)
        self.output_text_general = []
        self.output_text_general.append('<><> Time Stamp <><>')
        self.output_text_general.append('')
        self.output_text_general.append(self.start_date_time_string)
        self.output_text_general.append('Total Run Time: ' + str(self.run_time) + 's')
        self.output_text_general.append('')
        self.output_text_general.append('')
        self.output_text_general.append('<><> Bore Information <><>')
        self.output_text_general.append('')
        self.output_text_general.append('Bore ID: ' + self.bore.id)
        self.output_text_general.append('Bore Coordinates: ' + self.bore_coordinates)
        self.output_text_general.append('Silo Grid Point Coordinates: ' + self.silo_coordinates)
        self.output_text_general.append('Region: ' + self.bore.region)
        self.output_text_general.append('')
        self.output_text_general.append('')
        self.output_text_general.append('<><> Model Output <><>')
        self.output_text_general.append('')
        self.output_text_general.append('Averaged Period: ' + str(bore.av_period) + ' day(s)')
        self.output_text_general.append('Output: ' + str(self.used_outputs) + ' in ' + str(self.model_parameters.out_after) + ' period(s) time')        
        self.output_text_general.append('')
        self.output_text_general.append('')
        self.output_text_general.append('<><> Model Inputs <><>')
        self.output_text_general.append('')
        self.output_text_general.append('Data Range: ' + self.data_range_start_date + ' - ' + self.data_range_end_date)
        self.output_text_general.append('Train Set Size: ' + str((1-self.model_parameters.test_size)*100) + '%')
        self.output_text_general.append('Test Set Size: ' + str(self.model_parameters.test_size*100) + '%')        
        self.output_text_general.append('Input Timesteps: Current period + ' + str(self.model_parameters.times_in - 1) + ' preceeding period(s)')
        self.output_text_general.append('Input Variables (Daily if not specified as average): ')
        for iv in self.used_inputs:
            self.output_text_general.append('    ' + iv)
            
        self.output_text_general.append('')
        self.output_text_general.append('')

        self.temp_general_text_file = FPDF()
        self.temp_general_text_file.add_page()
        self.temp_general_text_file.set_font('Courier', size=9)
        
        for line in self.output_text_general:
            self.temp_general_text_file.cell(0, 5, txt=line, ln=1, align='L')
        
        self.temp_general_text_file.output(self.temp_general_text_path)

# %%
        ## Input Graphs 
        self.temp_input_graph_file = PdfPages(self.temp_input_graph_path)
        
        if self.num_inputs == 1:
            plt.figure(figsize=(8.27, 11.69))
            if model_parameters.input_variables[0] == 'daily_delta_swl' or model_parameters.input_variables[0] == 'av_period_delta_swl':
                plt.plot(model_parameters.dates_for_graph[1:], bore.data_dict[model_parameters.input_variables[0]])
            else:
                plt.plot(model_parameters.dates_for_graph, bore.data_dict[model_parameters.input_variables[0]])
            plt.title(self.inputs_dict[model_parameters.input_variables[0]], fontsize=11)
            plt.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
        
            fig, ax = plt.subplots(1, 2, figsize=(8.27,11.69))
            plot_acf(bore.data_dict[model_parameters.input_variables[0]], lags=model_parameters.times_in, ax=ax[0], 
                     title=self.inputs_dict[model_parameters.input_variables[0]] + ' - Autocorrelation')
            plot_pacf(bore.data_dict[model_parameters.input_variables[0]], lags=model_parameters.times_in, ax=ax[1],
                      title=self.inputs_dict[model_parameters.input_variables[0]] + ' - Partial Autocorrelation', method='ywm')
            
            if len(self.inputs_dict[model_parameters.input_variables[0]] + ' - Partial Autocorrelation') >= 90:
                ax[0].set_title(self.inputs_dict[model_parameters.input_variables[0]] + ' - Autocorrelation',fontsize=4.5)
                ax[1].set_title(self.inputs_dict[model_parameters.input_variables[0]] + ' - Partial Autocorrelation',fontsize=4.5)
            
            fig.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
            
            # acf(bore.data_dict[model_parameters.input_variables[0]], lags=model_parameters.times_in)
            # pacf(bore.data_dict[model_parameters.input_variables[0]], lags=model_parameters.times_in)
        elif self.num_inputs <= 10:
            figure, axis = plt.subplots(self.num_inputs, 1, figsize=(8.27, 11.69))
            for i in range(self.num_inputs):
                if model_parameters.input_variables[i] == 'daily_delta_swl' or model_parameters.input_variables[i] == 'av_period_delta_swl':
                    axis[i].plot(model_parameters.dates_for_graph[1:], bore.data_dict[model_parameters.input_variables[i]])
                else:
                    axis[i].plot(model_parameters.dates_for_graph, bore.data_dict[model_parameters.input_variables[i]])
                axis[i].set_title(self.inputs_dict[model_parameters.input_variables[i]], fontsize=11)
            figure.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
            
            fig, ax = plt.subplots(self.num_inputs, 2, figsize=(8.27,11.69))
            for i in range(self.num_inputs):
                plot_acf(bore.data_dict[model_parameters.input_variables[i]], lags=model_parameters.times_in, ax=ax[i,0], 
                         title=self.inputs_dict[model_parameters.input_variables[i]] + ' - Autocorrelation')
                plot_pacf(bore.data_dict[model_parameters.input_variables[i]], lags=model_parameters.times_in, ax=ax[i,1], method='ywm')
                
                ax[i,0].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Autocorrelation',fontsize=7)
                ax[i,1].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation',fontsize=7)
                
                if len(self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation') >= 90:
                    ax[i,0].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Autocorrelation',fontsize=4.5)
                    ax[i,1].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation',fontsize=4.5)
                
            fig.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
            
        else:
            num_per_page = int(np.ceil(self.num_inputs/2))
            page_1 = num_per_page
            figure, axis = plt.subplots(page_1, 1, figsize=(8.27, 11.69))
            for i in range(num_per_page):
                if model_parameters.input_variables[i] == 'daily_delta_swl' or model_parameters.input_variables[i] == 'av_period_delta_swl':
                    axis[i].plot(model_parameters.dates_for_graph[1:], bore.data_dict[model_parameters.input_variables[i]])
                else:
                    axis[i].plot(model_parameters.dates_for_graph, bore.data_dict[model_parameters.input_variables[i]])
                axis[i].set_title(self.inputs_dict[model_parameters.input_variables[i]], fontsize=11)
            figure.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
            
            page_2 = int(self.num_inputs - num_per_page)
            figure, axis = plt.subplots(page_2, 1, figsize=(8.27, 11.69))
            for i in range(num_per_page, self.num_inputs):
                if model_parameters.input_variables[i] == 'daily_delta_swl' or model_parameters.input_variables[i] == 'av_period_delta_swl':
                    axis[i-num_per_page].plot(model_parameters.dates_for_graph[1:], bore.data_dict[model_parameters.input_variables[i]])
                else:
                    axis[i-num_per_page].plot(model_parameters.dates_for_graph, bore.data_dict[model_parameters.input_variables[i]])
                axis[i-num_per_page].set_title(self.inputs_dict[model_parameters.input_variables[i]], fontsize=11)
            figure.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
            
            
            #acf/pacf
            fig, ax = plt.subplots(page_1, 2, figsize=(8.27,11.69))
            for i in range(num_per_page):
                plot_acf(bore.data_dict[model_parameters.input_variables[i]], lags=model_parameters.times_in, ax=ax[i,0], 
                         title=self.inputs_dict[model_parameters.input_variables[i]] + ' - Autocorrelation')
                plot_pacf(bore.data_dict[model_parameters.input_variables[i]], lags=model_parameters.times_in, ax=ax[i,1],
                          title=self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation', method='ywm')
                
                ax[i,0].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Autocorrelation',fontsize=7)
                ax[i,1].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation',fontsize=7)
                
                if len(self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation') >= 90:
                    ax[i,0].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Autocorrelation',fontsize=4.5)
                    ax[i,1].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation',fontsize=4.5)
                
            fig.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
            
            fig, ax = plt.subplots(page_2, 2, figsize=(8.27,11.69))
            for i in range(num_per_page, self.num_inputs):
                plot_acf(bore.data_dict[model_parameters.input_variables[i-num_per_page]], lags=model_parameters.times_in, ax=ax[i-num_per_page,0], 
                         title=self.inputs_dict[model_parameters.input_variables[i]] + ' - Autocorrelation')
                plot_pacf(bore.data_dict[model_parameters.input_variables[i-num_per_page]], lags=model_parameters.times_in, ax=ax[i-num_per_page,1],
                          title=self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation', method='ywm')
                
                ax[i-num_per_page,0].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Autocorrelation',fontsize=7)
                ax[i-num_per_page,1].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation',fontsize=7)
                
                if len(self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation') >= 90:
                    ax[i-num_per_page,0].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Autocorrelation',fontsize=4.5)
                    ax[i-num_per_page,1].set_title(self.inputs_dict[model_parameters.input_variables[i]] + ' - Partial Autocorrelation',fontsize=4.5)
                
            fig.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
        
        self.temp_input_graph_file.close()
        

        #acf and pacf go here
        #choose alpha (CI)


#%%
        ## Section (Keras)
        
        self.output_text_lstm = []
        
        self.output_text_lstm.append('<><> TensorFlow Keras LSTM Model <><>')
        self.output_text_lstm.append('')
        self.output_text_lstm.append('Optimiser: ' + str(self.keras_model.optimiser))
        self.output_text_lstm.append('Loss: ' + str(self.keras_model.loss))
        # self.output_text_2.append('Metric: ' + str(self.keras_model.metric))
        self.output_text_lstm.append('Number of Epochs: '+ str(self.keras_model.epochs))
        self.output_text_lstm.append('Percentage of Training Data for Validation: ' + str(self.keras_model.val_split * 100) + '%')
        if self.keras_model.shuffle_order == True:
            self.output_text_lstm.append('Time Series Order: Shufffled')
        else:
            self.output_text_lstm.append('Time Series Order: Chronological')
        if self.keras_model.verbose == 0:
            self.output_text_lstm.append('Verbose: Off')
        else:
            self.output_text_lstm.append('Verbose: On')
            
        self.output_text_lstm.append('')
        self.output_text_lstm.append('')
        
        self.output_text_lstm.append('<><> Model Architecture <><>')
        self.output_text_lstm.append('')
        self.output_text_lstm.append('Input Shape (Samples, Timesteps, Features): ' + str(self.keras_model.full_input_shape))
        
        self.output_text_lstm.append('LSTM Cells: ' + str(self.keras_model.lstm_cells))
        self.output_text_lstm.append('Fully Connected Hidden Neurons: ' + str(self.keras_model.fc_neurons))
        self.output_text_lstm.append('Fully Connected Output Neurons: ' + str(self.keras_model.num_outputs))
        self.output_text_lstm.append('LSTM Dropout Rate: ' + str(self.keras_model.lstm_dropout * 100) + '%')
        self.output_text_lstm.append('LSTM Recurrent Dropout Rate: ' + str(self.keras_model.lstm_recurrent_dropout * 100) + '%')
        self.output_text_lstm.append('')
        
        summary_string = []
        self.keras_model.model.summary(print_fn=lambda x: summary_string.append(x))
        for line in summary_string:
            self.output_text_lstm.append(line)
   
        self.output_text_lstm.append('')
        self.output_text_lstm.append('')
        if self.keras_model.val_split > 0:
            self.output_text_lstm.append('<><> Training and Validation Loss <><>')
        else:
            self.output_text_lstm.append('<><> Training Loss <><>')
        self.output_text_lstm.append('')
        self.output_text_lstm.append('Training Loss:')
        
        for i in range(len(self.keras_model.history.history['loss'])):
            if (i+1)%(len(self.keras_model.history.history['loss'])//10) == 0:
                self.output_text_lstm.append('    Epoch: ' + str(i+1) + ', Loss: ' + str(self.keras_model.history.history['loss'][i]))    
        
        if self.keras_model.val_split > 0:
            self.output_text_lstm.append('')
            self.output_text_lstm.append('Validation Loss:')
            for i in range(len(self.keras_model.history.history['val_loss'])):
                if (i+1)%(len(self.keras_model.history.history['val_loss'])//10) == 0:
                    self.output_text_lstm.append('    Epoch: ' + str(i+1) + ', Loss: ' + str(self.keras_model.history.history['val_loss'][i]))    
            
        self.output_text_lstm.append('')
        self.output_text_lstm.append('')

        self.output_text_lstm.append('<><> Results <><>')
        self.output_text_lstm.append('')

        self.output_text_lstm.append('Train Scores')
        self.output_text_lstm.append('')
        print_scores(self.keras_model.train_scores_dict, self.output_text_lstm)
        
        self.output_text_lstm.append('')
        self.output_text_lstm.append('Test Scores')
        self.output_text_lstm.append('')
        print_scores(self.keras_model.test_scores_dict, self.output_text_lstm)    

        self.temp_lstm_text_file = FPDF()
        self.temp_lstm_text_file.add_page()
        self.temp_lstm_text_file.set_font('Courier', size=9)
        
        for line in self.output_text_lstm:
            self.temp_lstm_text_file.cell(0, 5, txt=line, ln=1, align='L')
        
        self.temp_lstm_text_file.output(self.temp_lstm_text_path)

        ## Keras Graphs
        self.temp_keras_graph_file = PdfPages(self.temp_keras_graph_path)

        self.keras_model.plot_loss(self.temp_keras_graph_file)
        self.keras_model.plot_results(self.temp_keras_graph_file)
        self.temp_keras_graph_file.close()
        
        
#%%        ## Section 4 (SVR)
        
        self.output_text_4 = []
        self.output_text_4.append('<><> Scikit Learn SVR Model <><>')
        self.output_text_4.append('')
        self.output_text_4.append('Kernel Function: ' + str(self.sklearn_model.kernel))
        self.output_text_4.append('Kernel Coefficient for RBF/Polynomial/Sigmoid: ' + str(self.sklearn_model.gamma))
        self.output_text_4.append('Independent Kernel Term for Polynomial/Sigmoid: ' + str(self.sklearn_model.coef0))
        self.output_text_4.append('Degree of Polynomial Kernel Function: ' + str(self.sklearn_model.degree))
        self.output_text_4.append('Epsilon: ' + str(self.sklearn_model.epsilon))
        self.output_text_4.append('Stopping Criterion Tolerance: ' + str(self.sklearn_model.tol))
        self.output_text_4.append('Regularisation Parameter: ' + str(self.sklearn_model.C))
        self.output_text_4.append('Shrinking: ' + str(self.sklearn_model.shrinking))
        
        if self.sklearn_model.shuffle == True:
            self.output_text_4.append('Time Series Order: Shufffled')
        else:
            self.output_text_4.append('Time Series Order: Chronological')

        if self.sklearn_model.verbose == 0:
            self.output_text_4.append('Verbose: Off')
        else:
            self.output_text_4.append('Verbose: On')
            
        self.output_text_4.append('')
        self.output_text_4.append('')
        
        self.output_text_4.append('<><> Model Architecture <><>')
        self.output_text_4.append('')
        self.output_text_4.append('Number of Support Vectors: ' + str(self.sklearn_model.model.n_support_[0]))
        self.output_text_4.append('Input/Support Vector Size: ' + str(self.sklearn_model.model.n_features_in_))
        self.output_text_4.append('')
        
        """ ADD COMMENT BELOW TO SPREADSHEET? --> Fit so fits neatly in the log?? """
        # self.output_text_4.append('Model Intercept: ' + str(self.sklearn_model.model.intercept_[0]))
        # self.output_text_4.append('Support Vector Indicies: ')
        # for i in range(len(self.sklearn_model.model.support_)):
        #     self.output_text_4.append('    ' + str(self.sklearn_model.model.support_[i]))
        # self.output_text_4.append('')
        # self.output_text_4.append('Support Vector Dual Coefficients: ')
        # for i in range(len(self.sklearn_model.model.dual_coef_[0])):
        #     self.output_text_4.append('    ' + str(self.sklearn_model.model.dual_coef_[0][i]))
        
        # self.output_text_4.append('')
        # """
        # MAKE THE SVs Fit On Screen!!
     
        # self.output_text_4.append('Support Vectors: ')
        # for i in range(self.sklearn_model.model.support_vectors_.shape[0]):
        #    self.output_text_4.append('    ' + str(self.sklearn_model.model.support_vectors_[i,:]))
        
        # """
        
        self.output_text_4.append('')
        self.output_text_4.append('')
        self.output_text_4.append('<><> Training Loss <><>')
        self.output_text_4.append('')
        self.output_text_4.append('TBC using https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html')
        self.output_text_4.append('')
        self.output_text_4.append('')
        
        self.output_text_4.append('<><> Results <><>')
        self.output_text_4.append('')

        self.output_text_4.append('Train Scores')
        self.output_text_4.append('')
        print_scores(self.sklearn_model.train_scores_dict, self.output_text_4)
        
        self.output_text_4.append('')
        self.output_text_4.append('Test Scores')
        self.output_text_4.append('')
        print_scores(self.sklearn_model.test_scores_dict, self.output_text_4)
        self.output_text_4.append('')
        self.output_text_4.append('')

        self.temp_section_4_file = FPDF()
        self.temp_section_4_file.add_page()
        self.temp_section_4_file.set_font('Courier', size=9)
        
        for line in self.output_text_4:
            self.temp_section_4_file.cell(0, 5, txt=line, ln=1, align='L')
        
        self.temp_section_4_file.output(self.temp_section_4_path)

        # sklearn graphs
        self.temp_sklearn_graph_file = PdfPages(self.temp_sklearn_graph_path)
    
        self.sklearn_model.plot_results(self.temp_sklearn_graph_file)
        self.temp_sklearn_graph_file.close()  

        self.log_file_name = 'output_log_' + self.file_name_date_time + '.pdf'
        self.log_file_path = import_functions.path('/' + self.bore_id + '/logs/', self.log_file_name)
        
        pdfs = [self.temp_general_text_path, self.temp_input_graph_path, 
                self.temp_lstm_text_path, self.temp_keras_graph_path, 
                self.temp_section_4_path, self.temp_sklearn_graph_path]
        
        merger = PyPDF2.PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(self.log_file_path)
        merger.close()
        

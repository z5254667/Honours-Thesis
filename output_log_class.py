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

"""
Output predictions to a xlsx??
"""

class output_log():
    def __init__(self, start_date_time, start_time, end_time, bore, model_parameters, keras_model, sklearn_model, torch_model):
        self.start_time = start_time
        self.end_time = end_time
        self.start_date_time = start_date_time
        self.bore = bore
        self.model_parameters = model_parameters
        self.keras_model = keras_model
        self.sklearn_model = sklearn_model
        self.torch_model = torch_model
        self.bore_id = bore.id
    
        self.temp_input_graph_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_input_graphs.pdf')
        self.temp_section_1_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_section_1.pdf')
        self.temp_section_2_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_section_2.pdf')
        self.temp_keras_graph_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_keras_graphs.pdf')
        self.temp_section_3_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_section_3.pdf')
        self.temp_torch_graph_path = import_functions.path('/' + self.bore_id + '/temp/', 'temp_torch_graphs.pdf')
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
                            'daily_delta_swl': 'Daily Change in Standing Water Level (m)'}
        
        self.run_time = end_time - start_time
        
        self.num_inputs = len(self.model_parameters.input_variables)
        
        self.used_inputs = []
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
    
    
        ## Section 1 (General)
        self.output_text_1 = []
        self.output_text_1.append('<><> Time Stamp <><>')
        self.output_text_1.append('')
        self.output_text_1.append(self.start_date_time_string)
        self.output_text_1.append('Total Run Time: ' + str(self.run_time) + 's')
        self.output_text_1.append('')
        self.output_text_1.append('')
        self.output_text_1.append('<><> Bore Information <><>')
        self.output_text_1.append('')
        self.output_text_1.append('Bore ID: ' + self.bore.id)
        self.output_text_1.append('Bore Coordinates: ' + self.bore_coordinates)
        self.output_text_1.append('Silo Grid Point Coordinates: ' + self.silo_coordinates)
        self.output_text_1.append('')
        self.output_text_1.append('')
        self.output_text_1.append('<><> Model Output <><>')
        self.output_text_1.append('')
        self.output_text_1.append('Output: Standing Water Level (m) in ' + str(self.model_parameters.out_after) + ' days time')        
        self.output_text_1.append('')
        self.output_text_1.append('')
        self.output_text_1.append('<><> Model Inputs <><>')
        self.output_text_1.append('')
        self.output_text_1.append('Data Range: ' + self.data_range_start_date + ' - ' + self.data_range_end_date)
        self.output_text_1.append('Train Set Size: ' + str((1-self.model_parameters.test_size)*100) + '%')
        self.output_text_1.append('Test Set Size: ' + str(self.model_parameters.test_size*100) + '%')        
        self.output_text_1.append('Input Timesteps: Current day + ' + str(self.model_parameters.times_in - 1) + ' preceeding days')
        self.output_text_1.append('Input Variables (Daily): ')
        for iv in self.used_inputs:
            self.output_text_1.append('    ' + iv)
            
        self.output_text_1.append('')
        self.output_text_1.append('')

        self.temp_section_1_file = FPDF()
        self.temp_section_1_file.add_page()
        self.temp_section_1_file.set_font('Courier', size=9)
        
        for line in self.output_text_1:
            self.temp_section_1_file.cell(0, 5, txt=line, ln=1, align='L')
        
        self.temp_section_1_file.output(self.temp_section_1_path)

        ## Input Graphs 
        self.temp_input_graph_file = PdfPages(self.temp_input_graph_path)
        
        if self.num_inputs == 1:
            plt.figure(figsize=(8.27, 11.69))
            if model_parameters.input_variables[0] == 'daily_delta_swl':
                plt.plot(bore.dates[1:], bore.data_dict[model_parameters.input_variables[0]])
            else:
                plt.plot(bore.dates, bore.data_dict[model_parameters.input_variables[0]])
            plt.title(self.inputs_dict[model_parameters.input_variables[0]], fontsize=11)
            plt.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
        elif self.num_inputs <= 10:
            figure, axis = plt.subplots(self.num_inputs, 1, figsize=(8.27, 11.69))
            for i in range(self.num_inputs):
                if model_parameters.input_variables[i] == 'daily_delta_swl':
                    axis[i].plot(bore.dates[1:], bore.data_dict[model_parameters.input_variables[i]])
                else:
                    axis[i].plot(bore.dates, bore.data_dict[model_parameters.input_variables[i]])
                axis[i].set_title(self.inputs_dict[model_parameters.input_variables[i]], fontsize=11)
            figure.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
        else:
            num_per_page = int(np.ceil(self.num_inputs/2))
            page_1 = num_per_page
            figure, axis = plt.subplots(page_1, 1, figsize=(8.27, 11.69))
            for i in range(num_per_page):
                if model_parameters.input_variables[i] == 'daily_delta_swl':
                    axis[i].plot(bore.dates[1:], bore.data_dict[model_parameters.input_variables[i]])
                else:
                    axis[i].plot(bore.dates, bore.data_dict[model_parameters.input_variables[i]])
                axis[i].set_title(self.inputs_dict[model_parameters.input_variables[i]], fontsize=11)
            figure.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
            
            page_2 = int(self.num_inputs - num_per_page)
            figure, axis = plt.subplots(page_2, 1, figsize=(8.27, 11.69))
            for i in range(num_per_page, self.num_inputs):
                if model_parameters.input_variables[i] == 'daily_delta_swl':
                    axis[i-num_per_page].plot(bore.dates[1:], bore.data_dict[model_parameters.input_variables[i]])
                else:
                    axis[i-num_per_page].plot(bore.dates, bore.data_dict[model_parameters.input_variables[i]])
                axis[i-num_per_page].set_title(self.inputs_dict[model_parameters.input_variables[i]], fontsize=11)
            figure.tight_layout()
            plt.savefig(self.temp_input_graph_file, format='pdf')
            plt.close()
        self.temp_input_graph_file.close()
        

        ## Section 1 (Keras)
        
        self.output_text_2 = []
        
        self.output_text_2.append('<><> TensorFlow Keras LSTM Model <><>')
        self.output_text_2.append('')
        self.output_text_2.append('Optimiser: ' + str(self.keras_model.optimiser))
        self.output_text_2.append('Loss: ' + str(self.keras_model.loss))
        # self.output_text_2.append('Metric: ' + str(self.keras_model.metric))
        self.output_text_2.append('Number of Epochs: '+ str(self.keras_model.epochs))
        self.output_text_2.append('Percentage of Training Data for Validation: ' + str(self.keras_model.val_split * 100) + '%')
        if self.keras_model.shuffle_order == True:
            self.output_text_2.append('Time Series Order: Shufffled')
        else:
            self.output_text_2.append('Time Series Order: Chronological')
        if self.keras_model.verbose == 0:
            self.output_text_2.append('Verbose: Off')
        else:
            self.output_text_2.append('Verbose: On')
            
        self.output_text_2.append('')
        self.output_text_2.append('')
        
        self.output_text_2.append('<><> Model Architecture <><>')
        self.output_text_2.append('')
        self.output_text_2.append('Input Shape (Samples, Timesteps, Features): ' + str(self.keras_model.full_input_shape))
        
        self.output_text_2.append('LSTM Cells: ' + str(self.keras_model.lstm_cells))
        self.output_text_2.append('Fully Connected Hidden Neurons: ' + str(self.keras_model.fc_neurons))
        self.output_text_2.append('Fully Connected Output Neurons: ' + str(self.keras_model.num_outputs))
        self.output_text_2.append('LSTM Dropout Rate: ' + str(self.keras_model.lstm_dropout * 100) + '%')
        self.output_text_2.append('LSTM Recurrent Dropout Rate: ' + str(self.keras_model.lstm_recurrent_dropout * 100) + '%')
        self.output_text_2.append('')
        
        summary_string = []
        self.keras_model.model.summary(print_fn=lambda x: summary_string.append(x))
        for line in summary_string:
            self.output_text_2.append(line)
   
        self.output_text_2.append('')
        self.output_text_2.append('')
        if self.keras_model.val_split > 0:
            self.output_text_2.append('<><> Training and Validation Loss <><>')
        else:
            self.output_text_2.append('<><> Training Loss <><>')
        self.output_text_2.append('')
        self.output_text_2.append('Training Loss:')
        
        for i in range(len(self.keras_model.history.history['loss'])):
            if (i+1)%(len(self.keras_model.history.history['loss'])//10) == 0:
                self.output_text_2.append('    Epoch: ' + str(i+1) + ', Loss: ' + str(self.keras_model.history.history['loss'][i]))    
        
        if self.keras_model.val_split > 0:
            self.output_text_2.append('')
            self.output_text_2.append('Validation Loss:')
            for i in range(len(self.keras_model.history.history['val_loss'])):
                if (i+1)%(len(self.keras_model.history.history['val_loss'])//10) == 0:
                    self.output_text_2.append('    Epoch: ' + str(i+1) + ', Loss: ' + str(self.keras_model.history.history['val_loss'][i]))    
            
        self.output_text_2.append('')
        self.output_text_2.append('')

        self.output_text_2.append('<><> Results <><>')
        self.output_text_2.append('')

        self.output_text_2.append('Train Scores')
        self.output_text_2.append('')
        print_scores(self.keras_model.train_scores_dict, self.output_text_2)
        
        self.output_text_2.append('')
        self.output_text_2.append('Test Scores')
        self.output_text_2.append('')
        print_scores(self.keras_model.test_scores_dict, self.output_text_2)    

        self.temp_section_2_file = FPDF()
        self.temp_section_2_file.add_page()
        self.temp_section_2_file.set_font('Courier', size=9)
        
        for line in self.output_text_2:
            self.temp_section_2_file.cell(0, 5, txt=line, ln=1, align='L')
        
        self.temp_section_2_file.output(self.temp_section_2_path)

        ## Keras Graphs
        self.temp_keras_graph_file = PdfPages(self.temp_keras_graph_path)

        self.keras_model.plot_loss(self.temp_keras_graph_file)
        self.keras_model.plot_results(self.temp_keras_graph_file)
        self.temp_keras_graph_file.close()

        ## Section 3 (Torch)
        self.output_text_3 = []
        
        self.output_text_3.append('<><> PyTorchLSTM Model <><>')
        self.output_text_3.append('')

        self.output_text_3.append('Optimiser: ' + str(self.torch_model.opt_key))
        self.output_text_3.append('Loss: ' + str(self.torch_model.loss_key))
        self.output_text_3.append('Number of Epochs: ' + str(self.torch_model.num_epochs))
        self.output_text_3.append('Learning Rate: ' + str(self.torch_model.lr))

        if self.torch_model.shuffle == True:
            self.output_text_3.append('Time Series Order: Shufffled')
        else:
            self.output_text_3.append('Time Series Order: Chronological')
        if self.torch_model.verbose == 0:
            self.output_text_3.append('Verbose: Off')
        else:
            self.output_text_3.append('Verbose: On')
            
        self.output_text_3.append('')
        self.output_text_3.append('')
        
        self.output_text_3.append('<><> Model Architecture <><>')
        self.output_text_3.append('')
        self.output_text_3.append('Input Size (Features): ' + str(self.torch_model.input_size))
        self.output_text_3.append('Sequence Length (Timesteps): ' + str(self.torch_model.seq_length))
        
        self.output_text_3.append('LSTM Cells: ' + str(self.torch_model.lstm_cells))
        self.output_text_3.append('Stacked LSTM Layers: ' + str(self.torch_model.num_layers))
        self.output_text_3.append('Fully Connected Hidden Neurons: ' + str(self.torch_model.hidden_size))
        self.output_text_3.append('Fully Connected Output Neurons: ' + str(self.torch_model.num_classes))
        self.output_text_3.append('LSTM Dropout Rate: ' + str(self.torch_model.dropout * 100) + '%')
        self.output_text_3.append('')
        
        self.output_text_3.append('Network Structure:')
        flag = 0
        new_line = '    '
        for i in str(self.torch_model):
            if i == str('\n') and flag == 1:
                flag += 1
            if flag == 1:
                new_line += i
            if i == str('\n') and flag == 0:
                flag = 1
            if flag == 2:
                self.output_text_3.append(new_line)
                new_line = '    '
                flag += -1
        
        self.output_text_3.append('')
        self.output_text_3.append('')
        self.output_text_3.append('<><> Training Loss <><>')
        self.output_text_3.append('')
        self.output_text_3.append('Training Loss:')
        
        for i in range(len(self.torch_model.history)):
            if (i+1)%(len(self.torch_model.history)//10) == 0:
                self.output_text_3.append('    Epoch: ' + str(self.torch_model.history[i][0]+1) + ', Loss: ' + str(self.torch_model.history[i][1]))
        
        self.output_text_3.append('')
        self.output_text_3.append('')

        self.output_text_3.append('<><> Results <><>')
        self.output_text_3.append('')

        self.output_text_3.append('Train Scores')
        self.output_text_3.append('')
        print_scores(self.torch_model.train_scores_dict, self.output_text_3)
        
        self.output_text_3.append('')
        self.output_text_3.append('Test Scores')
        self.output_text_3.append('')
        print_scores(self.torch_model.test_scores_dict, self.output_text_3)    

        self.temp_section_3_file = FPDF()
        self.temp_section_3_file.add_page()
        self.temp_section_3_file.set_font('Courier', size=9)
        
        for line in self.output_text_3:
            self.temp_section_3_file.cell(0, 5, txt=line, ln=1, align='L')
        
        self.temp_section_3_file.output(self.temp_section_3_path)

        
        ## Torch Graphs
        self.temp_torch_graph_file = PdfPages(self.temp_torch_graph_path)
    
        self.torch_model.plot_loss(self.temp_torch_graph_file)
        self.torch_model.plot_results(self.temp_torch_graph_file)
        self.temp_torch_graph_file.close()        

        ## Section 4 (SVR)
        
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
        if self.torch_model.verbose == 0:
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
        
        pdfs = [self.temp_section_1_path, self.temp_input_graph_path, 
                self.temp_section_2_path, self.temp_keras_graph_path, 
                self.temp_section_3_path, self.temp_torch_graph_path,
                self.temp_section_4_path, self.temp_sklearn_graph_path]
        
        merger = PyPDF2.PdfMerger()
        for pdf in pdfs:
            merger.append(pdf)
        merger.write(self.log_file_path)
        merger.close()
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:42:59 2022

@author: johnsalvaris
"""

import import_functions
import bore_class
import model_parameters_class
import score_functions
import keras_lstm_class
import sklearn_svr_class
import output_log_class

import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
from time import time
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE


# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.enable_v2_behavior()

# %%

# CODE NOT AS FUNCTION
bore_id = 'GW075405.1.1' #'GW036872.1.1', 'GW075025.1.1', 'GW075405.1.1', 'GW080079.1.1' 'GW080415.1.1', 'GW080980.1.1', 'GW081101.1.2', 'GW273314.1.1', 'GW403746.3.3'

gwl_input = 'standard' # 'standard', 'delta', 'none', 'average', 'average_delta'
gwl_output = 'standard' # 'standard', 'delta', 'average', 'average_delta'

silo_variables = ['daily_rain', 'max_temp', 'min_temp', 'vp', 'vp_deficit',
                  'evap_pan', 'evap_syn', 'evap_comb', 'evap_morton_lake',
                  'radiation', 'rh_tmax', 'rh_tmin', 'et_short_crop', 
                  'et_tall_crop', 'et_morton_actual', 'et_morton_potential',
                  'et_morton_wet','mslp',
                  # Averages below here
                  'av_daily_rain', 'av_max_temp', 'av_min_temp', 'av_vp', 'av_vp_deficit',
                  'av_evap_pan', 'av_evap_syn', 'av_evap_comb', 'av_evap_morton_lake',
                  'av_radiation', 'av_rh_tmax', 'av_rh_tmin', 'av_et_short_crop', 
                  'av_et_tall_crop', 'av_et_morton_actual', 'av_et_morton_potential',
                  'av_et_morton_wet','av_mslp']

# silo_variables = ['evap_comb', 'radiation', 'et_morton_actual','et_morton_wet']

silo_variables = ['max_temp', 'daily_rain', 'evap_pan']
# silo_variables = ['daily_rain']

 
start_date_time = dt.datetime.now()
print("Code started at: " + str(start_date_time))
start_time = time()


times_in = 30 # time lags 
av_period = 1 # days to average
times_out = 1 # multiple outputs?
out_after = 30 # time leads --> Periods ahead
between_outs = 1 # space between mutliple outputs

kernel = 'rbf' # 'linear', 'rbf', 'sigmoid', 'precomputed'
gamma = 'scale'
epsilon = 0.1
tolerance = 0.00001
reg_parameter = 1.0
degree = 3 # For polynomial kernel --> Ignored otherwise
coef0 = 0.0 # only for ploy and sigmoid
svr_shrink = True

scaler_type = 'mm' # 'mm', 'ss' 
test_size = 0.2
shuffle_order = False

optimiser ='adam'
loss ='mae' # 'mse', 'mae'
metric = 'accuracy' #RMSE() # 'accuracy'
epochs = 100 # 200
keras_validation_split = 0.2
verbose = 0
learning_rate = 0.001 

num_outputs = times_out
num_fc_neurons = 32
lstm_dropout_rate = 0.2
lstm_recurrent_dropout_rate = 0.2
stacked_lstm = 1

#store raw data about the bore
bore = bore_class.bore(bore_id) 

#class for things used by the model
model_parameters = model_parameters_class.model_parameters(silo_variables, gwl_input, gwl_output)
model_parameters.add_in_out_params(epochs, times_in, out_after, times_out, between_outs, av_period)

#import data
bore.add_dfs() #adds gwl_df, bore_df, swl_df

#define the bore + silo coordinates
bore.add_location() 

#remove unused dates from the imported data
bore.remove_null_dates()


"""
NEED TO HANDLE MISSING DAYS --> possibly add back in dates
"""

# add reference (unscaled) bore data to bore for reference 
bore.add_silo_data()
bore.add_gwl_data(out_after)
bore.average_data(av_period)
bore.add_data_dict()

# bore.add_change_swl() # add change in swl and shorten all variables



# format data for training
model_parameters.add_average_data(bore)
model_parameters.scale_data(scaler_type)
model_parameters.format_inputs_outputs()
model_parameters.divide_data(test_size, shuffle_order)

# Dependent Variables
input_shape = model_parameters.X_train.shape
num_lstm_cells = times_in * len(model_parameters.input_variables) #64

num_lstm_cells = 64

## Keras Model
keras_model = keras_lstm_class.keras_LSTM(optimiser=optimiser, loss=loss, metric=metric, epochs=epochs, keras_validation_split=keras_validation_split, verbose=verbose)
keras_model.add_sets(model_parameters)
keras_model.create_network(input_shape, num_outputs, num_lstm_cells, num_fc_neurons, lstm_dropout_rate)
print("LSTM training begun at: " + str(dt.datetime.now()))
keras_model.train_model(model_parameters.X_train, model_parameters.y_train)
print("LSTM training complete: " + str(dt.datetime.now()))

keras_model.predict(model_parameters.X_test, scaler=model_parameters.output_scaler, dataset='test')
keras_model.predict(model_parameters.X_train, scaler=model_parameters.output_scaler, dataset='train')
keras_model.scores()



## SVR Model
model_parameters.format_for_sklearn_SVR()
sklearn_model = sklearn_svr_class.sklearn_SVR(kernel=kernel, gamma=gamma, epsilon=epsilon, C=reg_parameter, tolerance=tolerance, degree=degree, coef0=coef0, shrinking=svr_shrink, verbose=verbose, shuffle_order=shuffle_order)
sklearn_model.add_sets(model_parameters)
sklearn_model.create_model()
print("SVR training begun at: " + str(dt.datetime.now()))
sklearn_model.train_model(model_parameters.X_train_svr, model_parameters.y_train_svr)
print("SVR training complete: " + str(dt.datetime.now()))

sklearn_model.predict(model_parameters.X_test_svr, scaler=model_parameters.output_scaler, dataset='test')
sklearn_model.predict(model_parameters.X_train_svr, scaler=model_parameters.output_scaler, dataset='train')
sklearn_model.scores()

end_time = time()

output_log = output_log_class.output_log(start_date_time, start_time, end_time, bore, model_parameters, keras_model, sklearn_model)

# %%

# import shap
# import matplotlib.pyplot as plt

# shap.initjs()
# # explainer = shap.KernelExplainer(model=keras_model.model.predict, data=keras_model.X_test[:50], link="identity")


# # explainer = shap.DeepExplainer(keras_model.model, model_parameters.X_train[:10,0,:])
# # explainer = shap.DeepExplainer((keras_model.model.layers[0].input, keras_model.model.layers[-1].output), model_parameters.X_train[:20])
# # shap_values = explainer.shap_values(model_parameters.X_test[:20])

# # explainer = shap.DeepExplainer(keras_model.model, model_parameters.X_train) #, keras.backend.get_session()
# # shap_values = explainer.shap_values(model_parameters.X_test)


# # explainer = shap.DeepExplainer(torch_model, model_parameters.X_train_tensor[:10])
# # shap_values = explainer.shap_values(model_parameters.X_test_tensor)

# # explainer = shap.KernelExplainer(sklearn_model.model.predict, model_parameters.X_train_svr[:10], link='identity')
# # shap_values = explainer.shap_values(model_parameters.X_test_svr[:10])


# # shap_model = keras_model.model
# # shap_fit_data = model_parameters.X_train
# # shap_plot_data = model_parameters.X_test[:100]


# test_for = 'keras' # 'keras', 'torch', 'sklearn'
# test_number = 100

# if test_for == 'keras':
#     shap_model = keras_model.model
#     shap_fit_data = model_parameters.X_train
#     shap_plot_data = model_parameters.X_test[:test_number]
    
#     explainer = shap.DeepExplainer(shap_model, shap_fit_data)
#     o_shap_values = explainer.shap_values(shap_plot_data)
    
#     shap_values = np.asarray(o_shap_values[0]) # originally is a list

# elif test_for == 'torch':
#     shap_model = torch_model
#     shap_fit_data = model_parameters.X_train_tensor
#     shap_plot_data = model_parameters.X_test_tensor[:test_number]
    
#     explainer = shap.DeepExplainer(shap_model, shap_fit_data)
#     shap_values = explainer.shap_values(shap_plot_data)



# reshaped_shap_values = shap_values.reshape(-1, shap_values.shape[-1])
# reshaped_shap_plot_data = shap_plot_data.reshape(-1, shap_plot_data.shape[-1])

# plt.title(test_for)
# shap.summary_plot(reshaped_shap_values, reshaped_shap_plot_data,feature_names=model_parameters.input_variables) # FEATURE NAMES MAY NOT BE ASSIGNING CORRECTLY!!! COMPARE WITH "shap.summary_plot(reshaped_shap_values)"


# shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[1,0], feature_names=model_parameters.input_variables)


# shap.summary_plot(reshaped_shap_values, reshaped_shap_plot_data,feature_names=model_parameters.input_variables) 

# i=3

# exp = shap.Explanation(reshaped_shap_values[i], data=reshaped_shap_plot_data[i], base_values = explainer.expected_value[0], feature_names=model_parameters.input_variables)
# shap.waterfall_plot(exp)
# shap.plots.heatmap(exp)


###########
###########
# shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[1,0], feature_names=model_parameters.input_variables)

# shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0], shap_values[1,0], feature_names=model_parameters.input_variables)
###########
###########





# df23 = pd.DataFrame(shap_plot_data[0][0]).T
# df23.columns=model_parameters.input_variables

# shap.force_plot(explainer.expected_value[0], shap_values[0][0], df23
# shap.Explanation(reshaped_shap_values, data=reshaped_shap_plot_data[0], feature_names=model_parameters.input_variables[0])

# %%

# from sklearn.model_selection import GridSearchCV
# from scikeras.wrappers import KerasClassifier

# grid_model = KerasClassifier(model=keras_model.model, verbose=0)

# # fc_neuron_grid = [i for i in range(1,16,2)]
# # num_fc_neurons
# epochs_grid = [50,100,150,200,250,300]
# param_grid = dict(epochs=epochs_grid) #, model__neurons=fc_neuron_grid)
# grid = GridSearchCV(estimator=grid_model, param_grid=param_grid, n_jobs=-1, cv=3)
# grid_result = grid.fit(model_parameters.X_train, model_parameters.y_train, error_score='raise')
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# %% Correlation plot testing

# import seaborn as sns
# import matplotlib.pyplot as plt

# change = []
# for i in range(1,len(bore.swl)):
#     change.append(bore.swl[i]-bore.swl[i-1])
# change_swl = np.array(change)

# change_corr_df = pd.DataFrame({'delta_swl': change_swl})

# for i in range(len(list(bore.data_dict.keys()))):
#     df = pd.DataFrame({list(bore.data_dict.keys())[i]:list(bore.data_dict.values())[i][1:]})
#     change_corr_df = pd.concat([change_corr_df, df], axis=1)

# corr = change_corr_df.corr()
# sns.heatmap(corr)
# plt.show()




# %%%

"""
TO DO

separate out validation data + make plots in output log
handle missing days
pacf/acf
shap/sensitivity analysis
make work for >1 output
how to use nse
all other green comments
"""


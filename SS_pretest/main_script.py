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
import output_summary_class

import numpy as np
import pandas as pd
import datetime as dt
import tensorflow as tf
from time import time
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE


# tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.enable_v2_behavior()

# %%
print(f"Pre-processing started at: \t {dt.datetime.now()}")
start_date_time = dt.datetime.now()
start_time = time()

bore_id = 'GW036872.1.1' #'GW036872.1.1', 'GW075025.1.1', 'GW075405.1.1', 'GW080079.1.1' 'GW080415.1.1', 'GW080980.1.1', 'GW081101.1.2', 'GW273314.1.1', 'GW403746.3.3'

gwl_input = 'standard' # 'standard', 'delta', 'none', 'average', 'average_delta'
gwl_output = 'standard' # 'standard', 'delta', 'average', 'average_delta'

other_variables = ['daily_rain', 'max_temp', 'min_temp', 'vp', 'vp_deficit',
                  'evap_pan', 'evap_syn', 'evap_comb', 'evap_morton_lake',
                  'radiation', 'rh_tmax', 'rh_tmin', 'et_short_crop', 
                  'et_tall_crop', 'et_morton_actual', 'et_morton_potential',
                  'et_morton_wet','mslp', 'sm_pct', 's0_pct', 'ss_pct', 'sd_pct', 'dd',
                  # Averages below here
                  'av_daily_rain', 'av_max_temp', 'av_min_temp', 'av_vp', 'av_vp_deficit',
                  'av_evap_pan', 'av_evap_syn', 'av_evap_comb', 'av_evap_morton_lake',
                  'av_radiation', 'av_rh_tmax', 'av_rh_tmin', 'av_et_short_crop', 
                  'av_et_tall_crop', 'av_et_morton_actual', 'av_et_morton_potential',
                  'av_et_morton_wet', 'av_mslp', 'av_sm_pct', 'av_s0_pct', 'av_ss_pct', 'av_sd_pct', 'av_dd'] #sm = root zone, s0 = upper, ss = lower, sd = deep layer, dd = deep drainage

# other_variables = ['evap_comb', 'radiation', 'et_morton_actual','et_morton_wet']

other_variables = ['daily_rain', 'max_temp', 'min_temp', 'vp', 'vp_deficit',
                  'evap_pan', 'evap_syn', 'evap_comb', 'evap_morton_lake',
                  'radiation', 'rh_tmax', 'rh_tmin', 'et_short_crop', 
                  'et_tall_crop', 'et_morton_actual', 'et_morton_potential',
                  'et_morton_wet','mslp', 'sm_pct', 's0_pct', 'ss_pct', 'sd_pct', 'dd']

# other_variables = ['av_daily_rain', 'av_max_temp', 'av_min_temp', 'av_vp', 'av_vp_deficit',
#                     'av_evap_pan', 'av_evap_syn', 'av_evap_comb', 'av_evap_morton_lake',
#                     'av_radiation', 'av_rh_tmax', 'av_rh_tmin', 'av_et_short_crop', 
#                     'av_et_tall_crop', 'av_et_morton_actual', 'av_et_morton_potential',
#                     'av_et_morton_wet', 'av_mslp', 'av_sm_pct', 'av_s0_pct', 'av_ss_pct', 'av_sd_pct', 'av_dd']



times_in = 2 # time lags 
av_period = 30 # days to average
out_after = 1 # time leads --> Periods ahead

kernel = 'rbf' # 'linear', 'rbf', 'sigmoid', 'precomputed'
gamma = 'scale'
epsilon = 0.1
tolerance = 0.00001
reg_parameter = 1.0
degree = 3 # For polynomial kernel --> Ignored otherwise
coef0 = 0.0 # only for ploy and sigmoid
svr_shrink = True
svr_scoring_code = 'neg_mean_squared_error' # 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2', 'max_error'

scaler_type = 'mm' # 'mm', 'ss' 
test_size = 0.2
shuffle_order = False

optimiser ='adam'
loss ='mae' # 'mse', 'mae'
metric = 'accuracy' #RMSE() # 'accuracy'
epochs = 100 # 200
keras_validation_split = 0.2#0.2
verbose = 0
learning_rate = 0.001 

num_fc_neurons = 32
lstm_dropout_rate = 0.2
lstm_recurrent_dropout_rate = 0.2
stacked_lstm = 1


interpolation_method = 'Spline'

#store raw data about the bore
bore = bore_class.bore(bore_id) 

#class for things used by the model
model_parameters = model_parameters_class.model_parameters(epochs, times_in, out_after, other_variables, av_period, gwl_input, gwl_output)

#import data
bore.add_dfs() #adds gwl_df, bore_df, swl_df, awo_df
bore.handle_missing_dates(interpolation_method)

#remove unused dates from the imported data
bore.remove_null_dates()

# add reference (unscaled) bore data to bore for reference 
bore.add_silo_data()
bore.add_awo_data()
bore.add_gwl_data(out_after)
bore.average_data(av_period)
bore.add_data_dict()

# format data for training
model_parameters.add_data(bore)


#%%%
model_parameters.scale_data(scaler_type)
model_parameters.format_inputs_outputs()
model_parameters.divide_data(test_size, shuffle_order)

# Dependent Variables
input_shape = model_parameters.X_train.shape
num_lstm_cells = times_in * len(model_parameters.input_variables) #64

num_lstm_cells = 64
print(f"Pre-processing completed at: \t {dt.datetime.now()}")

## Keras Model
keras_model = keras_lstm_class.keras_LSTM(optimiser=optimiser, loss=loss, metric=metric, epochs=epochs, keras_validation_split=keras_validation_split, verbose=verbose)
keras_model.add_sets(model_parameters)
keras_model.create_network(input_shape, num_lstm_cells, num_fc_neurons, lstm_dropout_rate)
print(f"LSTM training started at: \t {dt.datetime.now()}")
keras_model.train_model(model_parameters.X_train, model_parameters.y_train)
print(f"LSTM training completed at: \t {dt.datetime.now()}")

keras_model.predict(model_parameters.X_test, scaler=model_parameters.output_scaler, dataset='test')
keras_model.predict(model_parameters.X_train, scaler=model_parameters.output_scaler, dataset='train')
score_functions.calculate_scores(keras_model)

## SVR Model
model_parameters.format_for_sklearn_SVR()
sklearn_model = sklearn_svr_class.sklearn_SVR(kernel=kernel, gamma=gamma, epsilon=epsilon, C=reg_parameter, tolerance=tolerance, degree=degree, coef0=coef0, shrinking=svr_shrink, verbose=verbose, shuffle_order=shuffle_order, epochs=epochs)
sklearn_model.add_sets(model_parameters)
sklearn_model.create_model()
print(f"SVR training started at:   \t {dt.datetime.now()}")
sklearn_model.train_model(model_parameters.X_train_svr, model_parameters.y_train_svr, svr_scoring_code)
print(f"SVR training completed at: \t {dt.datetime.now()}")

sklearn_model.predict(model_parameters.X_test_svr, scaler=model_parameters.output_scaler, dataset='test')
sklearn_model.predict(model_parameters.X_train_svr, scaler=model_parameters.output_scaler, dataset='train')
score_functions.calculate_scores(sklearn_model)

end_time = time()

print(f"Output log started at:    \t {dt.datetime.now()}")
output_summary = output_summary_class.output_summary(start_date_time, start_time, end_time, bore, model_parameters, keras_model, sklearn_model)
output_summary.create_general_text()
output_summary.create_input_variable_graphs()
output_summary.create_lstm_text()
output_summary.create_lstm_learning_graphs()
output_summary.create_result_graphs(keras_model)
output_summary.create_svr_text()
output_summary.create_svr_learning_graphs()
output_summary.create_result_graphs(sklearn_model)
output_summary.create_log()
output_summary.create_spreadsheet()
output_summary.save_models()
print(f"Output log completed at:  \t {dt.datetime.now()}")

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

# %%%

"""
TO DO

separate out validation data + make plots in output log
shap/sensitivity analysis
how to use nse
all other green comments
"""


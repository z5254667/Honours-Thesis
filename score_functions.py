#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 10:44:53 2022

@author: johnsalvaris
"""

import numpy as np

# Root Mean Squared Error
def rmse(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])
    
    return np.sqrt(np.sum((actual - predicted)**2)/actual.shape[0])

# Mean Squared Error
def mse(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])
    
    return np.sum((actual - predicted)**2)/actual.shape[0]

# Coefficient of Determination
def r_squared(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])
    
    rss = np.sum((actual - predicted)**2)
    actual_mean = np.mean(actual)
    tss = np.sum((actual - actual_mean)**2)
    return 1 - rss/tss

# Mean Absolue Percentage Error --> Note calculates decimal value
def mape(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])
    
    return np.sum(np.abs((actual - predicted)/actual))/actual.shape[0]

# Mean Absolute Error
def mae(actual, predicted):
    actual = actual.reshape(actual.shape[0])
    predicted = predicted.reshape(predicted.shape[0])
    
    return np.sum(np.abs(predicted - actual))/actual.shape[0]

# Unsure how Nash-Sutcliffe Efficiency differes from Coefficient of Determination
def nse(actual, predicted):
    pass
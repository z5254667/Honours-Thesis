#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:22:57 2022

@author: johnsalvaris
"""

from main_script import main_script
import datetime as dt
import shutil
import os
import pandas as pd
import matplotlib.pyplot as plt
import import_functions
from fpdf import FPDF
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import PyPDF2
    
other_variables = ['av_min_temp', 'av_vp', 'av_vp_deficit', 'av_evap_pan', 'av_evap_syn', 'av_evap_comb', 'av_evap_morton_lake', 'av_radiation', 'av_rh_tmax', 'av_rh_tmin', 'av_et_short_crop', 'av_et_tall_crop', 'av_et_morton_actual', 'av_et_morton_potential', 'av_et_morton_wet', 'av_mslp', 'av_sm_pct', 'av_s0_pct', 'av_ss_pct', 'av_sd_pct', 'av_dd']
#'av_daily_rain', 'av_max_temp',

for o in other_variables:
    # Initialise runs in test
    test_runs = 10
    run = 1
    times_list = [] # Stores unique time identifier of each run
    
    while run <= test_runs: # Choose number of runs
        try:
            print('')
            print(f'Run {run}')
            print('')
            current_run = main_script()
            current_run.run([o])
        except TimeoutError: # Restart run if times out
            print('')
            print(f'TimeoutError - Retry Run {run}: {dt.datetime.now()}')
            print('')
        except KeyboardInterrupt: # Breaks loop manually 
            print('')
            print(f'KeyboardInterrupt - Exit code: {dt.datetime.now()}')
            print('')
            break
        except: # Restart run if other error
            print('')
            print(f'Error - Retry Run {run}: {dt.datetime.now()}')
            print('')
        else:
            times_list.append(current_run.start_date_time.strftime("%Y_%m_%d__%H_%M_%S"))
            run += 1
    
    # Path to overall file
    bore_path = os.getcwd() + f'/{current_run.bore_id}'
    
    # Create working folder number 
    n = 1
    folder_needed = True
    while folder_needed:
        try:
            os.mkdir(f'{bore_path}/multi_tests/test_{n}')
        except FileExistsError:
            n += 1
        else:
            test_path = f'{bore_path}/multi_tests/test_{n}'
            folder_needed = False
    
    # Initialise dictionaries for scores
    lstm_train_dict = {}
    lstm_test_dict = {}
    svr_train_dict = {}
    svr_test_dict = {}
    
    for i in range(len(times_list)):
        t = times_list[i]
        # Copy outputs of each run to the test folder
        shutil.copyfile(f'{bore_path}/logs/output_log_{t}.pdf', f'{test_path}/run_{i + 1}_test_{n}_output_log_{t}.pdf')
        shutil.copyfile(f'{bore_path}/spreadsheets/output_spreadsheet_{t}.xlsx', f'{test_path}/run_{i + 1}_test_{n}_output_spreadsheet_{t}.xlsx')
        shutil.copyfile(f'{bore_path}/models/svr_model_{t}.pkl', f'{test_path}/run_{i + 1}_test_{n}_svr_model_{t}.pkl')
        shutil.copytree(f'{bore_path}/models/lstm_model_{t}', f'{test_path}/run_{i + 1}_test_{n}_lstm_model_{t}')
        
        # Add scores from the run to the dictionary
        lstm_scores = pd.read_excel(f'{test_path}/run_{i + 1}_test_{n}_output_spreadsheet_{t}.xlsx', sheet_name='lstm_scores')
        svr_scores = pd.read_excel(f'{test_path}/run_{i + 1}_test_{n}_output_spreadsheet_{t}.xlsx', sheet_name='svr_scores')
        
        if len(lstm_train_dict) == 0:
            for i in range(len(lstm_scores['score'])):
                lstm_train_dict[lstm_scores['score'][i]] = []
                lstm_test_dict[lstm_scores['score'][i]] = []
                svr_train_dict[svr_scores['score'][i]] = []
                svr_test_dict[svr_scores['score'][i]] = []            
        for i in range(len(lstm_scores['score'])):
            lstm_train_dict[lstm_scores['score'][i]].append(lstm_scores['train'][i])
            lstm_test_dict[lstm_scores['score'][i]].append(lstm_scores['test'][i])
            svr_train_dict[svr_scores['score'][i]].append(svr_scores['train'][i])
            svr_test_dict[svr_scores['score'][i]].append(svr_scores['test'][i])
    
    # Create temporary pdf to show average scores
    summary_scores_text_path = import_functions.path(f'/{current_run.bore_id}/temp/', 'score_summary_text.pdf')
    summary_scores_text = FPDF()
    summary_scores_text.add_page()
    summary_scores_text.set_font('Courier', size=8)    
    
    # Create temporary pdf to show general test information
    summary_general_text_path = import_functions.path(f'/{current_run.bore_id}/temp/', 'summary_general_text.pdf')
    summary_general_text = FPDF()
    summary_general_text.add_page()
    summary_general_text.set_font('Courier', size=8)    
    
    # Create general pdf to show boxplots
    summary_scores_graph_path = import_functions.path(f'/{current_run.bore_id}/temp/', 'score_summary_graph.pdf')
    summary_scores_graph = PdfPages(summary_scores_graph_path)
    
    # Create LSTM Train boxplot
    plt.clf()
    lstm_train = pd.DataFrame(lstm_train_dict)
    lstm_train.boxplot(figsize=(8.27, 11.69), showmeans=True, rot=90)
    plt.title('LSTM Train Scores')
    plt.tight_layout()
    plt.savefig(summary_scores_graph, format='pdf')
    plt.close()
    # plt.show()
    
    # Create LSTM Test boxplot
    plt.clf()
    lstm_test = pd.DataFrame(lstm_test_dict)
    lstm_test.boxplot(figsize=(8.27, 11.69), showmeans=True, rot=90)
    plt.title('LSTM Test Scores')
    plt.tight_layout()
    plt.savefig(summary_scores_graph, format='pdf')
    plt.close()
    # plt.show()
    
    # Create SVR Train boxplot
    plt.clf()
    svr_train = pd.DataFrame(svr_train_dict)
    svr_train.boxplot(figsize=(8.27, 11.69), showmeans=True, rot=90)
    plt.title('SVR Train Scores')
    plt.tight_layout()
    plt.savefig(summary_scores_graph, format='pdf')
    plt.close()
    # plt.show()
    
    # Create SVR Test boxplot
    plt.clf()
    svr_test = pd.DataFrame(svr_test_dict)
    svr_test.boxplot(figsize=(8.27, 11.69), showmeans=True, rot=90)
    plt.title('SVR Test Scores')
    plt.tight_layout()
    plt.savefig(summary_scores_graph, format='pdf')
    plt.close()
    # plt.show()
    
    
    summary_scores_graph.close()
    
    # Text for scores summary pdf
    lines = []
    lines.append('<><> LSTM <><>')
    lines.append('')
    
    for i in lstm_train:
        char_len = len(i) # Max character length = 36 for alignment
        lines.append(f'Average Train {i}:{" " * (36 - char_len)} \t {np.mean(lstm_train[i])}') # Calculates average score
    lines.append('')
    
    for i in lstm_test:
        char_len = len(i) # Max character length = 36 for alignment
        lines.append(f'Average Test {i}:{" " * (36 - char_len)} \t {np.mean(lstm_test[i])}') # Calculates average score
    lines.append('')
    
    lines.append('<><> SVR <><>')
    lines.append('')
    
    for i in svr_train:
        char_len = len(i) # Max character length = 36 for alignment
        lines.append(f'Average Train {i}:{" " * (36 - char_len)} \t {np.mean(svr_train[i])}') # Calculates average score
    lines.append('')
    
    for i in svr_test:
        char_len = len(i) # Max character length = 36 for alignment
        lines.append(f'Average Test {i}:{" " * (36 - char_len)} \t {np.mean(svr_test[i])}') # Calculates average score
    
    for line in lines:
        summary_scores_text.cell(0, 5, txt=line, ln=1, align='L')
    summary_scores_text.output(summary_scores_text_path)
        
    
    # Text for general info pdf
    gen_df = pd.read_excel(f'{test_path}/run_{test_runs}_test_{n}_output_spreadsheet_{t}.xlsx')
    lines = []
    lines.append('<><> General Information <><>')
    lines.append('')
    lines.append(f'Test: {n}')
    for i in range(len(gen_df)):
        lines.append(f"{gen_df['Unnamed: 0'][i]}: {gen_df['Parameters'][i]}")
    lines.append('')
    lines.append('<><> Run Identifiers <><>')
    for i in range(len(times_list)):
        lines.append(f'Run {i + 1}: {times_list[i]}')
    
    for line in lines:
        summary_general_text.cell(0, 5, txt=line, ln=1, align='L')
    summary_general_text.output(summary_general_text_path)
    
    # Include Input Correlation Heatmap
    temp_input_graphs_path_4 = import_functions.path(f'/{current_run.bore_id}/temp/', 'temp_input_graphs_4.pdf') 
    
    # Combine into single pdf
    summary_file = f'{test_path}/test_{n}_summary.pdf'
            
    pdfs = [summary_general_text_path, summary_scores_text_path, summary_scores_graph_path, temp_input_graphs_path_4]
    
    merger = PyPDF2.PdfMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(summary_file)
    merger.close()
    
    with pd.ExcelWriter(f'{test_path}/test_{n}_all_scores.xlsx') as writer:
        lstm_train.to_excel(writer, sheet_name='lstm_train')
        lstm_test.to_excel(writer, sheet_name='lstm_test')
        svr_train.to_excel(writer, sheet_name='svr_train')
        svr_test.to_excel(writer, sheet_name='svr_test')



# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:13:59 2021

@author: Kanishka
"""
# =============================================================================

# Name - kanishka Goyal

# =============================================================================

# This is the main module and program execution starts from here. This project 
# classifies three types of datasets using two different machine learning 
# algorithms. The dataset and classifier is picked by the user via GUI. K-fold 
# cross validation and parameter selection for both the algorithms. 
# Confusion matrix and parameter analysis plot. 

# =============================================================================

# This module contains all the code for GUI creation. 
# Contains code for selecting radiobuttons for data set and 
# classifier. Contains a dropdown for selecting the value for k-fold. A run button
# which will run all the code for classificating and print the results. 

# =============================================================================

# Set directory to the same folder where file is stored 

# import os;
# os.getcwd()
# os.chdir("")

# =============================================================================

# Import libraries & modules
import tkinter as tk 
import classifier_code
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import numpy as np 

# Window base measurements 
window = tk.Tk() 
window.title('Programming for Data Science') 
# width x height + x_offset + y_offset: 
window.geometry("850x650+100+10") 

# Set background color
window.configure(bg='#ffcc99')

# Set font 
myfont = "Arial, 11" 
  
# Add a label - The top label for application 

lbl_header = tk.Label(window,text="Classification Algorithms", font= "Arial, 12", 
                      height=2, width = 100, anchor = "center", fg = "grey15", bg ="coral",
                      relief = 'groove') 
lbl_header.place(x=0, y=0)

# Dataset selection------------------------------------------------------------

lbl_dataset = tk.Label(window,text="Select a dataset :- ", anchor="w", relief = "raised",
                       font='Arial, 12', bg='coral', borderwidth = 3) 
lbl_dataset.place(x=10, y= 50)

# Adding variables for dataset selection

data_set = tk.StringVar(value = " ") # variable for dataset selection 

# Radio buttons for data sets to select between 

# 1. Dataset - Iris
radio_ds1 = tk.Radiobutton(window,text = "Iris", variable = data_set, anchor = "w",  
                           value = 'Iris',font = myfont, relief = 'ridge', width = 11,
                           activebackground = "coral", bg = "#ff9966")
radio_ds1.place(x = 10, y = 85) # position

# Dataset - Breast Cancer
radio_ds2 = tk.Radiobutton(window,text = "Breast Cancer", variable = data_set,anchor = "w",
                            value = 'Breast Cancer',font = myfont , relief = 'ridge', width = 11,
                            activebackground = "coral", bg = "#ff9966")
radio_ds2.place(x = 10, y = 110) # position

# Dataset - Wine
radio_ds3 = tk.Radiobutton(window,text = "Wine", variable = data_set,anchor = "w",
                            value = 'Wine', font = myfont, relief = 'ridge', width = 11,
                            activebackground = "coral", bg = "#ff9966")
radio_ds3.place(x = 10, y = 135) # position


# Classifier selection---------------------------------------------------------

# Label for classifier selection 
lbl_classifier = tk.Label(window,text="Select a classifier :- ", anchor="w", relief = "raised",
                       font='Arial, 12', bg='coral', borderwidth = 3) 
lbl_classifier.place(x=170, y=50) # position

# Adding variables for Classifier selection 

classifier_name = tk.StringVar(value = " ") # variable for classifier selection 

# Radio buttons for classifier

# Classifier - K Nearest Neighbor 
radio_cf1 = tk.Radiobutton(window, text = "K Nearest Neighbor", variable = classifier_name, 
                            value = 'KNN', font = myfont, relief = 'ridge', width = 15,
                            anchor = "w",activebackground = "coral", bg = "#ff9966")
radio_cf1.place(x = 170, y = 85) # position

# Classifier - Support Vector Machine
radio_cf2 = tk.Radiobutton(window, text = "Support Vector", variable = classifier_name, 
                            value = 'SVM', font = myfont, relief = 'ridge', width = 15,
                            anchor = "w",activebackground = "coral", bg = "#ff9966")
radio_cf2.place(x = 170, y = 110) # position

# K-fold value selection-------------------------------------------------------

# Label for K-fold (
lbl_classifier = tk.Label(window,text="Value for K(K-fold) :- ", anchor="w", relief = "raised",
                       font='Arial, 12', bg='coral', borderwidth = 3) 
lbl_classifier.place(x=10, y = 180) # position 

# Set the options for k-fold
kFold_options = [3,5,7,9]

# Variable for k-fold 
kFold_var = tk.IntVar() 

# Value selected as default
kFold_var.set(3)

# Dropdown for k-fold selection 
dropDown = tk.OptionMenu(window , kFold_var, *kFold_options)
dropDown.place(x = 170, y = 180) # position
dropDown.configure(relief = "raised", bg='#ff9966', borderwidth = 3)

#==============================================================================
#==============================================================================

# Print result

# This is a function for printing results on GUI 
# gscv is the modified classifier & parameter_range contains paramter 
# values with name

def print_result(gscv_classifier,parameter_range): 
    
    # store all the results into different lists
    means = gscv_classifier.cv_results_['mean_test_score'] 
    stds = gscv_classifier.cv_results_['std_test_score'] 
    results = gscv_classifier.cv_results_['params'] 
    
    # Pick up the parameter range & parameter name from 'parameter_range'
    param_name = list(parameter_range[0].keys())[0]
    param_values = list(parameter_range[0].values())[0]
    
    # Initiate a result string to store the result 
    myResult_list = ''
    
    # Loop through all three lists to build the whole result 
    for mean, std, param in zip(means, stds, results): 
        param_text = str(param)
        mean_text = str(round(mean,3))
        std_text = str(round(std*2,3))
        toAppend = "Parameter: " + param_text + " accuracy: " + mean_text + " (+/- " + std_text + ")" + "           "
        myResult_list = myResult_list + toAppend
        
    # Print the grid parameter results on GUI
    lbl_grid_result.config(text = myResult_list)
    
    # Print the best parameter results on GUI
    best_param_text = "Best parameter:" + str(gscv_classifier.best_params_)
    lbl_bestParam.config(text = best_param_text)
    
# PLOT ON GUI

    # the figure that will contain the plot
    fig = Figure(figsize = (5.2,4.5))
    
    # adding the subplot
    plot1 = fig.add_subplot(111)
    
# IF_else condition depending on the parameter name as to decide 
# between gamma and k

    if param_name == 'gamma':
        plot1.semilogx(param_values, means)
        plot1.semilogx(param_values, np.array(means) + np.array(stds), "b--")
        plot1.semilogx(param_values, np.array(means) - np.array(stds), "b--")
        plot1.set_xlabel("Parameter C", fontsize=12) # set x label
    else:
        plot1.plot(param_values, means)
        plot1.set_xlabel("Value of K for KNN", fontsize=12) # set x label 
    
    # Set title and y label 
    plot1.set_title ("Cross Validation Score w.r.t parameters", fontsize=14)
    plot1.set_ylabel("CV score", fontsize=12)
    
    # plotting the graph
    plot1.plot()
    
    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, window)  
    canvas.draw()
  
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    
    # Placing it on the Tkinter GUI 
    canvas.get_tk_widget().place(x = 10, y = 240)
    
#==============================================================================
#==============================================================================

# Execution Button-------------------------------------------------------------

# Button for execution and storing the result 
# Function call to 'set_all_variables' in module 'classifier_code'
def execute_code():
    gscv_classifier,parameter_range = classifier_code.set_all_variables(data_set.get(), 
                                      classifier_name.get(), kFold_var.get())
    
    # Fxn call to print results on GUI
    print_result(gscv_classifier,parameter_range)
    
    
# Button for execution by submitting all the user selected values 
run_btn = tk.Button(window,text = "Run Model", command = execute_code,
                fg = "black", font = "Verdana 13",
                bd = 2, bg = "coral", relief = "raised")

run_btn.place(x = 240, y = 180) # position 



# Output space ---------------------------------------------------------------

# Grid scores 

# Label
lbl_output1 = tk.Label(window, text ="Grid scores on validation set :- ",   anchor="w", relief = "raised",
                       font='Arial, 12', bg='coral', borderwidth = 3)
lbl_output1.place(x=400, y = 50)

# Output 
lbl_grid_result = tk.Label(window, text ="", fg ="navy", width = 45,
                      anchor="nw", height = 18, font = myfont, bg = "white",
                      wraplength = 450, justify = "left")
lbl_grid_result.place(x = 400, y = 80)

# Best Parameter 

# Label
lbl_output2 = tk.Label(window, text ="Best parameter :- ",   anchor="w", relief = "raised",
                       font='Arial, 12', bg='coral', borderwidth = 3)
lbl_output2.place(x=400, y = 420)

# Output 
lbl_bestParam = tk.Label(window, text ="", fg ="navy", width = 45,
                      anchor="w", height = 2, font = myfont, bg = "white",
                      wraplength = 460, justify = "left")
lbl_bestParam.place(x =400, y = 450)

# Confusion Matrix 

lbl_output3 = tk.Label(window, text ="View the Confusion Matrix and class prediction on IDE/Console",   anchor="w", relief = "raised",
                       font='Arial, 12', bg='coral', borderwidth = 3)
lbl_output3.place(x=400, y = 510)


window.mainloop()



#==============================================================================
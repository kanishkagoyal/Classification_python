# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 12:56:17 2021

@author: Kanishka
"""
###############################################################################

# This module contains all the code for classification. This module will be imported by
# via 'gui_app' module. Contains code for setting the variables as selected by 
# users and then split the dataset into 80% training and 20% test. Then based on
# classifier using gridsearchCV find the best parameter for our model. Then 
# fir using training data, test later and find accuracy. 

###############################################################################

# import Libraries/packages/modules

from sklearn import datasets, neighbors, metrics, mixture, svm 
from sklearn.model_selection import train_test_split, GridSearchCV 
import matplotlib.pyplot as plt 

# =============================================================================
# Function for setting all variables with inputs from users 

def set_all_variables(data_set,classifier_name,kCV):

# if-else for selecting dataset as per user requirement 
    if data_set == 'Iris' :
        dataset = datasets.load_iris() # load iris
    
    elif data_set == 'Breast Cancer' :
        dataset = datasets.load_breast_cancer() # load breast cancer
        
    elif data_set == 'Wine' :
        dataset = datasets.load_wine() # load wine
        
    else : # If user did not select anything
        print("No data set selected") 
        return 'No dataset selected'
        raise SystemExit
        
# if else for selecting classifier as per user requirement 
    if classifier_name == 'KNN' :
        parameter_range = [{'n_neighbors': list(range(1,18))}] # parameter options for KNN
        classifier = neighbors.KNeighborsClassifier() # KNN classifier
        
    elif classifier_name == 'SVM':
        parameter_range = [{'gamma': [0.00000001,0.0000001,0.000001, 0.00001,0.0001, 0.001,
                                      0.01, 0.1, 1.0]}] # parameter options for SVM
        classifier = svm.SVC() # Support Vector Machine 
        
    else: # If user did not select anything 
        print("No classifier selected")
        return 'No classifier selected'
        raise SystemExit
# return the results via function "run_classifier" and pass the set variables to it         
    return run_classifier(dataset, classifier, parameter_range, kCV)
    
# =============================================================================
 
# Function definintion 
# Function for classifying the dataset and printing results
def run_classifier(dataset, classifier, parameter_range, kCV):  
    
    X = dataset.data # select explanatory data
    y = dataset.target # select response data 
    class_names = dataset.target_names # pick class names 
    
    # split the data into test and train data 
    
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=0)
    
    # Modify the classifier for cross validation and best parameter selection 
    gscv_classifier = GridSearchCV( 
            estimator = classifier,  
            param_grid = parameter_range,  
            cv = kCV,
            scoring ='accuracy' 
        )
    
    # Fit the model based on best parameter selected(modified classifier)
    gscv_classifier.fit(X_train, y_train) 
    
# Print the results for Grid score & Best parameter 

    print("---------------------------------------------------------")
    
# Print the value for grid scores
    print()
    print("Grid scores on validation set:") 
    print()
    
    # store all the results in lists
    means = gscv_classifier.cv_results_['mean_test_score'] 
    stds = gscv_classifier.cv_results_['std_test_score'] 
    results = gscv_classifier.cv_results_['params'] 
    
    # Loop through all the lists and print 
    for mean, std, param in zip(means, stds, results): 
        print("Parameter: %r, accuracy: %0.3f (+/-%0.03f)" % (param, mean, 
    std*2)) 
        
# Print the value for Best Parameter 
    print()
    print("Best parameter for classification: ", gscv_classifier.best_params_)
    print()

# Predictions for test data & Confusion Matrix

# Make predictions for y(Response data)
    y_pred = gscv_classifier.predict(X_test)
    
# Plot confusion matrix   
  
    # accuracy calculation
    accuracy = metrics.accuracy_score(y_test, y_pred) * 100 
    plotcm = metrics.plot_confusion_matrix(gscv_classifier, X_test, y_test, 
                                            display_labels=class_names) 
    plotcm.ax_.set_title('Accuracy = {0:.2f}%'.format(accuracy)) 
    plt.plot()
    plt.show() 

# Print predicted values 
    print()
    print("Y Actual Values :- ")
    print(y_test)
    print()
    print()
    print("Y Predicted values :- ")
    print(y_pred)
    print()
    print("---------------------------------------------------------")
    
# return the classifier contaiing results and parameter range used in classification 
    return gscv_classifier, parameter_range





# =============================================================================
# To test without GUI

# x = set_all_variables('Iris','SVM',5)
# print(x)
# 
# =============================================================================

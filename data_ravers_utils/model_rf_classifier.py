import pandas as pd
import numpy as np
import itertools
import logging

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

dict_test_results_template = {
    'accuracy': None,
    'class_report': None,
    'conf_matrix': None
}

def get_random_forest_classifier_model(X_train, y_train, random_state, n_estimators) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=20, min_samples_split=10)
    model.fit(X_train,y_train)

    return model

def random_forest_classifier_test(model, X_test, y_test):
    # Make predictions on the test dataset
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    #Printing the results
    logging.debug(f"accuracy = {round(accuracy, 4)}")
    logging.debug(f"class_report:\n {class_report}")
    logging.debug(f"conf_matrix:\n {conf_matrix}")

    return accuracy, class_report, conf_matrix

def random_forest_classifier_control(df, feature_list, target_variable, test_size, random_state, n_estimators):
    # Test/Train split
    X = df[feature_list]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logging.debug(f'{100 - test_size * 100}% for training data: {len(X_train)}.')
    logging.debug(f'{test_size * 100}% for test data: {len(X_test)}.')

    # Train the model
    model = get_random_forest_classifier_model(X_train, y_train, random_state, n_estimators)

    accuracy, class_report, conf_matrix = random_forest_classifier_test(model, X_test, y_test)

    # collecting test results to compare
    dict_test_results = dict_test_results_template.copy()
    dict_test_results['accuracy'] = accuracy
    dict_test_results['class_report'] = class_report
    dict_test_results['conf_matrix'] = conf_matrix

    logging.debug(dict_test_results)

    return model, dict_test_results



def grid_search(model, X_train, y_train):
    # Define hyperparameter grid
    param_grid = {
        'max_depth': [5, 10, 15, 20, None],  # None allows trees to grow fully
        'min_samples_split': [2, 5, 10, 20]  # Default is 2, higher values reduce overfitting
    }

    # Perform Grid Search Cross-Validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # Return the best parameters and best score
    return grid_search.best_params_, grid_search.best_score_

def feature_importances(model, X_train):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # sorts indices of importances in descending order
    return importances, indices



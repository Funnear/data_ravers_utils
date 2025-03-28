import pandas as pd
import numpy as np
import itertools
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor

dict_test_results_template = {
    'test_size': '',
    'random_state': '',
    'R2': '',
    'MAE': '',
    'RMSE': '',
    'MSE': '',
    'n_estimators': ''
}

def get_random_forest_regressor_model(X_train, y_train, random_state, n_estimators) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, oob_score=True, warm_start=True)
    model.fit(X_train,y_train)

    return model

def random_forest_regressor_test(model, X_test, y_test):
    # Make predictions on the test dataset
    predictions = model.predict(X_test)
    r2_sc = r2_score(y_test, predictions)
    MAE_sc = mean_absolute_error(y_test, predictions)
    RMSE_sc = root_mean_squared_error(y_test, predictions)
    MSE_sc = mean_squared_error(y_test, predictions)

    #Printing the results
    logging.debug(f"R2 = {round(r2_sc, 4)}")
    logging.debug(f"MAE = {round(MAE_sc, 4)}")
    logging.debug(f"RMSE = {round(RMSE_sc, 4)}")
    logging.debug(f"MSE =  {round(MSE_sc, 4)}")

    return r2_sc, MAE_sc, RMSE_sc, MSE_sc


def random_forest_regressor_control(df, feature_list, target_variable, test_size, random_state, n_estimators):
    # Test/Train split
    X = df[feature_list]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    logging.debug(f'{100 - test_size * 100}% for training data: {len(X_train)}.')
    logging.debug(f'{test_size * 100}% for test data: {len(X_test)}.')

    # Train the model
    model = get_random_forest_regressor_model(X_train, y_train, random_state, n_estimators)

    r2_sc, MAE_sc, RMSE_sc, MSE_sc = random_forest_regressor_test(model, X_test, y_test)

    # collecting test results to compare
    dict_test_results = dict_test_results_template.copy()
    dict_test_results['test_size'] = test_size
    dict_test_results['random_state'] = random_state
    dict_test_results['R2'] = r2_sc
    dict_test_results['MAE'] = MAE_sc
    dict_test_results['RMSE'] = RMSE_sc
    dict_test_results['MSE'] = MSE_sc
    dict_test_results['n_estimators'] = n_estimators

    logging.debug(dict_test_results)

    return model, dict_test_results

def random_forest_regressor_combo_test(df, feature_groups, target_variable, test_sizes, random_states, list_n_estimators):
    # Pre-generate all combinations
    combinations = list(itertools.product(feature_groups.items(), test_sizes, random_states, list_n_estimators))

    # Evaluate and store results
    dict_model_test = {
        f"LR_{group_name}_ts{test_size}_rs{random_state}_ne{n_estimators}": random_forest_regressor_control(
            df, feature_list, target_variable, test_size, random_state, n_estimators
        )[1]  # Get only the dict_test_results
        for (group_name, feature_list), test_size, random_state, n_estimators in combinations
    }

    # Format results into DataFrame
    df_model_test = pd.DataFrame(dict_model_test).T

    # Format metrics to 4 decimal places
    columns_to_format = ['R2', 'MAE', 'RMSE', 'MSE']
    df_model_test[columns_to_format] = df_model_test[columns_to_format].astype(float).round(4)

    # Sort values
    df_model_test.sort_values('R2', ascending=False, inplace=True)

    return df_model_test


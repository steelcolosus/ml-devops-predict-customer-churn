"""
This is the Python Test for the churn_library.py module.

This module will be used to test
    1. import_data
    2. peform_eda
    3. encode_data
    4. perform_feature_engineering
    5. train_test_model

Author: Eduardo Aviles
Date: Apr 05, 2023
"""
import os
from pathlib import Path
import logging
import time
import pytest
import pandas as pd


import churn_library as cls


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename=f"./logs/churn_library_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

log = logging.getLogger("[CHURN_LOGGING_AND_TESTING]")


@pytest.fixture(scope="module")
def path():
    ''' returns path to data '''
    return "./data/bank_data.csv"


def check_data_frame(data_frame):
    ''' checks if data frame is a pandas data frame and has rows and columns 
        input:  data_frame: pandas data frame
    '''
    assert isinstance(data_frame, pd.DataFrame)
    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        log.error(
            "Testing data frame: The file doesn't appear to have rows and columns")
        raise err


def test_import(path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data = cls.import_data(path)
        log.info("SUCCESS: Testing import_data")
        pytest.data = data
    except FileNotFoundError as err:
        log.error("Testing import_eda: The file wasn't found")
        raise err
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        log.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''

    check_data_frame(pytest.data)

    eda_data = cls.perform_eda(pytest.data)

    check_data_frame(eda_data)

    pytest.eda_data = eda_data

    # Check if each file exists
    images_path = Path("./images/eda")

    for file in [
        'Churn',
        'Customer_Age',
        'Marital_Status',
        'Total_Trans_Ct',
            'Heatmap']:
        file_path = images_path.joinpath(f'{file}.png')
        try:
            assert file_path.is_file()
        except AssertionError as err:
            log.error("ERROR: Eda results not found.")
            raise err
    log.info("SUCCESS: EDA results successfully saved!")


def test_encoder_helper():
    '''
    test encoder helper
    '''
    check_data_frame(pytest.eda_data)
    eda_data = pytest.eda_data

    try:
        encoded_data = cls.encoder_helper(eda_data, cls.cat_columns, 'Churn')
        check_data_frame(encoded_data)
        pytest.encoded_data = encoded_data
    except AssertionError as err:
        log.error("ERROR: encoder_helper failed to encode the categorical columns")
        raise err

    log.info("SUCCESS: encoder_helper successfully encoded the categorical columns")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''

    check_data_frame(pytest.encoded_data)

    encoded_data = pytest.encoded_data

    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        encoded_data)

    pytest.features = (X_train, X_test, y_train, y_test)

    try:
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        log.info(
            "SUCCESS: perform_feature_engineering successfully split the data")
    except AssertionError as err:
        log.error(
            "ERROR: perform_feature_engineering failed to split the data")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = pytest.features

    cls.train_models(X_train, X_test, y_train, y_test)

    model_base_path = Path('./models')
    for model_name in ['logistic_model.pkl', 'rfc_model.pkl']:
        model_path = model_base_path.joinpath(model_name)
        try:
            assert model_path.is_file()
            log.info("SUCCESS: %s model successfully saved!", model_name)
        except AssertionError as err:
            log.error("ERROR: %s model not found.", model_name)
            raise err

'''Description: This script is used to test the functions in the churn_library.py script'''
import os
from pathlib import Path
import logging
import pytest
import pandas as pd

import churn_library as cls


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

log = logging.getLogger("[CHURN_LOGGING_AND_TESTING]")


@pytest.fixture(scope="module")
def path():
    ''' returns path to data '''
    return "./data/bank_data.csv"


def check_data_frame(data_frame):
    ''' checks if data frame is a pandas data frame and has rows and columns '''
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
        log.info("Testing import_data: SUCCESS")
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

    for file in ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct', 'Heatmap']:
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
    except AssertionError as err:
        log.error("ERROR: encoder_helper failed to encode the categorical columns")
        raise err

    log.info("SUCCESS: encoder_helper successfully encoded the categorical columns")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''


def test_train_models():
    '''
    test train_models
    '''

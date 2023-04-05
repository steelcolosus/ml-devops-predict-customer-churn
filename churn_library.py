''' Description: This file contains all the functions that are used in the churn notebook '''
import logging
# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

log = logging.getLogger('[CHURN_LIBRARY]')

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    log.info('Importing data from %s', pth)
    return pd.read_csv(pth)


def perform_eda(data):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)

    for col_name in ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct', 'Heatmap']:
        plt.figure(figsize=(20, 10))

        if col_name == "Churn":
            data['Churn'].hist()
        elif col_name == "Customer_Age":
            data['Customer_Age'].hist()
        elif col_name == "Marital_Status":
            data['Marital_Status'].value_counts('normalize').plot(kind='bar')
        elif col_name == 'Total_Trans_Ct':
            sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
        elif col_name == 'Heatmap':
            sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)

        plt.title(col_name)
        plt.savefig(f"images/eda/{col_name}.png")
        plt.close()

    return data


def encoder_helper(data, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string naming variables or index y column

    output:
            data: pandas dataframe with new columns for
    '''

    data_frame = pd.DataFrame()

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    for column_name in category_lst:
        categories = []
        groups = data.groupby(column_name).mean()[response]
        for val in data[column_name]:
            categories.append(groups.loc[val])
        data[f'{column_name}_{response}'] = categories

    data_frame[keep_cols] = data[keep_cols]

    return data_frame


def perform_feature_engineering(data, response):
    '''
    input:
              data: pandas dataframe
              response: string naming variables or index y column

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass

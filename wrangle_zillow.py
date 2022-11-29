import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

# acquire
from env import get_db_url
from pydataset import data
import seaborn as sns

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")


def null_dropper(df, prop_required_column, prop_required_row):
    
    prop_null_column = 1 - prop_required_column
    
    for col in list(df.columns):
        
        null_sum = df[col].isna().sum()
        null_pct = null_sum / df.shape[0]
        
        if null_pct > prop_null_column:
            df.drop(columns=col, inplace=True)
            
    row_threshold = int(prop_required_row * df.shape[1])
    
    df.dropna(axis=0, thresh=row_threshold, inplace=True)
    
    return df

def wrangle_zillow():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                WHERE propertylandusetypeid = 261
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    # replace any whitespace with null values
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    # drop out any null values:
    df = df.dropna()
    # cast everything as an integer:
    df = df.astype(int)
    
    #@@@@@@@@ added later, may need to be deleted
    df = df.rename(columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 'square_feet', 'taxvaluedollarcnt': 'property_value'})
    df.drop(df[df.square_feet > 70000].index, inplace=True)
    df.drop(df[df.property_value > 50000000].index, inplace=True)
    df.drop(df[df.taxamount > 400000].index, inplace=True)
    
    lm = LinearRegression()

    # fit the model to trainig data
    lm.fit(df[['property_value']], df.square_feet)

    # make prediction
    # lm.predict will output a numpy array of values,
    # we will put those values into a series in df
    #df['yhat'] = lm.predict(df[['property_value']])
    
    #df['baseline_residual'] = df.square_feet - df.baseline
    #df['residual'] = df.square_feet - df.yhat
    
    #df['baseline_residual_2'] = df.baseline_residual**2
    #df['residual_2'] = df.residual**2
    
    return df

def split_zillow(df):
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test
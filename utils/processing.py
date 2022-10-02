import numpy as np
import pandas as pd

from utils.config import *

def convert_to_lowercase(df, columns):
    for col in columns:
        df[col] = df[col].str.lower()
    
    return df


def preprocess_property_type(df):
    '''
    1. Standardise all casing to lower case
    2. One-hot encode property type to public/hybrid/private type
    '''
    df = convert_to_lowercase(df, convert_to_lowercase_cols)
    df['property_type'].replace(public_housing_property_type, 'property_type_public', inplace=True)
    df['property_type'].replace(private_housing_property_type, 'property_type_private', inplace=True)
    df['property_type'].replace(hybrid_housing_property_type, 'property_type_hybrid', inplace=True)
    df = pd.merge(df, pd.get_dummies(df['property_type'], drop_first=True), left_index=True, right_index=True)
    
    return df


def preprocess_tenure(df):
    '''
    1. Impute missing value with NA
    2. One-hot encode tenure to low, high, freehold, NA type
    '''
    df['tenure'] = df['tenure'].replace({np.nan: 'NA'})
    df['tenure'].replace(tenure_low_year, 'tenure_low_year', inplace=True)
    df['tenure'].replace(tenure_high_year, 'tenure_high_year', inplace=True)
    df = pd.merge(df, pd.get_dummies(df['tenure'], drop_first=True), left_index=True, right_index=True)
    
    return df


def preprocess_num_beds(df):
    '''
    Remove missing values
    '''
    df = df[df['num_beds'].notna()]
    
    return df 


def preprocess_num_baths(df):
    '''
    Fill missing values with median num_baths of corresponding num_beds values
    '''
    median_num_baths = df.groupby('num_beds').agg({'num_baths':'median'}).rename(columns={'num_baths': 'median_num_baths'}).reset_index()
    df = pd.merge(df, median_num_baths, on='num_beds', how='left')
    df['num_baths'] = np.where(df['num_baths'].isnull(),df['median_num_baths'],df['num_baths'])
    df.drop(columns='median_num_baths', inplace=True)
    
    return df 


def preprocess_size_sqft(df):
    '''
    Remove outliers:
    1. Remove if sqft = 0
    2. Remove if sqft >= 70000  
    '''
    df = df[(df['size_sqft'] > 0) & (df['size_sqft'] < 70000)]
    
    return df 


def preprocess_floor_level(df):
    '''
    Do simple replacement. nans are filled with NA and others we strip the unnnecessary info
    ''' 
    df['floor_level'] = df['floor_level'].fillna('NA')
    df.loc[df['floor_level'] == 'ground (9 total)', 'floor_level'] = 'ground'
    df.loc[df['floor_level'] == 'high (70 total)', 'floor_level'] = 'high'
    df.loc[df['floor_level'] == 'low (17 total)', 'floor_level'] = 'low'
    df.loc[df['floor_level'] == 'high (25 total)', 'floor_level'] = 'high'
    df.loc[df['floor_level'] == 'mid (25 total)', 'floor_level'] = 'mid'
    df.loc[df['floor_level'] == 'low (23 total)', 'floor_level'] = 'low'
    df.loc[df['floor_level'] == 'high (23 total)', 'floor_level'] = 'high'
    df.loc[df['floor_level'] == 'high (10 total)', 'floor_level'] = 'high'
    df.loc[df['floor_level'] == 'high (9 total)', 'floor_level'] = 'high'
    df.loc[df['floor_level'] == 'high (17 total)', 'floor_level'] = 'high'
    df.loc[df['floor_level'] == 'mid (9 total)', 'floor_level'] = 'mid'
    df_ohe = pd.get_dummies(df['floor_level'], drop_first=True)
    renamed_cols = ['floor_level_' + x for x in df_ohe.columns]
    df_ohe.columns = renamed_cols
    df = pd.merge(df, df_ohe, left_index=True, right_index=True)
    
    return df 


def preprocess_furnishing(df):
    '''
    1. Group 'na' and 'unspecified' values to same category
    2. One-hot encode furnishing to fully, partial, unfurnished, unspecified
    '''
    df['furnishing'].replace({'na': 'unspecified'}, inplace=True)
    df_ohe = pd.get_dummies(df['furnishing'], drop_first=True)
    renamed_cols = ['furnishing_' + x for x in df_ohe.columns]
    df_ohe.columns = renamed_cols
    df = pd.merge(df, df_ohe, left_index=True, right_index=True)
    
    return df


def preprocess(df):
    df = preprocess_property_type(df)
    df = preprocess_tenure(df)
    df = preprocess_num_beds(df)
    df = preprocess_num_baths(df)
    df = preprocess_size_sqft(df)
    df = preprocess_floor_level(df)
    df = preprocess_furnishing(df)
    
    return df


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    df = preprocess(df)

    print(df[:10])

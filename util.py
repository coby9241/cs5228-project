import pandas as pd

def preprocess(df):
    # Drop metadata and unimportant columns
    # 1. title: meta data on property
    # 2. property_details_url: meta data on property
    # 3. elevation: 100% of value 0
    drop_columns = ['title', 'property_details_url', 'elevation', 'available_unit_types', 'total_num_units']
    df = df.drop(drop_columns, axis=1)

    # property_type
    # 1. Make all values lower case
    # 2. Reduce all hdb types to hdb
    df['property_type'] = df['property_type'].str.lower()
    df['property_type'] = df['property_type'].replace(to_replace=r'^hdb.*$', value='hdb', regex=True)
    
    # floor_level, do simple replacement. nans are filled with NA and others we strip the unnnecessary info
    df['floor_level'].fillna('NA')
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

    # Return preprocess data
    return df

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    df = preprocess(df)

    print(df[:10])

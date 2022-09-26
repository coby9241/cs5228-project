import pandas as pd

def preprocess(df):
    # Drop metadata and unimportant columns
    # 1. title: meta data on property
    # 2. property_details_url: meta data on property
    # 3. elevation: 100% of value 0
    drop_columns = ['title', 'property_details_url', 'elevation']
    df = df.drop(drop_columns, axis=1)

    # property_type
    # 1. Make all values lower case
    # 2. Reduce all hdb types to hdb
    df['property_type'] = df['property_type'].str.lower()
    df['property_type'] = df['property_type'].replace(to_replace=r'^hdb.*$', value='hdb', regex=True)

    # Return preprocess data
    return df

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    df = preprocess(df)

    print(df[:10])

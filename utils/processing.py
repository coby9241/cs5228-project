from doctest import DocFileSuite
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

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


def preprocess_num_beds(df, is_target=False):
    '''
    Remove missing values
    '''
    if not is_target:
        df = df[df['num_beds'].notna()]
    else:
        # todo, fill with mean or median of training data
        df['num_beds'].fillna(0, inplace=True)

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


def preprocess_size_sqft(df, is_target=False):
    '''
    Remove outliers:
    1. Remove if sqft = 0
    2. Remove if sqft >= 70000
    '''
    if not is_target:
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


def preprocess_latlong(df, is_target=False):
    '''
    Filter only lat-lng coordinates within Singapore
    '''
    if is_target:
        return df

    min_lat, min_lng, max_lng, max_lat = 0., 100., 115., 10.
    df = df[(df.lat > min_lat) & (df.lat < max_lat)]
    df = df[(df.lng > min_lng) & (df.lng < max_lng)]

    return df


def preprocess_price(df, is_target=False):
    '''
    Filter out prices = 0 and prices in the top 1%
    '''
    if is_target:
        return df

    min_price = 0.
    max_price = 2.289000e7
    df = df[(df.price > min_price) & (df.price <= max_price)]

    return df


def calculate_haversine_distance_in_km(lon1, lat1, lon2, lat2):
    '''
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    '''
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # radius of earth in kilometers
    return c * r


def join_with_mrt_stations(df, mrt_stations_df):
    '''
    Merge dataframe with auxillary mrt stations dataframe to find:
    1. Distance to the nearest mrt station
    2. Line of the nearest mrt station
    '''
    # create a common joining key
    df['key'] = 0
    mrt_stations_df['key'] = 0

    # rename the overlapping columns
    mrt_stations_df = mrt_stations_df.rename(columns={'lat': 'lat_y', 'lng': 'lng_y'})

    # get nearest mrt station in km
    df = df.merge(mrt_stations_df[['name', 'key', 'lat_y', 'lng_y']], on='key')
    df['nearest_mrt_distance_in_km'] = calculate_haversine_distance_in_km(df['lng'], df['lat'], df['lng_y'], df['lat_y'])
    df = df.loc[df.groupby('listing_id')['nearest_mrt_distance_in_km'].idxmin()]

    # one hot encoding of the line of nearest mrt station
    mrt_lines_df = pd.concat([
        mrt_stations_df[['name']],
        pd.get_dummies(mrt_stations_df[['line']])
    ], axis=1).groupby('name').sum().reset_index()
    df = df.merge(mrt_lines_df, on='name', how='left')

    # drop unncessary columns
    df = df.drop(columns=['key', 'lat_y', 'lng_y', 'name'])

    return df


def join_with_regions(df, regions_df):
    '''
    Merge dataframe with auxillary regions dataframe to find:
    1. Region in Singapore that the planning area belongs to
    '''
    regions_df['planning_area'] = regions_df.planning_area.str.lower()
    df = df.merge(regions_df, on='planning_area', how='left')

    return df


def join_with_subzones(df, subzones_df):
    '''
    Merge dataframe with auxillary subzones dataframe to find:
    1. Population, area size and density of the subzone in which the listing resides
    '''
    df = df.merge(subzones_df[['name', 'area_size', 'population']], left_on='subzone', right_on='name', how='left')
    df['density'] = df['population']/df['area_size']
    df = df.drop(columns=['name'])

    return df


def join_with_primary_schools(df, primary_schools_df):
    '''
    Merge dataframe with auxillary primary schools dataframe to find:
    1. Distance to the nearest GEP primary school
    2. Distance to the nearest primary school
    '''
    # create a common joining key
    df['key'] = 0
    primary_schools_df['key'] = 0

    # rename the overlapping columns
    primary_schools_df = primary_schools_df.rename(columns={'lat': 'lat_y', 'lng': 'lng_y'})

    # get distances to all schools from all listings
    all_sch_df = df.merge(primary_schools_df[['name', 'key', 'lat_y', 'lng_y']], on='key')
    all_sch_df['pri_sch_distance_in_km'] = calculate_haversine_distance_in_km(all_sch_df['lng'], all_sch_df['lat'], all_sch_df['lng_y'], all_sch_df['lat_y'])
    all_sch_df['is_gep_pri_sch'] = np.where(all_sch_df['name'].isin(gep_school_names), 1, 0)

    # get nearest primary school in km for each listing_id
    nearest_pri_sch_df = all_sch_df.loc[all_sch_df.groupby('listing_id')['pri_sch_distance_in_km'].idxmin()][['listing_id', 'pri_sch_distance_in_km']]
    nearest_pri_sch_df = nearest_pri_sch_df.rename(columns={'pri_sch_distance_in_km': 'nearest_pri_sch_distance_in_km'})
    df = pd.merge(df, nearest_pri_sch_df, on='listing_id', how='left')

    # get nearest gep primary school in km for each listing_id
    nearest_gep_pri_sch_df = all_sch_df.loc[all_sch_df[all_sch_df.is_gep_pri_sch == 1].groupby('listing_id')['pri_sch_distance_in_km'].idxmin()][['listing_id', 'pri_sch_distance_in_km']]
    nearest_gep_pri_sch_df = nearest_gep_pri_sch_df.rename(columns={'pri_sch_distance_in_km': 'nearest_gep_pri_sch_distance_in_km'})
    df = pd.merge(df, nearest_gep_pri_sch_df, on='listing_id', how='left')

    # feature engineering
    df['gep_pri_sch_within_1km'] = np.where(df['nearest_gep_pri_sch_distance_in_km'] <= 1., 1, 0)
    df['gep_pri_sch_within_1km_2km'] = np.where((df['nearest_gep_pri_sch_distance_in_km'] > 1.) & (df['nearest_gep_pri_sch_distance_in_km'] <= 2.), 1, 0)
    df['gep_pri_sch_outside_2km'] = np.where(df['nearest_gep_pri_sch_distance_in_km'] > 2., 1, 0)
    df['pri_sch_within_500m'] = np.where(df['nearest_pri_sch_distance_in_km'] <= 0.5, 1, 0)
    df['pri_sch_outside_500m'] = np.where(df['nearest_pri_sch_distance_in_km'] > 0.5, 1, 0)

    # drop unnecessary columns
    df = df.drop(columns=['key'])

    return df


def join_with_shopping_malls(df, shopping_malls_df):
    '''
    Merge dataframe with auxillary shopping malls dataframe to find:
    1. Distance to the nearest shopping mall
    '''
    # create a common joining key
    df['key'] = 0
    shopping_malls_df['key'] = 0

    # rename the overlapping columns
    shopping_malls_df = shopping_malls_df.rename(columns={'lat': 'lat_y', 'lng': 'lng_y'})

    # get the nearest shopping mall for each listing
    df = df.merge(shopping_malls_df[['lat_y', 'lng_y', 'key']], on='key')
    df['nearest_mall_distance_in_km'] = calculate_haversine_distance_in_km(df['lng'], df['lat'], df['lng_y'], df['lat_y'])
    df = df.loc[df.groupby('listing_id')['nearest_mall_distance_in_km'].idxmin()]

    # drop unnecessary columns
    df = df.drop(columns=['key', 'lat_y', 'lng_y'])

    return df


def join_with_commercial_centres(df, commercial_centres_df):
    '''
    Merge dataframe with auxillary commercial centres dataframe to find:
    1. Distance to the nearest commercial centre
    '''
    # create a common joining key
    df['key'] = 0
    commercial_centres_df['key'] = 0

    # rename the overlapping columns
    commercial_centres_df = commercial_centres_df.rename(columns={'lat': 'lat_y', 'lng': 'lng_y'})

    # get the nearest commercial centre for each listing
    df = df.merge(commercial_centres_df[['type', 'lat_y', 'lng_y', 'key']], on='key')
    df['nearest_com_centre_distance_in_km'] = calculate_haversine_distance_in_km(df['lng'], df['lat'], df['lng_y'], df['lat_y'])
    df = df.loc[df.groupby('listing_id')['nearest_com_centre_distance_in_km'].idxmin()]

    df = pd.merge(df, pd.get_dummies(df['type'], prefix='cc_type'), left_index=True, right_index=True)

    # drop unnecessary columns
    df = df.drop(columns=['key', 'lat_y', 'lng_y', 'type'])

    return df


def preprocess(df, is_target=False):
    df = preprocess_property_type(df)
    df = preprocess_tenure(df)
    df = preprocess_num_beds(df, is_target)
    df = preprocess_num_baths(df)
    df = preprocess_size_sqft(df, is_target)
    df = preprocess_floor_level(df)
    df = preprocess_furnishing(df)
    df = preprocess_latlong(df, is_target)
    if not is_target:
        df = preprocess_price(df, is_target)

    return df


def read_aux_csv(path):
    dfs = dict()
    dfs['mrt_stations'] = pd.read_csv("{}/auxiliary-data/sg-mrt-stations.csv".format(path))
    dfs['primary_schools'] = pd.read_csv("{}/auxiliary-data/sg-primary-schools.csv".format(path))
    dfs['commercial_centres'] = pd.read_csv("{}/auxiliary-data/sg-commerical-centres.csv".format(path))
    dfs['shopping_malls'] = pd.read_csv("{}/auxiliary-data/sg-shopping-malls.csv".format(path))
    dfs['secondary_schools'] = pd.read_csv("{}/auxiliary-data/sg-secondary-schools.csv".format(path))
    dfs['subzones'] = pd.read_csv("{}/auxiliary-data/sg-subzones.csv".format(path))
    dfs['regions'] = pd.read_csv("{}/extra/sg-regions.csv".format(path))

    return dfs


def join_aux(df, adfs):
    fn_dict = {
        'mrt_stations': join_with_mrt_stations,
        'primary_schools': join_with_primary_schools,
        'commercial_centres': join_with_commercial_centres,
        'shopping_malls': join_with_shopping_malls,
        'subzones': join_with_subzones,
        'regions': join_with_regions,
    }

    df = df.copy()

    for k,adf in adfs.items():
        if k in fn_dict:
            df = fn_dict[k](df, adf)

    return df


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    df = preprocess(df)

    print(df.columns)
    print(df[:10])

    # read and join auxiliary data
    adfs = read_aux_csv('./data')
    df = join_aux(df, adfs)

    print(df.columns)
    print(df[:10])

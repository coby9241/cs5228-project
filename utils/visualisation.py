import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
    
    
def plot_boxplot(df, feature=None, y='price', y_min=-np.inf, y_max=np.inf, figsize=(15, 3), whis=1.5, categories=None, binary=False):
    '''
    df: dataframe containing predictors and target variable
    feature: name of feature
    y: target variable. Default value: price
    y_min: min threshold of target variable to be plotted on box plot for better visualisation. Default value: - infinity
    y_max: max threshold of target variable to be plotted on box plot for better visualisation. Default value: infinity
    figsize: figure size of box plot. Default value: (15, 3) 
    whis: whis of boxplot. Can pass float or (float, float). E.g. if (0.05, 0.95) means whis is at 5th and 95th percentile. Default value: 1.5
    categories: pre-defined list of categories to group the boxplots by. Default value: None
    binary: in the case that the features are one-hot encoded values. Default value: False
    '''
    fig, ax = plt.subplots(figsize=figsize)

    # Filter outliers in y 
    dataset = df.copy(deep=True)
    dataset = dataset[(dataset[y] > y_min) & (dataset[y] < y_max)]

    row = []

    if categories is None:
        categories = sorted(df[feature].drop_duplicates().tolist())

    if binary: 
        for cat in categories: 
            row.append(dataset[dataset[cat]==1].price.tolist())
    else:
        for cat in categories: 
            row.append(dataset[dataset[feature]==cat][y].tolist())

    pos = np.array(range(len(row))) + 1
    ax.boxplot(row, positions=pos, vert=False, whis=whis)
    ax.set_yticklabels(categories)
    
    plt.ylabel(feature)
    plt.xlabel(y)
    plt.title(f'{y.upper()} distribution (Box Plot) across {feature.upper()}')
    plt.show()
    
    
def plot_scatterplot(df, X, y, y_min=-np.inf, y_max=np.inf, figsize=(15,10), group=None):
    '''
    df: dataframe containing predictors and target variable
    X: x variable
    y: y variable
    y_min: min threshold of y variable to be plotted on box plot for better visualisation. Default value: - infinity
    y_max: max threshold of y variable to be plotted on box plot for better visualisation. Default value: infinity
    figsize: figure size of box plot. Default value: (15, 10) 
    '''
    ax = df[(df[y] > y_min) & (df[y] < y_max)].plot(
        kind="scatter", 
        x=X, 
        y=y, 
        figsize=figsize,
        alpha=0.4
    )
    
    # add axis labels
    plt.ylabel(y)
    plt.xlabel(X)
    
    if group is not None: 
        plt.title(f'{y.upper()} vs {X.upper()} for {group}')
    else: 
        plt.title(f'{y.upper()} vs {X.upper()}')

    plt.show()

def plot_map(property_listings_df, feature_df, title, group_col=None, figsize=(10, 5)): 

    singapore_img = mpimg.imread('data/extra/sg-map.png')

    fig = plt.figure(figsize=figsize)
    plt.scatter(
        x=property_listings_df.lng, 
        y=property_listings_df.lat, 
        c=property_listings_df.price,
        cmap="Reds",
        alpha=0.4, 
        s=5
    )
    if group_col is not None:
        for grp in feature_df[group_col].unique():
            plt.scatter(
                x=feature_df[feature_df[group_col] == grp].lng, 
                y=feature_df[feature_df[group_col] == grp].lat, 
                s=10, 
                marker="x",
                label=grp
            )
    else: 
        plt.scatter(
            x=feature_df.lng, 
            y=feature_df.lat, 
            s=10, 
            marker="x"
        )

    # use our map with it's bounding coordinates
    plt.imshow(singapore_img, extent=[103.5, 104., 1.15, 1.50], alpha=0.5)     
    # add axis labels
    plt.ylabel("latitude")
    plt.xlabel("longitude")
    # set the min/max axis values - these must be the same as above
    plt.ylim(1.15, 1.50)
    plt.xlim(103.5, 104)

    plt.title(title)

    plt.legend(loc="best")

    plt.show()
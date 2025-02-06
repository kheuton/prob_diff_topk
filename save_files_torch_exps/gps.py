import pandas as pd
from pandas.tseries.offsets import DateOffset
from shapely.geometry import Polygon
import numpy as np
import geopandas as gpd
import os
from datetime import datetime
import argparse


"""
OPTIONS FOR PARAMETERS:
temporal_res = ['daily', 'weekly', 'biweekly', 'monthly', '2monthly', '3monthly']
box_length_m: numeric, meters
years_cut_from_back: numeric, yrs --> As we know, the dataset's GPS trackers started dying in 2014. The 
    following parameter allows control over how many years are cut since 2018. 
    No more than 4 years is recommended.
keep_geometry_col: bool
"""

# defining date range
DATE_RANGE_TRANSLATOR = {  
    'daily': 'D',
    'weekly': 'W',
    'biweekly': '2W',
    'monthly': 'ME',
    '2monthly': '2ME',
    '3monthly': '3ME'
}
# how much temporal buffer to give based on resolution
DATE_OFFSET_TRANSLATOR = {  
    'daily': 1,
    'weekly': 7,
    'biweekly': 14,
    'monthly': 30,
    '2monthly': 60,
    '3monthly': 90
}
# naming the temporal column
DATE_NAME_TRANSLATOR = {  
    'daily': 'day',
    'weekly': 'week',
    'biweekly': 'biweek',
    'monthly': 'month',
    '2monthly': 'bimonth',
    '3monthly': 'trimonth',
    'seasonal': 'season'
}

MONTH_PAIRS_TRANSLATOR = {
    '2monthly': ["02-28", "04-30", "10-20", "12-25"],
    '3monthly': ["01-31", "04-30", "10-20"]
}

METERS_PER_DEGREE = 111111

# Function to generate a grid of boxes
def generate_grid(bbox, spacing, crs):
    """
    Generate box grid based on min x, min y, max x, and max y (LONG/LAT)
    Spacing: Space between each box in degrees
    Crs: Coordinate reference system
    """
    METERS_PER_DEGREE = 111111

    if crs.to_string() == 'EPSG:26914':
        spacing = spacing * METERS_PER_DEGREE

    minx, miny, maxx, maxy = bbox
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)

    grid = []
    for x in x_coords:
        for y in y_coords:
            grid.append(Polygon([(x, y), (x + spacing, y), (x + spacing, y + spacing), (x, y + spacing), (x, y)]))
    return gpd.GeoDataFrame({'geometry': grid}, crs=crs)



def sort_date_to_range(input_date, temporal_res):
    """
    Determines the range in which a given date falls based on month-day pairs.
    
    Parameters:
    - input_date (datetime): The date to evaluate.
    
    Returns:
    - str: The range in the format "year-month-day_to_year-month-day".
    """
    # set possible dates for range
    yr = input_date.year
    curr_year = [datetime.strptime(f"{yr}-{md}", "%Y-%m-%d") for md in MONTH_PAIRS_TRANSLATOR[temporal_res]]
    last_year = [datetime.strptime(f"{yr - 1}-{md}", "%Y-%m-%d") for md in MONTH_PAIRS_TRANSLATOR[temporal_res]]
    next_year = [datetime.strptime(f"{yr + 1}-{md}", "%Y-%m-%d") for md in MONTH_PAIRS_TRANSLATOR[temporal_res]]

    # extract correct range
    all_dates = last_year + curr_year + next_year
    idx = np.searchsorted(all_dates, input_date)
    correct_range = f"{all_dates[idx - 1].date()}_to_{all_dates[idx].date()}"

    return correct_range


def set_date_range_2_3mo(year, temporal_res):

    # Specify the year

    if temporal_res == '2monthly':
        # Create a date range for the specific dates
        dates = [
            pd.Timestamp(year, 10, 20),    # October 20
            pd.Timestamp(year, 12, 25),    # December 25
            pd.Timestamp(year + 1, 2, 28),  # Feb 28
            pd.Timestamp(year + 1, 4, 30)      # April 30
        ]
    elif temporal_res == '3monthly':
        # Create a date range for the specific dates
        dates = [
            pd.Timestamp(year, 10, 20),    # October 20
            pd.Timestamp(year + 1, 1, 31),    # December 31
            pd.Timestamp(year + 1, 4, 30)      # April 30
        ]

    # Convert to a Pandas DateTimeIndex
    return pd.DatetimeIndex(dates)



def determine_season_year(date):
    return date.year if date.month >= 7 else date.year - 1


def read_gps(years_cut_from_back=0, temporal_res='weekly', path='code/prob_diff_topk/save_files_torch_exps', keep_geometry_col=True):
    

    gps_gdf = pd.read_csv(f"{path}/raw-data/WHCR_locations_gps.csv")

    gps_gdf = gpd.GeoDataFrame(gps_gdf, geometry=gpd.points_from_xy(gps_gdf.Long, gps_gdf.Lat), crs='EPSG:4326')
    gps_gdf = gps_gdf.set_crs('EPSG:4326', allow_override=True)
    gps_gdf = gps_gdf.to_crs('EPSG:26914')
    gps_gdf['Date'] = pd.to_datetime(gps_gdf['Date'])

    if temporal_res == 'seasonal':
        gps_gdf['Year'] = gps_gdf['Date'].dt.year
        gps_gdf['season'] = gps_gdf['Date'].apply(determine_season_year)
        gps_gdf['season_name'] = gps_gdf['season']
    
    elif temporal_res == '2monthly' or temporal_res == '3monthly':

        gps_gdf['season'] = gps_gdf['Date'].apply(determine_season_year)

        all_dates = pd.DatetimeIndex([])
        
        for szn in gps_gdf['season'].unique():

            # set month ID
            curr_dates = set_date_range_2_3mo(year=szn, temporal_res=temporal_res)
            gps_gdf.loc[gps_gdf['season'] == szn, DATE_NAME_TRANSLATOR[temporal_res]] = np.searchsorted(curr_dates, gps_gdf[gps_gdf['season'] == szn]['Date'])
            gps_gdf = gps_gdf.sort_values(by=['season', DATE_NAME_TRANSLATOR[temporal_res]]).reset_index(drop=True)
            gps_gdf[DATE_NAME_TRANSLATOR[temporal_res]] = pd.factorize(list(zip(gps_gdf['season'], gps_gdf[DATE_NAME_TRANSLATOR[temporal_res]])))[0] + 1

            all_dates = np.concatenate((all_dates, curr_dates))

        gps_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = gps_gdf['Date'].apply(lambda d: sort_date_to_range(d, temporal_res))
        
        # only keep dates that fall in the designated 2 month range. So, nothing in the summertime (april to october)
        def extract_mo_day(date_rnge):
            return date_rnge.split('_')[0].split('-')[1] + '-' + date_rnge.split('_')[0].split('-')[2]

        valid_idxs = gps_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'].apply(lambda rnge: extract_mo_day(rnge) != '04-30')
        gps_gdf = gps_gdf[valid_idxs]
        gps_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}'] = gps_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}'].rank(method='dense').astype(int)

    else:
        gps_gdf['Year'] = gps_gdf['Date'].dt.year
        valid_years = gps_gdf['Year'].unique()[:len(gps_gdf['Year'].unique()) - years_cut_from_back]
        gps_gdf = gps_gdf[gps_gdf['Year'].isin(valid_years)]

        all_dates = pd.date_range(start=gps_gdf['Date'].min() - DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), end=gps_gdf['Date'].max() + DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), freq=DATE_RANGE_TRANSLATOR[temporal_res])
        gps_gdf[DATE_NAME_TRANSLATOR[temporal_res]] = np.searchsorted(all_dates, gps_gdf['Date'])  

        # add names for weeks for data clarity
        bin_names = {i + 1: f'{all_dates[i].date()}_to_{all_dates[i + 1].date()}' for i in range(len(all_dates) - 1)}
        gps_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = gps_gdf[DATE_NAME_TRANSLATOR[temporal_res]].map(bin_names)

    # final steps
    gps_gdf = gps_gdf.rename(columns={'Date': 'date', 'Long': 'X', 'Lat': 'Y'})
    gps_gdf['count'] = 1
    columns_of_interest = ['date', f'{DATE_NAME_TRANSLATOR[temporal_res]}', f'{DATE_NAME_TRANSLATOR[temporal_res]}_name', 'X', 'Y', 'count', 'geometry']
    return gps_gdf[columns_of_interest]

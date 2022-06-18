# %%
import pandas as pd
import numpy as np
import os
import mlflow
from pyparsing import Regex
from src.data.utils import build_output_dirs
from src.configs.data_config import MIN_YEAR,MAX_YEAR,RAW_COLUMNS,POS_IDS
from src.configs.data_config import IDENTIFIERS, REALS, CATEGORIES, TARGET


# %%

season_df  = pd.read_csv('./data/raw/raw_df.csv',index_col=0, encoding="utf=8")
season_df['name'] = season_df.name.str.replace('_',' ').str.replace('\d+', '',regex=True).str.strip()
agg_dict = {
    'value':'first',
    'ict_index':'sum',
    'minutes':'sum',
    'total_points':'sum'
        }
season_df = season_df.groupby(['uid','name','season'],as_index=False).agg(agg_dict)
season_df['start_year'] =season_df.season.str.split('-').apply(lambda x:x[0]).astype(int)
season_df= season_df.sort_values(['name','start_year']).reset_index(drop=True)

lag_cols = ['ict_index','minutes','total_points']
lagged_feats = season_df.groupby(['name'])[lag_cols].shift(1)
lagged_feats.columns = [col+'_lag1' for col in lagged_feats.columns]

model_data = season_df[['total_points','value']].merge(lagged_feats,how='left',left_index=True,right_index=True)

y = model_data.pop('total_points')
X = model_data

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

# %%
dirs= ['./data/processed/']
build_output_dirs(dirs)

X_train.to_csv('./data/processed/X_train.csv'),y_train.to_csv('./data/processed/y_train.csv')
X_val.to_csv('./data/processed/X_val.csv'),y_train.to_csv('./data/processed/y_val.csv')
X_test.to_csv('./data/processed/X_test.csv'),y_train.to_csv('./data/processed/y_test.csv')

# %%

# load 2024-11-09-11-28-26.parquet into a pandas dataframe

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import glob

# # Load all session files
# session_files = glob.glob('data/data_session_*.parquet')

# # Combine them into a single DataFrame for training
# df_list = [pd.read_parquet(file) for file in session_files]
# combined_df = pd.concat(df_list, ignore_index=True)

df = pd.read_parquet('data/data_session_2024-11-09-13-01-02.parquet')

# get the lists of the keys in the ask_prices column in the first row
keys = df['ask_prices'].iloc[0].keys()

# join the ask_prices column to the dataframe
df = df.join(pd.json_normalize(df['ask_prices'])).drop(columns=['ask_prices'])

# change the price in the keys columns to the relative change
for key in keys:
    df[key] = df[key].pct_change()

# drop the first row and reset the index
df = df.drop(0).reset_index(drop=True)

# normalize ticker using its min val: 1 and max val: 2
df['period'] = (df['period'] - 1) / (2 - 1)

# normalize ticker using its min and max values
df['ticker'] = (df['ticker'] - df['ticker'].min()) / (df['ticker'].max() - df['ticker'].min())



print(df.head())

print(keys)
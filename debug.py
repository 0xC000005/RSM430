# load the average_returns csv
import pandas as pd

average_returns = pd.read_csv('average_returns.csv')

# locate the row with BOC SOLD MORE 3- AND 5- YEAR BONDS - Bank of Canada sold $3 billion each of 3-Year and 5-Year bonds yesterday. The bid-to-cover ratios, which gauge demand by comparing the amount offered to the amount sold, were greater than 2.50 for both maturities. This showed greater than expected demand.

news = "BOC SOLD MORE 3- AND 5- YEAR BONDS - Bank of Canada sold $3 billion each of 3-Year and 5-Year bonds yesterday. The bid-to-cover ratios, which gauge demand by comparing the amount offered to the amount sold, were greater than 2.50 for both maturities. This showed greater than expected demand."

news_row = average_returns[average_returns['news'] == news]

# print the min and max value of the columns besides the news column in the row
if not news_row.empty:
    max_column = news_row.drop(columns=['news']).idxmax(axis=1).values[0]
    min_column = news_row.drop(columns=['news']).idxmin(axis=1).values[0]
    print(f"Max column: {max_column}, Min column: {min_column}")
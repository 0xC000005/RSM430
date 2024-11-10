# load the combined_data.csv

import pandas as pd
combined_df = pd.read_csv("combined_data.csv")

# calculate the return from news
# idea: for each none_empty news, calculate the return each each security over the next n ticks, 
# unless there are less than n ticks left
# the security columns are CorpBond* and GovBond*

def calculate_return_from_news(df, news_row_number, num_tickers):
    # input is the df, the row number of the news, and the number of tickers to consider 
    # first you crop the df to only include the rows from the news_row_number to the news_row_number + num_tickers
    # then you calculate the accumulative return for each security

    # Define the security columns
    security_columns = ['CorpBondA', 'CorpBondB', 'CorpBondC', 'GovtBondY2', 'GovtBondY5', 'GovtBondY10']

    # Calculate the ending row index
    end_row = min(news_row_number + num_tickers, len(df))

    # Extract the subset of the dataframe
    subset_df = df.iloc[news_row_number:end_row][security_columns]

    # Calculate the accumulated returns by compounding the percentage changes
    accumulated_returns = (subset_df + 1).prod() - 1

    return accumulated_returns

ticker_of_interest = 10

# find the first row where the news is not None
first_news_row = combined_df[combined_df['news'].notnull()].index[0]
print(first_news_row)
print(calculate_return_from_news(combined_df, first_news_row, ticker_of_interest))
# print the news
print(combined_df.iloc[first_news_row]['news'])

# find all the rows where the news is not None
news_rows = combined_df[combined_df['news'].notnull()].index

price_change = {}

for news_row in news_rows:
    # if the news is not in the price_change dictionary, add its return to the dictionary, the key is the news
    if combined_df.iloc[news_row]['news'] not in price_change:
        price_change[combined_df.iloc[news_row]['news']] = [calculate_return_from_news(combined_df, news_row, ticker_of_interest)]
    # if the news is in the price_change dictionary, append its return to the dictionary
    else:
        price_change[combined_df.iloc[news_row]['news']].append(calculate_return_from_news(combined_df, news_row, ticker_of_interest))

# print the average return of each security for each news
# Create a list to store the results
results = []

for key, value in price_change.items():
    print(key)
    avg_return = pd.DataFrame(value).mean()
    print(avg_return)
    # Append the result to the list
    results.append([key] + avg_return.tolist())

# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['news'] + ['CorpBondA', 'CorpBondB', 'CorpBondC', 'GovtBondY2', 'GovtBondY5', 'GovtBondY10'])

# Sort the DataFrame by the news column in alphabetical order
results_df = results_df.sort_values(by='news')

# Save the DataFrame to a CSV file
results_df.to_csv('average_returns.csv', index=False)
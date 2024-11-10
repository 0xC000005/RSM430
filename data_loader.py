import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import news_classifier
import sentence_embedder
import numpy as np
import tqdm
import glob

def get_news_tensor(news):
     if news != None:
             sentence_embedding = sentence_embedder.generate_embeddings(news)
             sentiment_probabilities = news_classifier.get_news_sentiments(news)
             return np.concatenate((sentence_embedding, sentiment_probabilities))
     else:
         return None


def prepare_dataframe_for_training(path):
    df = pd.read_parquet(path)

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

    # normalize ticker using its min val: 0 and max val: 331
    df['ticker'] = (df['ticker'] - 0) / (331 - 0)

    df['news'] = df['news'].apply(get_news_tensor)

    # # now since news will always have 771 elements, we can expand it into 771 columns
    # df = df.join(pd.DataFrame(df['news'].tolist())).drop(columns=['news'])

    return df


def prepare_dataframe_for_lookup(path):
    df = pd.read_parquet(path)

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

    # normalize ticker using its min val: 0 and max val: 331
    df['ticker'] = (df['ticker'] - 0) / (331 - 0)

    return df




if __name__ == '__main__':
    # Get the path to all data/data_session_*.parquet
    paths = glob.glob("data/data_session_*.parquet")
    combined_processed_df = pd.DataFrame()
    combined_df = pd.DataFrame()
    for path in tqdm.tqdm(paths):
        df = prepare_dataframe_for_lookup(path)
        processed_df = prepare_dataframe_for_training(path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_processed_df = pd.concat([combined_processed_df, processed_df], ignore_index=True)
        
    # save as a parquet file
    table = pa.Table.from_pandas(combined_processed_df)
    pq.write_table(table, "combined_processed_data.parquet")

    # save as a csv file
    combined_df.to_csv("combined_data.csv", index=False)

    
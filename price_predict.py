# load the combined data pq
import pandas as pd
combined_data = pd.read_parquet("combined_processed_d.parquet")
# save it as a csv
combined_data.to_csv("combined_processed_data.csv", index=False)
print(combined_data.shape)
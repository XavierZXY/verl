import pandas as pd

df = pd.read_parquet("data/valid_dataset.parquet")
# get the features
print(df.columns)
# show the first row
print(df.head(2))

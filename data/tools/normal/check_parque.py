import pandas as pd

df = pd.read_parquet("/home/takisobe@amd.com/zxy/codes/verl-compare/data/data-v2/mvtec/train_cropped.parquet")
# get the features
print(df.columns)
# show the first row
# print(df.head(2))
# get total number of rows
print(len(df))

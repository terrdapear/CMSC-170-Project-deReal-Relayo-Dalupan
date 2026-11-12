import pandas as pd

df = pd.read_parquet('repacts_with_summary.parquet')

df.to_csv('repacts_with_summary.txt', sep='\t', index=False)
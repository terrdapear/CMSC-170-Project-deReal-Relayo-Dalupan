import pandas as pd

df = pd.read_parquet('repacts_with_summary.parquet')

df.to_csv('output.txt', sep='\t', index=False)
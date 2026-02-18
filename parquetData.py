import pandas as pd

df = pd.read_parquet('notes_full.parquet')

print(f"Shape: {df.shape}")
print(f"\nColumn: {df.columns.tolist()}")
print(df.head())

topicCols =[col for col in df.columns if 'topic' in col.lower()]
print(f"Topic columns found: {topicCols}")

if topicCols:
    print(df[topicCols[0]].value_counts())
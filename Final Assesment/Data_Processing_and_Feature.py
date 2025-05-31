import pandas as pd

# Load the dataset
df = pd.read_csv("example_dataset.csv")

# Perform data preprocessing and feature engineering steps
# Add your code here
df_cleaned = df.dropna()

df_cleaned = df_cleaned.drop_duplicates()

df_cleaned['feature_sum'] = df_cleaned [['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']].sum(axis=1)

# Save the preprocessed dataset
df.to_csv("preprocessed_dataset.csv", index=False)

print("Original shape:", df.shape)
print("Cleaned shape:", df_cleaned.shape)
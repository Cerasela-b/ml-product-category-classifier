import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data/product.csv")

# Standardize column names

import re

def clean_column_name(col):
    col = col.strip()                     
    col = col.replace('_', ' ')           
    col = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', col)
    col = re.sub(r'\s+', ' ', col)       
    col = col.title()                     
    col = col.replace('Id', 'ID')         
    return col

df.columns = [clean_column_name(c) for c in df.columns]

# drop all rows with missing values
df = df.dropna()

# Define a mapping to standardize categories
category_mapping = {
    'fridge': 'Fridges',
    'Fridge Freezers': 'Fridge Freezers',
    'Fridges': 'Fridge Freezers',
    'Mobile Phone': 'Mobile Phones',
    'CPUs': 'CPUs',
    'CPU': 'CPUs'
}

# Apply the mapping
df['Category Label'] = df['Category Label'].replace(category_mapping)

# Strip extra spaces and capitalize consistently
df['Category Label'] = df['Category Label'].str.strip().str.title()

# Convert column type to 'category'
df['Category Label'] = df['Category Label'].astype('category')
print("Category Label type after converting:",df['Category Label'].dtype)
# Convert column type to 'category'
df['sentiment'] = df['sentiment'].astype('category')

# Keep only relevant columns
df_model = df[['Product Title', 'Category Label']].copy()
 
# Create the new feature: number of characters in each title
df_model['Product Title Length'] = df_model['Product Title'].str.len()

# Define features and label
X = df_model[['Product Title', 'Product Title Length']]
y = df_model['Category Label']
 
# Define preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "Product Title"),
        ("length", MinMaxScaler(), ["Product Title Length"])
    ]
)
# Define pipeline with the best model (e.g. RandomForestClassifier)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier())
])

# Train the model on the entire dataset
pipeline.fit(X, y)

# Save the model to a file
joblib.dump(pipeline, "model/category_classifier_model.pkl")

print("Model trained and saved as 'model/category_classifier_model.pkl'")
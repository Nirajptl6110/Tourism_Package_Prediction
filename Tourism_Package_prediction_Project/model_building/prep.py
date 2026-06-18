# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder # Keep for AgeGroup or other ordinal if needed
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/Niraj8767/Tourism-Package-Prediction-MLOPs/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop the unique identifier columns
df.drop(columns=['CustomerID', 'Unnamed: 0'], inplace=True)

# Correct spelling in 'Gender' column
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

# Merge 'Single' and 'Unmarried' in 'MaritalStatus' column
if 'MaritalStatus' in df.columns:
    df['MaritalStatus'] = df['MaritalStatus'].replace({'Unmarried': 'Single'})

# Convert Age to Age Groups (keeping as string labels for OneHotEncoder in train.py)
bins = [0, 18, 30, 45, 60, df['Age'].max()]
labels = ['<18', '18-30', '31-45', '46-60', '>60']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False).astype(str)
# Drop original 'Age' column
df.drop(columns=['Age'], inplace=True)

# No label encoding here for nominal categorical features, will be OneHotEncoded in train.py
# The following loop is removed as encoding will be done in train.py
# label_encoder = LabelEncoder()
# for column in ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']:
#     if column in df.columns:
#         df[column] = label_encoder.fit_transform(df[column])

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="Niraj8767/Tourism-Package-Prediction-MLOPs",
        repo_type="dataset",
    )

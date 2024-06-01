import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
#Load the cleaned dataset
df = pd.read_csv('/Users/zhengrongli/Downloads/cleaned_data.csv')
df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
# Inspect the dataset
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())
# Check the columns
print(df.columns)
#Ensure that the 'imdb_rating' column is numeric
df['Rating'] = pd.to_numeric(df['rating'], errors='coerce')
# Visualize the distribution of IMDb ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['rating'], bins=20, kde=True)
plt.title('Distribution of IMDb Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()
# Visualize relationships between key features and IMDb ratings
# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='year', y='rating', data=df)
plt.title('Ratings vs. Release Year')
plt.xlabel('Release Year')
plt.ylabel('Rating')
plt.show()
numeric_df = df.select_dtypes(include=[np.number])
#Drop rows with missing values that may have been introduced by coercion
numeric_df = numeric_df.dropna()
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
#for col in non_numeric_columns:
    #print(f"\nUnique values in non-numeric column '{col}':")
    #print(df[col].unique())

# Convert all non-numeric columns to numeric where possible
for col in non_numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
#Drop rows with any remaining missing values
numeric_df = numeric_df.dropna()
# Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
# Identify key features using correlation
correlations = df.corr()['rating'].sort_values(ascending=False)
# print(correlations)

# Assuming 'Rating' is the target variable
features = df.drop(columns=['genre', 'x', 'pg.rating', 'moive.name', 'cast', 'director', 'duration'])  # Dropping non-numeric and irrelevant columns
target = df['rating']

# Encode categorical variables
label_encoders = {}
for column in features.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column].astype(str))
    label_encoders[column] = le
# Identify and check columns with missing values
missing_values_columns = features.columns[features.isnull().any()].tolist()
print(f"Columns with missing values: {missing_values_columns}")

# Standardize numerical features
scaler = StandardScaler()
features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# Convert ratings to binary classification
threshold = 6.5  # Example threshold, adjust as needed
target_binary = np.where(target >= threshold, 1, 0)

# Split the dataset with binary target
X_train, X_test, y_train_binary, y_test_binary = train_test_split(features, target_binary, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
# Convert continuous predictions to binary for evaluation purposes
y_pred_lr_binary = np.where(y_pred_lr >= 5, 1, 0)
y_test_binary = np.where(y_test >= 5, 1, 0)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train_binary)
y_pred_dt = dt_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# Linear Regression Evaluation
lr_accuracy, lr_precision, lr_recall, lr_f1 = evaluate_model(y_test_binary, y_pred_lr_binary)

# Decision Tree Evaluation
dt_accuracy, dt_precision, dt_recall, dt_f1 = evaluate_model(y_test_binary, y_pred_dt)

# Compare models
model_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree'],
    'Accuracy': [lr_accuracy, dt_accuracy],
    'Precision': [lr_precision, dt_precision],
    'Recall': [lr_recall, dt_recall],
    'F1 Score': [lr_f1, dt_f1]
})

print(model_comparison)

# Print evaluation metrics for both models
print("Linear Regression Model Performance:")
print(f"Accuracy: {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"F1 Score: {lr_f1:.4f}")

print("\nDecision Tree Model Performance:")
print(f"Accuracy: {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall: {dt_recall:.4f}")
print(f"F1 Score: {dt_f1:.4f}")

# Select the best-performing model
best_model = 'Linear Regression' if lr_f1 > dt_f1 else 'Decision Tree'
print(f"\nThe best-performing model is: {best_model}")

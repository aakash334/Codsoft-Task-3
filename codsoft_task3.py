import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier # A common classification algorithm for Iris
from sklearn.svm import SVC # Support Vector Classifier, another powerful classification algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler # For scaling numerical data
from sklearn.datasets import load_iris # Scikit-learn has a built-in Iris dataset

# --- Step 1: Load the Dataset ---
# The Iris dataset is commonly available in scikit-learn, making it easy to load.
# Alternatively, you can load from a CSV if you have 'iris.csv' downloaded.
try:
    # Load Iris dataset directly from scikit-learn
    iris = load_iris(as_frame=True) # Load as a pandas DataFrame
    df = iris.frame
    # Rename target column for clarity and consistency
    df.rename(columns={'target': 'species'}, inplace=True)
    # Map numerical species to names for better readability in output
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    print("Iris dataset loaded successfully from scikit-learn!")
except Exception as e:
    print(f"Could not load Iris dataset from scikit-learn: {e}")
    print("Attempting to load from 'iris.csv'...")
    file_name = 'iris.csv' # Assuming the CSV has columns like sepal_length, sepal_width, petal_length, petal_width, species
    try:
        df = pd.read_csv(file_name)
        print(f"Dataset '{file_name}' loaded successfully!")
    except FileNotFoundError:
        print(f"Error: '{file_name}' not found. Please place 'iris.csv' in the correct directory.")
        exit()
    except Exception as e_csv:
        print(f"An error occurred while loading the dataset from CSV: {e_csv}")
        exit()

# --- Step 2: Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")
print("Data Head:\n", df.head())
print("\nData Info:\n")
df.info()
print("\nDescriptive Statistics:\n", df.describe(include='all'))
print("\nMissing values before handling:\n", df.isnull().sum())

# Visualize relationships between features and species
# Pairplot is excellent for visualizing relationships in small datasets like Iris
sns.pairplot(df, hue='species', palette='viridis')
plt.suptitle('Pairplot of Iris Features by Species', y=1.02) # Adjust title position
plt.show()

# Box plots for each feature by species
plt.figure(figsize=(15, 5))
for i, col in enumerate(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']):
    plt.subplot(1, 4, i + 1)
    sns.boxplot(x='species', y=col, data=df, palette='viridis')
    plt.title(f'{col} by Species')
plt.tight_layout()
plt.show()

# --- Step 3: Data Preprocessing ---
# The Iris dataset is generally very clean and does not have missing values.
# The target variable (species) is already categorical and needs to be encoded.

# No missing value handling typically required for Iris.
# If there were, steps would be similar to previous tasks (fillna, dropna).

# Convert target labels (species names) into numerical format
# This is necessary for models which expect numerical targets
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])
print(f"\nSpecies original names: {list(le.classes_)}")
print(f"Species encoded values: {list(range(len(le.classes_)))}")


# Drop original 'species' column as we'll use 'species_encoded' as target
df.drop('species', axis=1, inplace=True)

print("\nDataFrame head after preprocessing:\n", df.head())
print("\nDataFrame info after preprocessing:\n")
df.info()

# --- Step 4: Define Features (X) and Target (y) ---
# Features are the measurements, target is the encoded species.
X = df.drop('species_encoded', axis=1) # All columns except the encoded species
y = df['species_encoded']             # The encoded species column

print("\nFeatures (X) sample after preprocessing:\n", X.head())
print("\nTarget (y) sample after preprocessing:\n", y.head())

# --- Step 5: Split the Data into Training and Testing Sets ---
# 80% for training, 20% for testing. `random_state` ensures reproducibility.
# stratify=y ensures that the proportion of each class is the same in train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- Step 6: Feature Scaling ---
# Scaling is beneficial for K-Nearest Neighbors and SVMs which rely on distances.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrame for easier inspection (optional).
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

print("\nFeatures (X) sample after scaling:\n", X_train_scaled.head())


# --- Step 7: Choose and Train Machine Learning Models & Fine-tune hyperparameters ---
# We will train and tune K-Nearest Neighbors (KNN) and Support Vector Classifier (SVC).

# Model 1: K-Nearest Neighbors (KNN)
print("\n--- Training K-Nearest Neighbors (KNN) Classifier ---")
knn_model = KNeighborsClassifier()

# Define hyperparameters for GridSearchCV for KNN.
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9], # Number of neighbors to consider
    'weights': ['uniform', 'distance'], # Weight function used in prediction
    'metric': ['euclidean', 'manhattan'] # Distance metric
}

grid_search_knn = GridSearchCV(estimator=knn_model, param_grid=param_grid_knn,
                               cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search_knn.fit(X_train_scaled, y_train)

best_knn_model = grid_search_knn.best_estimator_
print(f"Best KNN Parameters: {grid_search_knn.best_params_}")
print(f"Best KNN Training Accuracy (Cross-validated): {grid_search_knn.best_score_:.4f}")

# Make predictions on the test set using the best KNN model.
y_pred_knn = best_knn_model.predict(X_test_scaled)

# Evaluate KNN Model performance.
print("\n--- KNN Model Evaluation ---")
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Test Accuracy: {accuracy_knn:.4f}")

print("\nClassification Report (KNN):")
print(classification_report(y_test, y_pred_knn, target_names=le.classes_)) # Use original class names

print("\nConfusion Matrix (KNN):")
cm_knn = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (KNN)')
plt.show()


# Model 2: Support Vector Classifier (SVC)
print("\n--- Training Support Vector Classifier (SVC) ---")
svc_model = SVC(random_state=42)

# Define hyperparameters for GridSearchCV for SVC.
param_grid_svc = {
    'C': [0.1, 1, 10], # Regularization parameter
    'kernel': ['linear', 'rbf'], # Specifies the kernel type
    'gamma': ['scale', 'auto'] # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
}

grid_search_svc = GridSearchCV(estimator=svc_model, param_grid=param_grid_svc,
                               cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search_svc.fit(X_train_scaled, y_train)

best_svc_model = grid_search_svc.best_estimator_
print(f"Best SVC Parameters: {grid_search_svc.best_params_}")
print(f"Best SVC Training Accuracy (Cross-validated): {grid_search_svc.best_score_:.4f}")

# Make predictions on the test set using the best SVC model.
y_pred_svc = best_svc_model.predict(X_test_scaled)

# Evaluate SVC Model performance.
print("\n--- SVC Model Evaluation ---")
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print(f"SVC Test Accuracy: {accuracy_svc:.4f}")

print("\nClassification Report (SVC):")
print(classification_report(y_test, y_pred_svc, target_names=le.classes_)) # Use original class names

print("\nConfusion Matrix (SVC):")
cm_svc = confusion_matrix(y_test, y_pred_svc)
sns.heatmap(cm_svc, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (SVC)')
plt.show()


# --- Step 8: Compare Evaluation Metrics of Various Classification Algorithms ---
print("\n--- Model Comparison Summary ---")
print(f"KNN Best Test Accuracy: {accuracy_knn:.4f}")
print(f"SVC Best Test Accuracy: {accuracy_svc:.4f}")

print("\nHigher accuracy indicates better model performance for classification.")
print("The model with the highest test accuracy is generally preferred.")

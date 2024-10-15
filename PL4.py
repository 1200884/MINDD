import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the dataset
df = pd.read_csv("wisc_bc_data.csv")
print(df.columns)

# Create boxplots for mean attributes by diagnosis
mean_attributes = df.filter(like='mean')

# Set the style for the plots
sns.set(style="whitegrid")

# Boxplots
plt.figure(figsize=(12, 6))
sns.boxplot(x="diagnosis", y="radius_mean", data=df, order=["M", "B"])
plt.title("Boxplot of radius_mean by Diagnosis")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="diagnosis", y="texture_mean", data=df, order=["M", "B"])
plt.title("Boxplot of texture_mean by Diagnosis")
plt.show()

# Histograms
plt.figure(figsize=(12, 6))
sns.histplot(df[df["diagnosis"] == "M"]["radius_mean"], kde=True, label="Malignant", color="red")
sns.histplot(df[df["diagnosis"] == "B"]["radius_mean"], kde=True, label="Benign", color="blue")
plt.title("Histogram of radius_mean by Diagnosis")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(df[df["diagnosis"] == "M"]["texture_mean"], kde=True, label="Malignant", color="red")
sns.histplot(df[df["diagnosis"] == "B"]["texture_mean"], kde=True, label="Benign", color="blue")
plt.title("Histogram of texture_mean by Diagnosis")
plt.legend()
plt.show()
X = df.drop("diagnosis", axis=1)  # Assuming 'diagnosis' is your target variable
y = df["diagnosis"]

# Step 2: Data Splitting
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
# i. Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# ii. k-Nearest Neighbors (KNN)
knn_model = KNeighborsClassifier(n_neighbors=5)  # Adjust 'n_neighbors' as needed
knn_model.fit(X_train, y_train)

# Step 4: Model Evaluation
# Naive Bayes evaluation
nb_predictions = nb_model.predict(X_test)
print("Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))
print("Naive Bayes Confusion Matrix:")
print(confusion_matrix(y_test, nb_predictions))

# KNN evaluation
knn_predictions = knn_model.predict(X_test)
print("K-Nearest Neighbors Classification Report:")
print(classification_report(y_test, knn_predictions))
print("K-Nearest Neighbors Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))
X = df.drop(columns=['id', 'diagnosis'])
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Optimize Naive Bayes parameters
param_grid_nb = {
    # Define the parameters and their possible values for Naive Bayes
}

nb_classifier = GaussianNB()
grid_search_nb = GridSearchCV(estimator=nb_classifier, param_grid=param_grid_nb, cv=5)
grid_search_nb.fit(X_train, y_train)

best_params_nb = grid_search_nb.best_params_
best_model_nb = grid_search_nb.best_estimator_

# Step 2: Optimize k-Nearest Neighbors parameters
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn_classifier = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn_classifier, param_grid=param_grid_knn, cv=5)
grid_search_knn.fit(X_train, y_train)

best_params_knn = grid_search_knn.best_params_
best_model_knn = grid_search_knn.best_estimator_

# Use the best models for predictions on the test data
y_pred_nb = best_model_nb.predict(X_test)
y_pred_knn = best_model_knn.predict(X_test)

# Evaluate the models
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("Accuracy of Naive Bayes: {:.2f}%".format(accuracy_nb * 100))
print("Accuracy of k-Nearest Neighbors: {:.2f}%".format(accuracy_knn * 100))
# Split the data into features (X) and the target (y)
X = df.drop(columns=['id', 'diagnosis'])
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Optimize Naive Bayes parameters
param_grid_nb = {
    # Define the parameters and their possible values for Naive Bayes
}

nb_classifier = GaussianNB()
grid_search_nb = GridSearchCV(estimator=nb_classifier, param_grid=param_grid_nb, cv=5)
grid_search_nb.fit(X_train, y_train)

best_model_nb = grid_search_nb.best_estimator_

# Step 2: Optimize k-Nearest Neighbors parameters
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn_classifier = KNeighborsClassifier()
grid_search_knn = GridSearchCV(estimator=knn_classifier, param_grid=param_grid_knn, cv=5)
grid_search_knn.fit(X_train, y_train)

best_model_knn = grid_search_knn.best_estimator_

# Step 3: Create a Voting Classifier
voting_classifier = VotingClassifier(estimators=[
    ('nb', best_model_nb),
    ('knn', best_model_knn)
], voting='hard')  # Use 'hard' for majority vote

# Fit the Voting Classifier on the training data
voting_classifier.fit(X_train, y_train)

# Predict with the Voting Classifier
y_pred_voting = voting_classifier.predict(X_test)

# Evaluate the Voting Classifier
accuracy_voting = accuracy_score(y_test, y_pred_voting)

print("Accuracy of Voting Classifier (Majority Vote): {:.2f}%".format(accuracy_voting * 100))

X = df.drop(columns=['id', 'diagnosis'])
y = df['diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the base models (optimized models)
nb_classifier = GaussianNB()
knn_classifier = KNeighborsClassifier()

# Create the meta-learner models (Logistic Regression and SVM)
logistic_reg = LogisticRegression()
svm_classifier = SVC()

# Create the Stacking Classifier
estimators = [
    ('nb', nb_classifier),
    ('knn', knn_classifier)
]

stacking_classifier = StackingClassifier(
    estimators=estimators,
    final_estimator=logistic_reg,  # Choose either logistic_reg or svm_classifier
)

# Fit the Stacking Classifier on the training data
stacking_classifier.fit(X_train, y_train)

# Predict with the Stacking Classifier
y_pred_stacking = stacking_classifier.predict(X_test)

# Evaluate the Stacking Classifier
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)

print("Accuracy of Stacking Classifier: {:.2f}%".format(accuracy_stacking * 100))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.map({'M': 1, 'B': 0})
y_test = y_test.map({'M': 1, 'B': 0})
# Exemplo de uso do BaggingClassifier
bagging_model = BaggingClassifier(n_estimators=100, random_state=42)
bagging_model.fit(X_train, y_train)
bagging_predictions = bagging_model.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_predictions)

# Exemplo de uso do RandomForestClassifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)
random_forest_predictions = random_forest_model.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)

# Exemplo de uso do AdaBoostClassifier
adaboost_model = AdaBoostClassifier(n_estimators=100, random_state=42)
adaboost_model.fit(X_train, y_train)
adaboost_predictions = adaboost_model.predict(X_test)
adaboost_accuracy = accuracy_score(y_test, adaboost_predictions)

# Exemplo de uso do GradientBoostingClassifier
gradient_boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gradient_boosting_model.fit(X_train, y_train)
gradient_boosting_predictions = gradient_boosting_model.predict(X_test)
gradient_boosting_accuracy = accuracy_score(y_test, gradient_boosting_predictions)

# Exemplo de uso do XGBoost
xgboost_model = XGBClassifier(n_estimators=100, random_state=42)
xgboost_model.fit(X_train, y_train)
xgboost_predictions = xgboost_model.predict(X_test)
xgboost_accuracy = accuracy_score(y_test, xgboost_predictions)

# Exemplo de uso do LightGBM
lightgbm_model = LGBMClassifier(n_estimators=100, random_state=42)
lightgbm_model.fit(X_train, y_train)
lightgbm_predictions = lightgbm_model.predict(X_test)
lightgbm_accuracy = accuracy_score(y_test, lightgbm_predictions)

# Agora você pode avaliar a precisão de cada modelo
print("Bagging Accuracy:", bagging_accuracy)
print("Random Forest Accuracy:", random_forest_accuracy)
print("AdaBoost Accuracy:", adaboost_accuracy)
print("Gradient Boosting Accuracy:", gradient_boosting_accuracy)
print("XGBoost Accuracy:", xgboost_accuracy)
print("LightGBM Accuracy:", lightgbm_accuracy)
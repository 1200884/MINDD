import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import chi2_contingency, f_oneway
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv("C:\\Users\\jgasp\\Downloads\\Churn_DataSet.csv", delimiter=',')
df = df.drop_duplicates()
print(df)
missing_data = df.isna().sum()
print("Column Names:")
print(df.columns)
scaler = MinMaxScaler()

# Print the count of missing values for each attribute
print("Count of Missing Values by Attribute:")
print(missing_data)
target_attribute = "Churn"

# Boxplots for numerical attributes by Churn status
numerical_attributes = df.select_dtypes(include=[np.number])
for column in numerical_attributes.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target_attribute, y=column, data=df)
    plt.title(f"Boxplot of {column} by {target_attribute}")
    #plt.show()

# Histograms for numerical attributes by Churn status
for column in numerical_attributes.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df, x=column, hue=target_attribute, kde=True)
    plt.title(f"Histogram of {column} by {target_attribute}")
    #plt.show()
    
categorical_attributes = df.select_dtypes(include=["object"])  # You may need to adjust the data types
numerical_attributes = df.select_dtypes(include=[np.number])
chi2_results = {}
for column in categorical_attributes.columns:
    crosstab = pd.crosstab(df[column], df[target_attribute])
    chi2, p, _, _ = chi2_contingency(crosstab)
    chi2_results[column] = p

#print("Chi-Square Test p-values:")
#print(chi2_results)
anova_results = {}
for column in numerical_attributes.columns:
    churn_yes = df[df[target_attribute] == "Yes"][column]
    churn_no = df[df[target_attribute] == "No"][column]
    _, p = f_oneway(churn_yes, churn_no)
    anova_results[column] = p

#print("ANOVA Test p-values:")
#print(anova_results)
df_encoded = df.copy()

# Mapeie a coluna 'Churn' para 1 e 0
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Use get_dummies apenas nas colunas categóricas
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns)

# Agora, defina as colunas numéricas que deseja normalizar e padronizar
numerical_attributes = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Normalização dos dados
scaler = MinMaxScaler()
df_encoded[numerical_attributes] = scaler.fit_transform(df_encoded[numerical_attributes])

# Divisão dos dados em atributos (X) e rótulos (y)
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Divisão dos dados com estratificação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Normalização e padronização dos dados apenas para as colunas numéricas
scaler = StandardScaler()
X_train_standardized = X_train.copy()
X_test_standardized = X_test.copy()
X_train_standardized[numerical_attributes] = scaler.fit_transform(X_train[numerical_attributes])
X_test_standardized[numerical_attributes] = scaler.transform(X_test[numerical_attributes])

# Verifique as distribuições em ambos os conjuntos
print("Distribuição de Churn no conjunto de treinamento:")
print(y_train.value_counts(normalize=True))

print("Distribuição de Churn no conjunto de teste:")
print(y_test.value_counts(normalize=True))
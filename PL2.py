import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
# Carregue o DataFrame com delimitador '\t'
df = pd.read_csv("C:\\Users\\jgasp\\Downloads\\week_purchases18-25.txt", delimiter='\t')

# Agora, você pode verificar os dados ausentes e plotar os gráficos corretamente
#print(df.size)
#print(df.describe())
numeric_columns = df.select_dtypes(include=['number'])

# Verifique quais colunas são categóricas
categorical_columns = df.select_dtypes(exclude=['number'])

#print("Numeric Columns:")
#print(numeric_columns.columns)

#print("\nCategorical Columns:")
#print(categorical_columns.columns)

missing_data_columns = df.isna().sum()

# Verifique dados ausentes por linhas
missing_data_rows = df.isna().sum(axis=1)

#print("Dados ausentes por colunas:")
#print(missing_data_columns > 0)

#print("\nDados ausentes por linhas:")
#print(missing_data_rows[missing_data_rows > 0])

sns.countplot(data=df, x='TitleCli')
plt.title('Distribution of Title Client')
plt.xticks(rotation=45)
#plt.show()

# Plot a bar chart for Country
sns.countplot(data=df, x='Country')
plt.title('Distribution of Country')
plt.xticks(rotation=90)
#plt.show()

#sns.countplot(data=df, x='Gender')
#plt.title('Distribution of Gender')
#plt.show()

#films_sold_by_style = df.groupby('Style')['Film_Title'].nunique()
#print(films_sold_by_style)
#unique_films_by_style = df.groupby('Style')['Film_Title'].unique().apply(len)
#print(unique_films_by_style)
#units_sold_by_film_style = df.groupby(['Film_Title', 'Style'])['Order_ID'].count()
#print(units_sold_by_film_style)

# Count the number of usernames with less than 5 Order_IDs
#usernames_with_less_than_5_orders = df.groupby('Username')['Order_ID'].nunique()
#usernames_count = (usernames_with_less_than_5_orders < 5).sum()
#print("Number of usernames with less than 5 Order_IDs:", usernames_count)

summary_table = df.groupby('Style').agg(
    Uni_Vnd=('Film_Title', 'nunique'),
    Min_Prc=('Price', 'min'),
    Max_Prc=('Price', 'max'),
    Avg_Prck=('Price', 'mean'),
    DesvPad_Prd=('Price', 'std')
)
print(summary_table)

df['PurchDate'] = pd.to_datetime(df['PurchDate'], dayfirst=True)
df['Day_of_Week_Purchase'] = df['PurchDate'].dt.strftime('%A')




def price_range(price):
    if price < 10:
        return "low"
    elif 10 <= price < 50:
        return "medium"
    else:
        return "high"
df['Price_Range'] = pd.cut(df['Price'], bins=[0, 6, 10, float('inf')], labels=['low', 'medium', 'high'])

df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 64, float('inf')], labels=['young','middle-aged','elderly'])

def age_group(age):
    if age < 30:
        return "young"
    elif 30 <= age < 60:
        return "middle-aged"
    else:
        return "elderly"

df['Age_Group'] = df['Age'].apply(age_group)

sns.countplot(data=df, x='Day_of_Week_Purchase')
plt.title('Purchases by Day of the Week')
plt.xticks(rotation=45)
#plt.show()


df['PurchTime'] = pd.to_datetime(df['PurchTime'], format='%H:%M:%S')

df['PurchHour'] = df['PurchTime'].dt.hour

purchases_by_hour = df.groupby('PurchHour').size()

plt.figure(figsize=(10, 6))
plt.bar(purchases_by_hour.index, purchases_by_hour.values, color='skyblue')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Purchases')
plt.title('Purchases by Hour of the Day')
plt.xticks(range(24))
#plt.show()


pivot_table = df.pivot_table(index='Style', columns='Price_Range', aggfunc='size', fill_value=0)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
plt.xlabel('Price Range')
plt.ylabel('Style')
plt.title('Price Range Distribution by Style')
#plt.show()

pivot_table = df.pivot_table(index='Style', columns='Age_Group', aggfunc='size', fill_value=0)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
plt.xlabel('Age Group')
plt.ylabel('Style')
plt.title('Age Group Distribution by Style')
#plt.show()

pivot_table = df.pivot_table(index='Price', columns='Age_Group', aggfunc='size', fill_value=0)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
plt.xlabel('Age Group')
plt.ylabel('Price')
plt.title('Age Group Distribution by Price')
plt.show()

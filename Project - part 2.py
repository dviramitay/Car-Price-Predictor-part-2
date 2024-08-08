#!/usr/bin/env python
# coding: utf-8

# ## dvir amitay  


# In[127]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


file_path = r'C:\Users\dvir\Desktop\שנה ג\סמסטר ב\כרייה וניתוח נתונים\מטלות\מטלה 2\dataset.csv'
df = pd.read_csv(file_path)

df


# In[128]:


df.isnull().sum()


# ## Filling missing values 
# Filling missing values in the table helps improve the RMSE of the model in the following ways:
# 
# Increased Data Utilization: By filling missing values, we ensure that more data points are used in training the model. This helps the model to learn better from the available data.
# 
# Reduced Bias: Missing values can introduce bias in the model if not handled properly. Filling these values appropriately reduces this bias.
# 
# Improved Data Quality: Filling missing values improves the overall quality of the dataset, making the training process more reliable and the model more robust.
# 
# Consistent Predictions: Handling missing values ensures that the model can make consistent predictions even when new data contains missing values.
# 
# ### Not all columns are essential and relevant to the accuracy of the model. We will only fill in the relevant columns

# In[129]:


def fill_missing_values(df):
    # Km according to how many years the car has been on the road *15000 (annual average in Israel)
    df['Km'] = df.apply(lambda row: (2024 - row['Year']) * 15000 if pd.isna(row['Km']) else row['Km'], axis=1)
    
    # For capacity Engine , we will find all vehicles of the same model and fill in the most common value. If there are no more vehicles of this model, we will fill in "0"
    def mode(series):
        return series.mode().iloc[0] if not series.mode().empty else np.nan
    
    df['capacity_Engine'] = df.groupby(['manufactor', 'Year', 'model'])['capacity_Engine'].transform(lambda x: x.fillna(mode(x)))
    df['capacity_Engine'] = df.groupby('model')['capacity_Engine'].transform(lambda x: x.fillna(mode(x)))
    df['capacity_Engine'].fillna('0', inplace=True)
    df['capacity_Engine'] = df['capacity_Engine'].astype(str).str.replace(',', '').str.replace(' ', '')
    df['capacity_Engine'] = df['capacity_Engine'].astype(int)
    
    #In the past, it was known that most vehicles were produced with a manual transmission and over the years an automatic transmission was used. Therefore, we will check which gear was the most common in that year and fill in the values ​​accordingly
    mode_gear_per_year = df.groupby('Year')['Gear'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown')
    def fill_missing_gear(row):
        if pd.isnull(row['Gear']):
            return mode_gear_per_year[row['Year']]
        else:
            return row['Gear']
    df['Gear'] = df.apply(fill_missing_gear, axis=1)
    
    df['Engine_type'].fillna('Unknown', inplace=True)
    
    df['Curr_ownership'].fillna('Unknown', inplace=True)
    
    df['Color'].fillna('Unknown', inplace=True)
    
    return df


# ## create new columns
# Creating New Columns Improves RMSE Accuracy:
# 
# Enriching the Information: New columns can add new and relevant information that wasn't available before or create new combinations of existing information. This can help the model to better understand the relationships between different variables and the target variable (in this case, the price).
# 
# Improving Model Performance: Through Feature Engineering, we can create variables that better represent the dynamics of the problem. Such variables can improve the model's ability to predict more accurately, thereby reducing the error.
# 
# Creating new columns through Feature Engineering can significantly improve the predictive power of the model, leading to a lower RMSE and better overall performance.

# In[130]:


def create_new_columns(df):
    # create a new column in order to accurately determine for the model how much the vehicle was used
    df['Car_age'] = 2024 - df['Year']    
    df['Km'] = df['Km'].astype(str).str.replace(',', '').str.replace(' ', '')
    df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
    df['Km'] = df.apply(lambda row: (2024 - row['Year']) * 15000 if pd.isna(row['Km']) else row['Km'], axis=1)
    df['Km'] = df['Km'].astype(int)
    df.loc[(df['Km'] < 1000) & (df['Car_age'] > 1), 'Km'] *= 1000
    df['Average_km_per_year'] = df['Km'] / df['Car_age']
    df.loc[df['Car_age'] == 0, 'Average_km_per_year'] = 0
    df['Average_km_per_year'] = df['Average_km_per_year'].round().astype(int)
    
    #create a new column that classifies the car's color by its popularity
    color_counts = df['Color'].value_counts()
    color_popularity_mapping = {color: rank for rank, color in enumerate(color_counts.index, start=1)}
    df['Color_Popularity'] = df['Color'].map(color_popularity_mapping)
    
    return df


# ## Columns to Consider Excluding from RMSE Calculation
# When building a price prediction model, it's important to select only the columns that provide relevant and meaningful information. Some columns might be less relevant or even detrimental to the model's performance
# 
# Columns Not Directly Influencing Price:
# Pic_num (Number of Pictures): The number of pictures in the listing is unlikely to directly affect the car's price.
# Cre_date (Creation Date) and Repub_date (Re-publication Date): These dates refer to the time of listing and not to the car's characteristics. They might not be relevant to the price prediction model.
# Description: A free-text description of the car, which is difficult to convert into a numerical feature efficiently. While text analysis can be useful, it requires advanced processing beyond basic analysis.
# Area and City: If the information about the area or city does not significantly improve the model, it can be excluded. Alternatively, this information can be combined into one column or grouped into broader regions if necessary.
# Prev_ownership (Previous Ownership) and Curr_ownership (Current Ownership): These data points can be relevant in some cases, but if they do not significantly impact the price or if there is a problem with data quality, they can be considered for exclusion
# 

# In[131]:


df_encoded = pd.get_dummies(df, drop_first=True)
correlation_matrix = df.corr()

# Displaying the correlation matrix
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# ## Arrangement and final preparation of the data

# In[132]:


def arrange_data(df):
    df = fill_missing_values(df)
    df = create_new_columns(df)
    
    columns_to_keep = ['Hand', 'capacity_Engine', 'Km', 'Car_age', 'Average_km_per_year', 'Color_Popularity', 'Price']
    df = df.filter(columns_to_keep + ['manufactor', 'model', 'Gear', 'Engine_type', 'Curr_ownership', 'Color'])
    
    df = pd.get_dummies(df, columns=['manufactor', 'model', 'Gear', 'Engine_type','Curr_ownership',"Color"], drop_first=True)
    
    return df


# ## Building the model

# In[143]:


def perform_elastic_net_with_cv(df, param_grid):
    X = df.drop(columns=['Price'])
    y = df['Price']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    elastic_net = ElasticNet(random_state=42)
    grid_search = GridSearchCV(elastic_net, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)
    
    best_model = grid_search.best_estimator_
   #best_params = grid_search.best_params_
    
    neg_mse_scores = cross_val_score(best_model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    
    mse_scores = -neg_mse_scores
    rmse_scores = np.sqrt(mse_scores)
    mean_rmse = np.mean(rmse_scores)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {mean_rmse}")
    
    return best_model


# In[144]:


def prepare_data(df):
    df = arrange_data(df)
    param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],'l1_ratio': [0.1, 0.5, 0.7, 0.9]}
    model = perform_elastic_net_with_cv(df, param_grid)
    return df, model


# In[145]:


df_prepared, model = prepare_data(df)


# In[ ]:





# In[146]:


def analyze_model_performance(df, model):
    X = df.drop(columns=['Price'])
    
    # Feature importance analysis
    feature_importances = model.coef_
    effect_type_list = ['+' if coef >= 0 else '-' for coef in feature_importances]
    
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': abs(feature_importances),
        'Effect_Type': effect_type_list
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df.head(5))  # Print top 5 most important features
    
    # Top 5 unique features by importance
    top_5_unique_features = importance_df.Feature.str.split('_').apply(lambda x: x[0]).unique()[:5]
    print(f"Top 5 unique features by importance: {top_5_unique_features}")
    
    return importance_df


# In[147]:


importance_df = analyze_model_performance(df_prepared, model)


# In[ ]:





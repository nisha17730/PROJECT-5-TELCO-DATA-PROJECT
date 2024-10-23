#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().system('pip install scikit-learn')
from sklearn.impute import SimpleImputer


# In[3]:


df = pd.read_csv('telcom_data.csv')


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


# Missing value
df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


for col in df.columns:
    print(df[col].value_counts())


# In[10]:


plt.figure(figsize=(18,10))
sns.heatmap(df.isnull())
plt.show()


# In[11]:


df.info()


# ### Top 10 handsets,Top 3 Handset Manufacturers and Top 5 Handsets per Manufacturer 

# In[12]:


# Identifying the top 10 handsets used by customers
top_10_handsets = df['Handset Type'].value_counts().head(10)
print("Top 10 Handsets:")
print(top_10_handsets)

# Identifying the top 3 handset manufacturers
top_3_manufacturers = df['Handset Manufacturer'].value_counts().head(3)
print("\nTop 3 Handset Manufacturers:")
print(top_3_manufacturers)

# Identifying the top 5 handsets per top 3 handset manufacturers
top_3_manufacturers_handsets = df[df['Handset Manufacturer'].isin(top_3_manufacturers.index)]
top_5_handsets_per_manufacturer = top_3_manufacturers_handsets.groupby('Handset Manufacturer')['Handset Type'].value_counts().groupby(level=0).head(5)
print("\nTop 5 Handsets per Manufacturer:")
print(top_5_handsets_per_manufacturer)

# Interpretation and recommendation to marketing teams
print("\nInterpretation and Recommendation:")
print("The top 10 handsets used by customers give an idea of the popular devices among the users.")
print("The top 3 handset manufacturers show the dominant players in the market.")
print("The top 5 handsets per manufacturer provide insights into the popular devices from the leading manufacturers.")
print("Based on these findings, the marketing teams can focus their efforts on promoting the popular handsets and collaborating with the top manufacturers to create targeted marketing campaigns.")


# ### Task 1.1: Aggregate user behavior information:
# 
# #### Group the data by user and calculate the following metrics: number of xDR sessions, session duration, total download and upload data, and total data volume for each application.
# #### Create a new DataFrame with the aggregated information.

# In[13]:


user_behavior = df.groupby('MSISDN/Number').agg({
    'Dur. (ms)': 'count',
    'Start ms': 'min',
    'End ms': 'max',
    'Total DL (Bytes)': 'sum',
    'Total UL (Bytes)': 'sum',
    'Social Media DL (Bytes)': 'sum',
    'Social Media UL (Bytes)': 'sum',
    'Google DL (Bytes)': 'sum',
    'Google UL (Bytes)': 'sum',
    'Email DL (Bytes)': 'sum',
    'Email UL (Bytes)': 'sum',
    'Youtube DL (Bytes)': 'sum',
    'Youtube UL (Bytes)': 'sum',
    'Netflix DL (Bytes)': 'sum',
    'Netflix UL (Bytes)': 'sum',
    'Gaming DL (Bytes)': 'sum',
    'Gaming UL (Bytes)': 'sum',
    'Other DL (Bytes)' : 'sum',
    'Other UL (Bytes)': 'sum'
}).reset_index()

# Display the resulting user behavior dataframe
print(user_behavior)


# In[14]:


# Select the columns for box plot
Bar_metrics= [
"Social Media DL (Bytes)", "Social Media UL (Bytes)", "Google DL (Bytes)", "Google UL (Bytes)",
"Email DL (Bytes)", "Email UL (Bytes)", "Youtube DL (Bytes)", "Youtube UL (Bytes)", "Netflix DL (Bytes)",    
"Netflix UL (Bytes)","Gaming DL (Bytes)", "Gaming UL (Bytes)", "Other DL (Bytes)" , "Other UL (Bytes)" ]

values = user_behavior[Bar_metrics].sum()

# Create the bar chart
plt.bar(Bar_metrics, values)

# Set the plot title
plt.title('User Behavior - Metrics')

# Set the x-axis label
plt.xlabel('Metrics')

# Set the y-axis label
plt.ylabel('Values')

# Rotate the x-axis labels for better readability 
plt.xticks(rotation=45)

# Display the bar chart
plt.show()


# In[15]:


# Calculate the total DL data for each category
dl_data = [
    user_behavior["Social Media DL (Bytes)"].sum(),
    user_behavior["Google DL (Bytes)"].sum(),
    user_behavior["Email DL (Bytes)"].sum(),
    user_behavior["Youtube DL (Bytes)"].sum(),
    user_behavior["Netflix DL (Bytes)"].sum(),
    user_behavior["Gaming DL (Bytes)"].sum(),
    user_behavior["Other DL (Bytes)"].sum()
]

# Calculate the total UL data for each category
ul_data = [
    user_behavior["Social Media UL (Bytes)"].sum(),
    user_behavior["Google UL (Bytes)"].sum(),
    user_behavior["Email UL (Bytes)"].sum(),
    user_behavior["Youtube UL (Bytes)"].sum(),
    user_behavior["Netflix UL (Bytes)"].sum(),
    user_behavior["Gaming UL (Bytes)"].sum(),
    user_behavior["Other UL (Bytes)"].sum()
]

# Create labels for each category
labels = ["Social Media", "Google", "Email", "Youtube", "Netflix", "Gaming", "Other"]

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the DL pie chart
ax1.pie(dl_data, labels=labels, autopct="%1.1f%%", startangle=90)
ax1.set_title("Total DL Consumption by Category")

# Plot the UL pie chart
ax2.pie(ul_data, labels=labels, autopct="%1.1f%%", startangle=90)
ax2.set_title("Total UL Consumption by Category")

# Set aspect ratio to be equal so that pie is drawn as a circle
ax1.axis("equal")
ax2.axis("equal")

# Add legends to the subplots
ax1.legend(loc="lower right")
ax2.legend(loc="lower right")

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4)

# Display the figure
plt.show()


# ## Task 1.2: Exploratory data analysis

# In[16]:


# Analyze basic metrics
metrics = ['Dur. (ms)','Start ms', 'End ms','Total DL (Bytes)', 'Total UL (Bytes)']
basic_stats = user_behavior[metrics].describe()
print(basic_stats)


# In[17]:


# Non-Graphical Univariate Analysis
dispersion_params = user_behavior[metrics].std()
print(dispersion_params)


# In[18]:


# Graphical Univariate Analysis
user_behavior.hist(column=metrics, bins=10, figsize=(10, 8))
plt.ylabel("Frequency")
plt.show()


# In[19]:


user_behavior.hist(bins=10, figsize=(12, 8))
plt.ylabel("Frequency")
# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Increase the values to increase the distance

# Show the figure
plt.show()


# In[20]:


import pandas as pd

# Assuming user_behavior is your DataFrame and it's already defined
bins = [0, 500, 1000, 1500, 2000, float('inf')]  # Example bins
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Create a new column for data usage categories
user_behavior['Data Usage Category'] = pd.cut(
    user_behavior['Total DL (Bytes)'] + user_behavior['Total UL (Bytes)'],
    bins=bins,
    labels=labels
)

# Compute the total data usage per category
data_usage_per_category = user_behavior.groupby('Data Usage Category')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum()

# Print the results
print(data_usage_per_category)
# Plot the data usage per category
data_usage_per_category.plot(kind='bar')
plt.xlabel('Data Usage Category')
plt.ylabel('Total Data Usage')
plt.title('Total Data Usage per Category')
plt.show()


# ### Correlation

# In[21]:


corr_matrix = user_behavior[["Social Media DL (Bytes)", "Social Media UL (Bytes)", "Google DL (Bytes)", "Google UL (Bytes)",
"Email DL (Bytes)", "Email UL (Bytes)", "Youtube DL (Bytes)", "Youtube UL (Bytes)", "Netflix DL (Bytes)",    
"Netflix UL (Bytes)","Gaming DL (Bytes)", "Gaming UL (Bytes)", "Other DL (Bytes)" , "Other UL (Bytes)"]].corr()
print(corr_matrix)


# In[22]:


plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# In[23]:


# Calculate the total duration for all sessions for each user
user_behavior['Total Duration'] = user_behavior['Dur. (ms)'].sum().level=0()

# Create deciles based on the total duration
user_behavior['Total Duration Decile'] = pd.qcut(user_behavior['Total Duration'], q=10, duplicates='drop')

# Group the data by decile class and calculate the sum of total data
data_usage_per_decile = user_behavior.groupby('Total Duration Decile')[['Total DL (Bytes)', 'Total UL (Bytes)']].sum()

# Print the data usage per decile class
print(data_usage_per_decile)


# In[ ]:


# Dimensionality Reduction
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
nishapal_components = pca.fit_transform(user_behavior[["Social Media DL (Bytes)", "Social Media UL (Bytes)", "Google DL (Bytes)", "Google UL (Bytes)",
"Email DL (Bytes)", "Email UL (Bytes)", "Youtube DL (Bytes)", "Youtube UL (Bytes)", "Netflix DL (Bytes)",    
"Netflix UL (Bytes)","Gaming DL (Bytes)", "Gaming UL (Bytes)", "Other DL (Bytes)" , "Other UL (Bytes)"]])
nishapal_df = pd.DataFrame(data=nishapal_components, columns=['PC1', 'PC2'])
print(nishapal_df.head())


# ##### PC1: The values in PC1 column represent the scores of each data point along the first nishapal component. PC1 captures the direction in the data that has the highest variance. In this case, the values range from -5.594646e+08 to 6.448334e+08. Positive values indicate that the corresponding data points have a higher projection along PC1, while negative values indicate a lower projection.
# 
# ##### PC2: The values in PC2 column represent the scores of each data point along the second nishapal component. PC2 is orthogonal (uncorrelated) to PC1 and captures the second highest variance in the data. The values range from -1.239993e+08 to 3.745330e+08. Similarly, positive values indicate a higher projection along PC2, while negative values indicate a lower projection.

# In[ ]:


# Create DataFrame for nishapal components
nishapal_df = pd.DataFrame(data=nishapal_components, columns=['PC1', 'PC2'])

# Variance explained by each nishapal component
explained_variance = pca.explained_variance_ratio_
print("Variance Explained by PC1:", explained_variance[0])
print("Variance Explained by PC2:", explained_variance[1])

# Scatter plot of nishapal components
plt.scatter(nishapal_df['PC1'], nishapal_df['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: User Behavior')
plt.show()


# In[ ]:


# Interpretation of Loadings
loadings = pca.components_.T  # Transpose to match variables with loadings
variables = ["Social Media DL", "Social Media UL", "Google DL", "Google UL", "Email DL", "Email UL",
             "Youtube DL", "Youtube UL", "Netflix DL", "Netflix UL", "Gaming DL", "Gaming UL",
             "Other DL", "Other UL"]
loadings_df = pd.DataFrame(data=loadings, columns=['PC1', 'PC2'], index=variables)
print("Loadings:")
print(loadings_df)


# In[ ]:


# Set the figure size
plt.figure(figsize=(6, 4))

# Plot the loadings
plt.bar(range(len(loadings_df)), loadings_df['PC1'], alpha=1, label='PC1')
plt.bar(range(len(loadings_df)), loadings_df['PC2'], alpha=1, label='PC2')

# Set the x-axis ticks and labels
plt.xticks(range(len(loadings_df)), loadings_df.index, rotation=90)

# Set the y-axis label
plt.ylabel('Loadings')

# Set the title of the plot
plt.title('nishapal Component Loadings')

# Add a legend
plt.legend()

# Show the plot
plt.show()


# in PC1, the variables "Social Media DL," "Social Media UL," "Google DL," "Google UL," "Email DL," "Email UL," "Youtube DL," "Youtube UL," "Netflix DL," and "Netflix UL" have relatively small loadings (close to zero), suggesting that they have less influence on the construction of PC1. On the other hand, the variables "Gaming DL" and "Other DL" have high positive loadings (0.707074 and 0.706386, respectively), indicating a strong positive relationship with PC1. This suggests that "Gaming DL" and "Other DL" contribute significantly to the variation captured by PC1.
# 
# Similarly, in PC2, all variables except "Gaming DL" and "Other DL" have small loadings close to zero, indicating less influence on the construction of PC2. "Gaming DL" and "Other DL" have loadings of 0.707661 and -0.707472, respectively, suggesting a strong negative relationship with PC2. This implies that "Gaming DL" and "Other DL" contribute significantly to the variation captured by PC2 and are negatively correlated with each other in PC2.

# In[ ]:





# In[ ]:





# In[ ]:





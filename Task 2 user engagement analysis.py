#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Read and clean data

# In[2]:


df = pd.read_csv('telcom_data.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[115]:


# Handle missing values
numeric_df = df.select_dtypes(include=[float, int])
df.fillna(numeric_df.mean(), inplace=True)


# In[116]:


df['session_frequency'] = df.groupby(by=['MSISDN/Number'])['Dur. (ms)'].transform('count')


# In[117]:


df['Session_Duration'] = df['Dur. (ms)']


# In[118]:


df['total_traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']


# In[119]:


df


# In[120]:


df['session_frequency']


# In[121]:


df['session_frequency'].plot(kind='hist', bins=10)

plt.xlabel('Session Frequency')
plt.ylabel('Count')
plt.title('Distribution of Session Frequency')
plt.show()


# In[122]:


df['Session_Duration'].plot(kind='hist', bins=10)

plt.xlabel('Session Duration')
plt.ylabel('Count')
plt.title('Distribution of Session Durations')
plt.show()


# In[123]:


df['total_traffic'].plot(kind='hist', bins=10)

plt.xlabel('Total Traffic')
plt.ylabel('Count')
plt.title('Distribution of Total Traffic')
plt.show()


# In[124]:


# Aggregate engagement metrics per customer id (MSISDN)
engagement_metrics = df.groupby('MSISDN/Number').agg({
    'session_frequency': 'sum',
    'Session_Duration': 'sum',
    'total_traffic': 'sum',
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
    'Other DL (Bytes)': 'sum',
    'Other UL (Bytes)': 'sum'
}).reset_index()


# In[125]:


# Report the top 10 customers per engagement metric
top_10_sessions_frequency = engagement_metrics.nlargest(10, 'session_frequency')
top_10_session_duration = engagement_metrics.nlargest(10, 'Session_Duration')
top_10_total_traffic = engagement_metrics.nlargest(10, 'total_traffic')
top_10_social_media = engagement_metrics.nlargest(10, 'Social Media DL (Bytes)')
top_10_google = engagement_metrics.nlargest(10, 'Google DL (Bytes)')
top_10_email = engagement_metrics.nlargest(10, 'Email DL (Bytes)')
top_10_youtube = engagement_metrics.nlargest(10, 'Youtube DL (Bytes)')
top_10_netflix = engagement_metrics.nlargest(10, 'Netflix DL (Bytes)')
top_10_gaming = engagement_metrics.nlargest(10, 'Gaming DL (Bytes)')
top_10_other = engagement_metrics.nlargest(10, 'Other DL (Bytes)')


# In[126]:


top_10_sessions_frequency


# In[127]:


# Group the data by 'Handset Type' and 'MSISDN/Number' and count the number of unique sessions
session_frequency = df.groupby(['Handset Type', 'MSISDN/Number'])['Dur. (ms)'].count().reset_index()

# Group the data by 'Handset Type' and calculate the total session frequency
type_frequency = session_frequency.groupby('Handset Type')['Dur. (ms)'].sum().reset_index()

# Sort the data by session frequency in descending order
type_frequency = type_frequency.sort_values('Dur. (ms)', ascending=False).head(10)

# Plot the session frequency by handset type
plt.figure(figsize=(10, 6))
sns.barplot(data=type_frequency, x='Handset Type', y='Dur. (ms)')
plt.xlabel('Handset Type')
plt.ylabel('Session Frequency')
plt.title('Top 10 Handset Types by Session Frequency')
plt.xticks(rotation=90)
plt.show()


# In[128]:


# Group the data by 'Handset Manufacturer' and 'MSISDN/Number' and count the number of unique sessions
session_frequency1 = df.groupby(['Handset Manufacturer', 'MSISDN/Number'])['Dur. (ms)'].count().reset_index()

# Group the data by 'Handset Manufacturer' and calculate the total session frequency
manufacturer_frequency = session_frequency1.groupby('Handset Manufacturer')['Dur. (ms)'].sum().reset_index()

# Sort the data by session frequency in descending order
manufacturer_frequency = manufacturer_frequency.sort_values('Dur. (ms)', ascending=False).head(10)

# Plot the session frequency by handset manufacturer
plt.figure(figsize=(10, 6))
sns.barplot(data=manufacturer_frequency, x='Handset Manufacturer', y='Dur. (ms)')
plt.xlabel('Handset Manufacturer')
plt.ylabel('Session Frequency')
plt.title('Top 10 Handset Manufacturers by Session Frequency')
plt.xticks(rotation=90)
plt.show()


# In[129]:


# Top 10 session duration
plt.figure(figsize=(6, 6))
plt.pie(top_10_session_duration['Session_Duration'], labels=top_10_session_duration['MSISDN/Number'], autopct='%1.1f%%')
plt.title('Top 10 Customers by Session Duration')
plt.show()


# In[130]:


# Top 10 total traffic
plt.figure(figsize=(6, 6))
plt.pie(top_10_total_traffic['total_traffic'], labels=top_10_total_traffic['MSISDN/Number'], autopct='%1.1f%%')
plt.title('Top 10 Customers by Total Traffic')
plt.show()


# In[131]:


# Top 10 Social Media DL (Bytes)
plt.figure(figsize=(6, 6))
plt.pie(top_10_social_media['Social Media DL (Bytes)'], labels=top_10_social_media['MSISDN/Number'], autopct='%1.1f%%')
plt.title('Top 10 Customers by Social Media DL (Bytes)')
plt.show()


# In[132]:


# Top 10 Google DL (Bytes)
plt.figure(figsize=(6, 6))
plt.pie(top_10_google['Google DL (Bytes)'], labels=top_10_google['MSISDN/Number'], autopct='%1.1f%%')
plt.title('Top 10 Customers by Google DL (Bytes)')
plt.show()


# In[133]:


# Top 10 Email DL (Bytes)
plt.figure(figsize=(6, 6))
plt.pie(top_10_email['Email DL (Bytes)'], labels=top_10_email['MSISDN/Number'], autopct='%1.1f%%')
plt.title('Top 10 Customers by Email DL (Bytes)')
plt.show()


# In[134]:


# Top 10 Email DL (Bytes)
plt.figure(figsize=(6, 6))
plt.pie(top_10_email['Email DL (Bytes)'], labels=top_10_email['MSISDN/Number'], autopct='%1.1f%%')
plt.title('Top 10 Customers by Email DL (Bytes)')
plt.show()


# In[135]:


# Top 10 Netflix DL (Bytes)
plt.figure(figsize=(6, 6))
plt.pie(top_10_netflix['Netflix DL (Bytes)'], labels=top_10_netflix['MSISDN/Number'], autopct='%1.1f%%')
plt.title('Top 10 Customers by Netflix DL (Bytes)')
plt.show()


# In[136]:


# Top 10 Gaming DL (Bytes)
plt.figure(figsize=(6, 6))
plt.pie(top_10_gaming['Gaming DL (Bytes)'], labels=top_10_gaming['MSISDN/Number'], autopct='%1.1f%%')
plt.title('Top 10 Customers by Gaming DL (Bytes)')
plt.show()


# ## Task 2.1

# In[137]:


# Normalize engagement metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
engagement_metrics_normalized = scaler.fit_transform(engagement_metrics[['session_frequency', 'Session_Duration', 'total_traffic',
                                                                         'Social Media DL (Bytes)',
                                                                         'Social Media UL (Bytes)',
                                                                         'Google DL (Bytes)',
                                                                         'Google UL (Bytes)',
                                                                         'Email DL (Bytes)',
                                                                         'Email UL (Bytes)',
                                                                         'Youtube DL (Bytes)',
                                                                         'Youtube UL (Bytes)',
                                                                         'Netflix DL (Bytes)',
                                                                         'Netflix UL (Bytes)',
                                                                         'Gaming DL (Bytes)',
                                                                         'Gaming UL (Bytes)',
                                                                         'Other DL (Bytes)',
                                                                         'Other UL (Bytes)']])


# In[138]:


# Run k-means clustering (k=3) on normalized engagement metrics
kmeans = KMeans(n_clusters=3, random_state=0)
engagement_clusters = kmeans.fit_predict(engagement_metrics_normalized)


# In[139]:


# Add engagement clusters to the DataFrame
engagement_metrics['engagement_cluster'] = engagement_clusters


# In[140]:


cluster_metrics = engagement_metrics.groupby('engagement_cluster').agg({
    'session_frequency': ['min', 'max', 'mean', 'sum'],
    'Session_Duration': ['min', 'max', 'mean', 'sum'],
    'total_traffic': ['min', 'max', 'mean', 'sum']})


# In[141]:


cluster_metrics


# In[142]:


# Assign cluster labels to data
customer_clusters = kmeans.labels_

# Add cluster labels to the aggregate dataframe
engagement_metrics['Cluster'] = customer_clusters

# Print the number of customers in each cluster
print(engagement_metrics['Cluster'].value_counts())


# In[143]:


for cluster in range(3):
    print(f"\nTop 10 customers in Cluster {cluster}:")
    top_10_customers = engagement_metrics[engagement_metrics['Cluster'] == cluster].nlargest(10, 'Session_Duration')
    print(top_10_customers[['session_frequency', 'Session_Duration', 'total_traffic']])

x_metric = 'session_frequency'
y_metric = 'Session_Duration'

# Create a scatter plot of the clusters
plt.figure(figsize=(10, 6))
for cluster in range(3):
    cluster_data = engagement_metrics[engagement_metrics['Cluster'] == cluster]
    plt.scatter(cluster_data[x_metric], cluster_data[y_metric], label=f'Cluster {cluster}',alpha=0.5)
plt.xlabel(x_metric)
plt.ylabel(y_metric)
plt.title('Clustering Analysis')
plt.legend()
plt.show()


# In[144]:


cluster_metrics


# In[145]:


# Plot the non-normalized metrics for each cluster
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for i, metric in enumerate(['session_frequency', 'Session_Duration', 'total_traffic']):
    ax = axes[i]
    cluster_data = cluster_metrics[metric]
    cluster_labels = cluster_data.index

    ax.bar(cluster_labels, cluster_data['min'], label='Min')
    ax.bar(cluster_labels, cluster_data['max'], label='Max')
    ax.bar(cluster_labels, cluster_data['mean'], label='Mean')
    ax.bar(cluster_labels, cluster_data['sum'], label='Sum')

    ax.set_xlabel('Engagement Cluster')
    ax.set_ylabel(metric)
    ax.set_title(f'Cluster Metrics: {metric}')
    ax.legend()


# In[146]:


print (cluster_labels)


# In[147]:


plt.figure(figsize=(10, 6))
for cluster in range(3):
    cluster_data = engagement_metrics[engagement_metrics['Cluster'] == cluster]
    plt.scatter(cluster_data[x_metric], cluster_data[y_metric], label=f'Cluster {cluster}',alpha=0.5)
plt.xlabel(x_metric)
plt.ylabel(y_metric)
plt.title('Clustering Analysis')
plt.legend()
plt.show()


# In[148]:


# Plot the top 3 most used applications
top_3_applications = ['Gaming', 'Netflix', 'Other']

for app in top_3_applications:
    plt.figure()  # Create a new figure for each application
    top_10_customers = engagement_metrics.sort_values(by=f"{app} DL (Bytes)", ascending=False).head(10)
    x = range(len(top_10_customers))
    plt.bar(x, top_10_customers[f"{app} DL (Bytes)"], label='Download')
    plt.bar(x, top_10_customers[f"{app} UL (Bytes)"], label='Upload')
    plt.xlabel('MSISDN/Number')
    plt.ylabel('Bytes')
    plt.title(f'Top 10 Customers - {app}')
    plt.xticks(x, top_10_customers.index, rotation='vertical')
    plt.legend()
    plt.show()


# In[149]:


top_10_customers


# In[150]:


import warnings
warnings.filterwarnings("ignore")

sse = {}
kmax = 10

for k in range(1, kmax + 1):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=42).fit(engagement_metrics_normalized)
    sse[k] = kmeans.inertia_

plt.plot(list(sse.keys()), list(sse.values()), marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method')
plt.show()


# ## K = 2

# In[151]:


from sklearn.preprocessing import MinMaxScaler

# Normalize each engagement metric
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(engagement_metrics)


# In[152]:


# Define a range of cluster numbers to test
cluster_range = range(1, 11)

# Initialize an empty list to store the inertia values (sum of squared distances from samples to their closest cluster center)
inertia_values = []

# Iterate over the cluster range and compute the inertia for each number of clusters
for num_clusters in cluster_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(normalized_data)
    inertia_values.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(cluster_range, inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# In[161]:


for app in applications:
    engaged_users_per_app[app] = engagement_metrics[f'{app} DL (Bytes)'] + engagement_metrics[f'{app} UL (Bytes)']
    top_10_users_per_app[app] = engaged_users_per_app[app].nlargest(10)

# Print the results
print('Engaged Users per - {app}')
print(engaged_users_per_app)
print(50*"*")
print(n'Top 10 Most Engaged Users per - {app}')
print(top_10_users_per_app)


# In[159]:


# Calculate the total engaged users per application
total_engaged_users = [engaged_users_per_app[app].sum() for app in applications]

# Plot the bar graph
plt.figure(figsize=(8, 6))
plt.bar(applications, total_engaged_users)
plt.xlabel('Application')
plt.ylabel('Total Engaged Users')
plt.title('Total Engaged Users per Application')

# Display the graph
plt.show()


# In[154]:


# Plot the top 3 most used applications
top_3_applications = ['Netflix','Youtube','Other']

for app in top_3_applications:
    plt.figure()
    top_10_users = top_10_users_per_app[app]
    x = range(len(top_10_users))
    plt.bar(x, top_10_users, label='Engaged Users')
    plt.xlabel('MSISDN/Number')
    plt.ylabel('Engagement')
    plt.title(f'Top 10 Users - {app}')
    plt.xticks(x, top_10_users.index, rotation='vertical')
    plt.legend()
    plt.show()


# In[155]:


# Use the elbow method to determine the optimized value of k for clustering
inertia = []
k_values = range(2, 11)  # Test values of k from 2 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(engagement_metrics_normalized)
    inertia.append(kmeans.inertia_)


# In[156]:


# Plot the SSE for different values of k
plt.plot(list(sse.keys()), list(sse.values()),marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method')
plt.show()


# In[157]:


print(top_10_sessions_frequency)
print(top_10_session_duration)
print(top_10_total_traffic)
print(cluster_metrics)


# In[ ]:





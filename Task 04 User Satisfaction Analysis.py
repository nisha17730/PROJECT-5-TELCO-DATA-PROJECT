#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[6]:


df = pd.read_csv('telcom_data.csv')


# In[7]:


df


# In[8]:


# Select only the numeric columns
numeric_cols = df.select_dtypes(include=['number'])

# Calculate the mean of the numeric columns and fill NaN values
df[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())


# In[9]:


# Fill NaN values only for numeric columns with their respective means
df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)


# In[10]:


df['session_frequency'] = df.groupby(by=['MSISDN/Number'])['Dur. (ms)'].transform('count')


# In[11]:


df['session_duration'] = df['Dur. (ms)']


# In[12]:


df['total_traffic'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']


# In[13]:


# Calculate average TCP retransmission
# Group the data by 'MSISDN/Number', and calculate the mean of 'TCP DL Retrans. Vol (Bytes)' and 'TCP UL Retrans. Vol (Bytes)'
# Then, take the overall mean of the two mean values for each group
avg_tcp_retransmission = df.groupby('MSISDN/Number')[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']].mean().mean(axis=1)

# Fill missing values in the 'avg_tcp_retransmission' with the overall mean
# This will help handle any NaN (Not a Number) values in the resulting Series
avg_tcp_retransmission.fillna(avg_tcp_retransmission.mean(), inplace=True)

# Calculate average RTT (Round Trip Time)
# Group the data by 'MSISDN/Number', and calculate the mean of 'Avg RTT DL (ms)' and 'Avg RTT UL (ms)'
# Then, take the overall mean of the two mean values for each group
avg_rtt = df.groupby('MSISDN/Number')[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].mean().mean(axis=1)

# Fill missing values in the 'avg_rtt' with the overall mean
avg_rtt.fillna(avg_rtt.mean(), inplace=True)

#  Replace missing values in 'Handset Type' with the mode (most common value) for each 'MSISDN/Number'
# Group the data by 'MSISDN/Number', and find the mode (most common value) of 'Handset Type' for each group
# If there are multiple modes, choose the first one; otherwise, set it to None
handset_mode = df.groupby('MSISDN/Number')['Handset Type'].agg(lambda x: x.mode().values[0] if len(x.mode()) > 0 else None)

#  Calculate average throughput
# Group the data by 'MSISDN/Number', and calculate the mean of 'Avg Bearer TP DL (kbps)' and 'Avg Bearer TP UL (kbps)'
# Then, take the overall mean of the two mean values for each group
avg_throughput = df.groupby('MSISDN/Number')[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean().mean(axis=1)

# Fill missing values in the 'avg_throughput' with the overall mean
avg_throughput.fillna(avg_throughput.mean(), inplace=True)

#  Create a new DataFrame with aggregated information
# Combine all the calculated averages and the 'Handset_Type' mode into a new DataFrame
aggregated_data = pd.DataFrame({
    'Avg_TCP_Retransmission': avg_tcp_retransmission,
    'Avg_RTT': avg_rtt,
    'Handset_Type': handset_mode,
    'Avg_Throughput': avg_throughput
})

# Step 7: Reset the index to make 'MSISDN/Number' a column instead of the index
# This step is done to bring the 'MSISDN/Number' back as a regular column in the DataFrame
aggregated_data.reset_index(inplace=True)

# Step 8: Display the aggregated data
# Print the final DataFrame that contains the aggregated information
print(aggregated_data)


# In[14]:


user_data = aggregated_data[['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']].valuesuser_data = aggregated_data[['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']].values


# In[15]:


import warnings
warnings.filterwarnings('ignore')
sse = {};
kmax = 10
fig = plt.subplots(figsize = (20,5))

# Elbow Method :
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(user_data)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
sns.lineplot(x = list(sse.keys()), y = list(sse.values()));
plt.title('Elbow Method')
plt.xlabel("k : Number of cluster")
plt.ylabel("Sum of Squared Error")
plt.grid()


# In[16]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters = 2,
               max_iter = 1000)
model.fit(user_data)


# In[17]:


cluster = model.cluster_centers_
cluster


# In[18]:


labels = model.labels_
labels
centroids= np.array(cluster)
centroids


# In[19]:


from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
distances = euclidean_distances(user_data, centroids)

less_engaged_cluster_index = 0

engagement_scores = distances[:, less_engaged_cluster_index]

mms = MinMaxScaler()
engagement_scores = mms.fit_transform(engagement_scores.reshape(-1, 1))

aggregated_data['Engagement Score'] = engagement_scores


aggregated_data


# In[20]:


cluster_centroids = centroids

distances = euclidean_distances(user_data, cluster_centroids)

worst_experience_cluster_index = cluster_centroids.shape[0] - 1

experience_scores = distances[:, worst_experience_cluster_index]

mms = MinMaxScaler()
experience_scores = mms.fit_transform(experience_scores.reshape(-1, 1))

aggregated_data['Experience Score'] = experience_scores

aggregated_data


# In[21]:


aggregated_data['satisfaction_score'] = (aggregated_data['Engagement Score'] + aggregated_data['Experience Score']) / 2
top_10_satisfied_customers = aggregated_data.nlargest(10, 'satisfaction_score')


# ### Task 4.2 - Reporting Top 10 Satisfied Customers

# In[22]:


top_10_satisfied_customers


# ### Task 4.3 Building a Regression Model for Satisfaction Prediction

# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X = aggregated_data[['Engagement Score', 'Experience Score']]
y = aggregated_data['satisfaction_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the regression model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Predict satisfaction scores for the test set
y_pred = regression_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)


# In[24]:


mse


# In[25]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)


# ### Task 4.4 - Running k-means on Engagement and Experience Scores

# In[26]:


scores = aggregated_data[['Engagement Score', 'Experience Score']]
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(scores)
aggregated_data['score_cluster'] = clusters


# In[27]:


# Perform KMeans clustering on the 'Engagement Score' and 'Experience Score'
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(scores)

# Add the 'score_cluster' column to 'aggregated_data' to store the cluster information
aggregated_data['score_cluster'] = clusters

# Plotting the 'Engagement Score' and 'Experience Score' with the corresponding cluster using a scatter plot
plt.figure(figsize=(10, 6))  # Setting the figure size to make the plot more visible

# Scatter plot for cluster 0 (first cluster)
plt.scatter(aggregated_data[aggregated_data['score_cluster'] == 0]['Engagement Score'],
            aggregated_data[aggregated_data['score_cluster'] == 0]['Experience Score'],
            color='red', label='Cluster 0')

# Scatter plot for cluster 1 (second cluster)
plt.scatter(aggregated_data[aggregated_data['score_cluster'] == 1]['Engagement Score'],
            aggregated_data[aggregated_data['score_cluster'] == 1]['Experience Score'],
            color='blue', label='Cluster 1')

# Adding labels and title for the scatter plot
plt.xlabel('Engagement Score')
plt.ylabel('Experience Score')
plt.title('KMeans Clustering of Engagement Score and Experience Score')

# Adding legend to differentiate between clusters
plt.legend()

# Display the scatter plot
plt.show()


# ### Task 4.5 - Aggregating Average Satisfaction and Experience Scores per Cluster

# In[28]:


cluster_agg = aggregated_data.groupby('score_cluster').agg({
    'satisfaction_score': 'mean',
    'Experience Score': 'mean'})


# In[29]:


cluster_agg


# In[30]:


# Define the cluster labels and average scores
cluster_labels = ['Cluster 0', 'Cluster 1']
avg_satisfaction_scores = cluster_agg['satisfaction_score']
avg_experience_scores =cluster_agg['Experience Score']

# Plot the average satisfaction scores
plt.figure(figsize=(8, 6))
plt.bar(cluster_labels, avg_satisfaction_scores, color='blue')
plt.xlabel('Cluster')
plt.ylabel('Average Satisfaction Score')
plt.title('Average Satisfaction Score per Cluster')

# Display the plot
plt.show()

# Plot the average experience scores
plt.figure(figsize=(8, 6))
plt.bar(cluster_labels, avg_experience_scores, color='green')
plt.xlabel('Cluster')
plt.ylabel('Average Experience Score')
plt.title('Average Experience Score per Cluster')

# Display the plot
plt.show()


# In[ ]:





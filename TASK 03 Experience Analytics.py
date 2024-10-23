#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('telcom_data.csv')


# In[3]:


df


# In[7]:


# Select only the numeric columns
numeric_cols = df.select_dtypes(include=['number'])

# Calculate the mean of the numeric columns and fill NaN values
df[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean())


# In[9]:


# Fill NaN values only for numeric columns with their respective means
df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)


# 3.1

# In[12]:


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


# In[13]:


aggregated_data.isnull().sum()


# In[14]:


# Filter out non-numeric columns
numeric_cols = aggregated_data.select_dtypes(include=np.number).columns

# Plot boxplots for numeric columns
for col in numeric_cols:
    sns.boxplot(data=aggregated_data, x=col)
    plt.show()


# In[15]:


# Filter out only the numeric columns from the DataFrame
# This will create a new DataFrame called 'numeric_data' containing only the numeric columns from 'aggregated_data'
numeric_data = aggregated_data.select_dtypes(include=np.number)

# Calculate percentiles (Q1, Q2, Q3) for the numeric columns
# Using the np.percentile function to calculate the 25th (Q1), 50th (Q2 or median), and 75th (Q3) percentiles
q1 = np.percentile(numeric_data, 25)
q2 = np.percentile(numeric_data, 50)
q3 = np.percentile(numeric_data, 75)

# Step 4: Display the calculated percentiles
print(f"My Q1 = {q1}, Q2 = {q2}, Q3 = {q3}")


# In[16]:


iqr=q3-q1
iqr


# In[17]:


# Calculate the lower and upper range using the interquartile range (IQR) method
lower_range = q1 - iqr * 1.5
upper_range = q3 + iqr * 1.5

# Display the calculated lower and upper ranges
print(f"Lower range = {lower_range}, Upper Range = {upper_range}")


# In[18]:


def find_outliers_iqr(df, threshold=1.5):
    outliers = pd.DataFrame()
    
    for column in df.columns:
        # Check if the column is numeric (exclude non-numeric columns)
        if np.issubdtype(df[column].dtype, np.number):
            # Calculate the first quartile (Q1), third quartile (Q3), and the interquartile range (IQR)
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            
            # Calculate the lower and upper bounds for outliers
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Find the rows that have values below the lower_bound or above the upper_bound
            column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outliers = pd.concat([outliers, column_outliers])
            
            # Replace the outliers with the mean value of the column
            df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound),
                                  df[column].mean(),
                                  df[column])
    
    return outliers


# In[19]:


# Use the `find_outliers_iqr` function to find and replace outliers in 'aggregated_data'
data = find_outliers_iqr(aggregated_data)


# In[20]:


# Step 1: Get the numeric columns from the DataFrame 'aggregated_data'
# This will create a pandas Index object containing the names of the numeric columns
numeric_cols = aggregated_data.select_dtypes(include=np.number).columns

# Step 2: Plot boxplots for each numeric column in 'data'
# The loop iterates through each numeric column, plots the boxplot, and displays it using plt.show()
for col in numeric_cols:
    sns.boxplot(data=data, x=col)
    plt.show()


# ### 3.2

# In[21]:


# Step 1: Create an empty list to store the names of the numerical columns
numerical_data = []

# Step 2: Loop through each column in the 'data' DataFrame
for col in data.columns:
    # Step 3: Check if the data type of the column is not "object" (i.e., not a string or categorical column)
    if data[col].dtypes != "object":
        # Step 4: If the column is numeric, append its name to the 'numerical_data' list
        numerical_data.append(col)


# In[22]:


# Step 1: Loop through each column in the 'data' DataFrame
for i in data.columns:
    print("*" * 50)
    # Step 2: Print the data type of the current column
    print(f"The datatype for {i} is {data[i].dtypes}")
    
    # Step 3: Check if the data type is not "object" (i.e., if it's a numeric column)
    if data[i].dtypes != "object":
        # Step 4: Print the top 10 categories for numeric columns
        print(f"Top 10 categories for {i}:")
        print(data[i].value_counts().head(10))
    print("*" * 50)


# In[23]:


# Top 10 TCP values
top_10_tcp = data['Avg_TCP_Retransmission'].nlargest(10)
print("Top 10 TCP values:")
print(top_10_tcp)
print(50*"*")


# In[24]:


# Plot histogram for top 10 TCP values
plt.figure(figsize=(10, 5))
sns.histplot(data=top_10_tcp, bins=10)
plt.title("Histogram of Top 10 TCP Values")
plt.xlabel("TCP Values")
plt.ylabel("Frequency")
plt.show()


# In[25]:


# Bottom 10 TCP values
print('\n',50*"*")
bottom_10_tcp = data['Avg_TCP_Retransmission'].nsmallest(10)
print("Bottom 10 TCP values:")
print(bottom_10_tcp)
print('\n',50*"*")


# In[26]:


# Plot histogram for bottom 10 TCP values
plt.figure(figsize=(10, 5))
sns.histplot(data=bottom_10_tcp, bins=10)
plt.title("Histogram of Bottom 10 TCP Values")
plt.xlabel("TCP Values")
plt.ylabel("Frequency")
plt.show()


# In[27]:


# Most frequent TCP values
print('\n',50*"*")
most_frequent_tcp = data['Avg_TCP_Retransmission'].value_counts().head(10)
print("Most frequent TCP values:")
print(most_frequent_tcp)
print('\n',50*"*")


# In[28]:


plt.figure(figsize=(10, 5))
sns.histplot(data=most_frequent_tcp, bins=10)
plt.title("Histogram of Most Frequent TCP Values")
plt.xlabel("TCP Values")
plt.ylabel("Frequency")
plt.show()


# In[29]:


# Top 10 RTT values
print('\n',50*"*")
top_10_rtt = data['Avg_RTT'].nlargest(10)
print("Top 10 RTT values:")
print(top_10_rtt)
print('\n',50*"*")

plt.figure(figsize=(10, 5))
sns.histplot(data=top_10_rtt, bins=10,kde=True)
plt.title("Histogram of top 10 RTT values")
plt.xlabel("RTT values")
plt.ylabel("Frequency")
plt.show()


# In[30]:


# Bottom 10 RTT values
print('\n',50*"*")
bottom_10_rtt = data['Avg_RTT'].nsmallest(10)
print("Bottom 10 RTT values:")
print(bottom_10_rtt)
print('\n',50*"*")
plt.figure(figsize=(10, 5))
sns.histplot(data=bottom_10_rtt, bins=10,kde=True)
plt.title("Histogram of bottom 10 RTT values")
plt.xlabel("RTT values")
plt.ylabel("Frequency")
plt.show()


# In[31]:


# Most frequent RTT values
print('\n',50*"*")
most_frequent_rtt = data['Avg_RTT'].value_counts().head(10)
print("Most frequent RTT values:")
print(most_frequent_rtt)
print('\n',50*"*")
plt.figure(figsize=(10, 5))
sns.histplot(most_frequent_rtt, bins=10,kde=True)
plt.title("Histogram of Most frequent RTT values")
plt.xlabel("TRTT values")
plt.ylabel("Frequency")
plt.show()


# In[32]:


# Top 10 throughput values
print('\n',50*"*")
top_10_throughput = data['Avg_Throughput'].nlargest(10)
print("Top 10 throughput values:")
print(top_10_throughput)
print('\n',50*"*")

plt.figure(figsize=(10, 5))
sns.histplot(top_10_throughput, bins=10,kde=True)
plt.title("Histogram of Top 10 Throughput Values")
plt.xlabel("Throughput Values")
plt.ylabel("Frequency")
plt.show()


# In[33]:


# Bottom 10 throughput values
print('\n',50*"*")
bottom_10_throughput = data['Avg_Throughput'].nsmallest(10)
print("Bottom 10 throughput values:")
print(bottom_10_throughput)
print('\n',50*"*")
plt.figure(figsize=(10, 5))
sns.histplot(bottom_10_throughput, bins=10,kde=True)
plt.title("Histogram of bottom Throughput Values")
plt.xlabel("Throughput Values")
plt.ylabel("Frequency")
plt.show()


# In[34]:


# Most frequent throughput values
print('\n',50*"*")
most_frequent_throughput = data['Avg_Throughput'].value_counts().head(10)
print("Most frequent throughput values:")
print(most_frequent_throughput)
print('\n',50*"*")

plt.figure(figsize=(10, 5))
sns.histplot(most_frequent_throughput, bins=10,kde=True)
plt.title("Histogram of Most Frequent Throughputs")
plt.xlabel("Throughput Values")
plt.ylabel("Frequency")
plt.show()


# In[35]:


data['Handset_Type'].value_counts()


# In[36]:


# Compute average throughput per handset type
avg_throughput_per_type = data.groupby('Handset_Type')['Avg_Throughput'].mean()

# Compute average TCP retransmission per handset type
avg_tcp_retransmission_per_type = data.groupby('Handset_Type')['Avg_TCP_Retransmission'].mean()


# In[37]:


avg_throughput_per_type


# In[38]:


avg_tcp_retransmission_per_type


# In[39]:


# Plotting the average throughput per handset type using a bar plot
plt.figure(figsize=(12, 6))  # Setting the figure size to make the plot more visible
plt.bar(avg_throughput_per_type.index, avg_throughput_per_type.values)

# Adding labels and title for the average throughput plot
plt.xlabel("Handset Type")
plt.ylabel("Average Throughput")
plt.title("Average Throughput per Handset Type")

# Rotating the x-axis labels for better readability
plt.xticks(rotation=45)

# Display the average throughput plot
plt.show()


# Plotting the average TCP retransmission per handset type using a bar plot
plt.figure(figsize=(12, 6))  # Setting the figure size to make the plot more visible
plt.bar(avg_tcp_retransmission_per_type.index, avg_tcp_retransmission_per_type.values)

# Adding labels and title for the average TCP retransmission plot
plt.xlabel("Handset Type")
plt.ylabel("Average TCP Retransmission")
plt.title("Average TCP Retransmission per Handset Type")

# Rotating the x-axis labels for better readability
plt.xticks(rotation=45)

# Display the average TCP retransmission plot
plt.show()


# In[40]:


# Select the top 10 most common handset types
top_n = 10
top_handset_types = aggregated_data['Handset_Type'].value_counts().head(top_n).index


# In[41]:


# Filter the data for the top handset types
top_handset_data = aggregated_data[aggregated_data['Handset_Type'].isin(top_handset_types)]


# In[42]:


# Create a bar plot of average TCP retransmission per handset type
plt.figure(figsize=(12, 6))
plt.bar(top_handset_data['Handset_Type'], top_handset_data['Avg_TCP_Retransmission'])
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission')
plt.title(f'Average TCP Retransmission per Top {top_n} Handset Types')
plt.xticks(rotation=45)
plt.show()


# In[43]:


# Select the bottom 10 least common handset types
bottom_n = 10
bottom_handset_types = aggregated_data['Handset_Type'].value_counts().tail(bottom_n).index

# Filter the data for the bottom handset types
bottom_handset_data = aggregated_data[aggregated_data['Handset_Type'].isin(bottom_handset_types)]

# Create a bar plot of average TCP retransmission per handset type
plt.figure(figsize=(12, 6))
plt.bar(bottom_handset_data['Handset_Type'], bottom_handset_data['Avg_TCP_Retransmission'])
plt.xlabel('Handset Type')
plt.ylabel('Average TCP Retransmission')
plt.title(f'Average TCP Retransmission per Bottom {bottom_n} Handset Types')
plt.xticks(rotation=90)
plt.show()


# ### 3.4 K=2 from Task 2

# In[45]:


experience_data = aggregated_data[['Avg_TCP_Retransmission', 'Avg_RTT', 'Avg_Throughput']]


# In[46]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(experience_data)


# In[47]:


from sklearn.cluster import KMeans

k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_data)


# ### In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# ### On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[49]:


cluster_labels = kmeans.labels_


# In[50]:


cluster_means = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=experience_data.columns)


# In[51]:


# Analyze the clusters
cluster_means = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=experience_data.columns)

# Print the cluster means
print("Cluster Means:")
print(cluster_means)

# Add cluster labels to the aggregated_data DataFrame
aggregated_data['Cluster'] = cluster_labels

# Display the updated aggregated_data DataFrame
print("Updated aggregated_data:")
print(aggregated_data)


# In[52]:


# Description of each cluster
for cluster in range(k):
    print("\nCluster", cluster)
    print("Number of users:", len(aggregated_data[aggregated_data['Cluster'] == cluster]))
    print("Cluster Mean:")
    print(cluster_means.iloc[cluster])


# In[53]:


# Analyze the clusters
cluster_means = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=experience_data.columns)
# Print the cluster means
for i, cluster_mean in enumerate(cluster_means.iterrows()):
    print(f"Cluster {i}:")
    print("Number of users:", aggregated_data['Cluster'].value_counts()[i])
    print("Cluster Mean:")
    print(cluster_mean[1])
    print()


# In[54]:


# Cluster descriptions
cluster_descriptions = {
    0: "Cluster 0 represents users with a relatively stable and high-performing network experience. They have lower average TCP retransmission (7.67 million), moderate average RTT (55.82 ms), and higher average throughput (7,079 kbps).",
    1: "Cluster 1 represents users with higher network latency and lower throughput. They have higher average TCP retransmission (19.96 million), higher average RTT (109.11 ms), and lower average throughput (3,487 kbps).",
    2: "Cluster 2 represents users with higher TCP retransmission and a mix of network performance. They have higher average TCP retransmission (19.89 million), lower average RTT (40.87 ms), and lower average throughput (3,548 kbps)."
}

# Print cluster descriptions
for i in range(k):
    print("Cluster", i, "Description:")
    print(cluster_descriptions[i])
    print()


# In[55]:


# Create scatter plots for each experience metric
cluster_sizes = aggregated_data['Cluster'].value_counts().sort_index()
for metric in experience_data.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=aggregated_data, x='Cluster', y=metric)
    plt.xlabel('Cluster')
    plt.ylabel(metric)
    plt.title(f'{metric} by Cluster')
    plt.show()

# Explore cluster characteristics
num_clusters = 3  # Replace with the actual number of clusters

for i, cluster_mean in enumerate(cluster_means.iterrows()):
    cluster_label = i
    num_users = cluster_sizes[i]

    # Explore cluster characteristics
    cluster_data = aggregated_data[aggregated_data['Cluster'] == i]
    cluster_description = f"Cluster {i}:\nNumber of users: {num_users}\nCluster Mean:\n{cluster_mean[1]}"

    # Perform additional analysis or calculations based on the cluster characteristics

    # Provide actionable insights or recommendations based on the analysis
    print(cluster_description)
    print("Actionable Insights:")
    print("Based on the analysis, it is recommended to...")
    print()

# Conduct statistical tests
import scipy.stats as stats


for metric in experience_data.columns:
    cluster_data = [aggregated_data[aggregated_data['Cluster'] == i][metric] for i in range(num_clusters)]
    f_value, p_value = stats.f_oneway(*cluster_data)
    print(f"ANOVA test for {metric}:")
    print("F-value:", f_value)
    print("p-value:", p_value)
    print()


# In[56]:


# Create a scatter plot for each pair of experience metrics
plt.figure(figsize=(12, 8))
for i, metric1 in enumerate(experience_data.columns):
    for j, metric2 in enumerate(experience_data.columns):
        plt.subplot(3, 3, i * 3 + j + 1)
        for cluster_label in range(3):
            cluster_data = experience_data[aggregated_data['Cluster'] == cluster_label]
            plt.scatter(cluster_data[metric1], cluster_data[metric2], label=f'Cluster {cluster_label}', alpha=0.5)
        plt.xlabel(metric1)
        plt.ylabel(metric2)
plt.legend()
plt.tight_layout()
plt.show()


# In[57]:


# Explore cluster characteristics using box plots
plt.figure(figsize=(12, 6))
for i, metric in enumerate(experience_data.columns):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(x='Cluster', y=metric, data=aggregated_data)
    plt.xlabel('Cluster')
    plt.ylabel(metric)
plt.tight_layout()
plt.show()


# In[ ]:





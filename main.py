import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

### Constants ###
fileName = "uberNYC_August2014.csv"
sampleSize = 100000
testSize = 0.1
headerX, headerY = "Lon", "Lat"

#Part A
data = pd.read_csv(fileName, header=0)
samples = data.sample(n=sampleSize)
trainSet, testSet = train_test_split(samples, test_size=testSize)
train, test = [], []
for i in range(round(sampleSize * (1 - testSize))):
    train.append([data.at[i, headerX], data.at[i, headerY]])
for i in range(round(sampleSize * testSize)):
    test.append([data.at[i, headerX], data.at[i, headerY]])
train, test = np.array(train), np.array(test)

#Part B
kmeans = KMeans().fit(train)
print(kmeans.labels_)
print(kmeans.cluster_centers_)

#Part C
plt.xlim(-74.6, -73)
plt.ylim(40.3, 41.2)
plt.xlabel('Longtitude')
plt.ylabel('Latitude')
plt.title('NYC Uber Pickups - Training Set')
plt.scatter(train[:, 0], train[:, 1], c=kmeans.labels_, cmap='Set1', s=1)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", color='k')
plt.show()

#Part D
plt.xlim(-74.6, -73)
plt.ylim(40.3, 41.2)
plt.scatter(test[:, 0], test[:, 1], c=kmeans.predict(test), cmap='Set1', s=1)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", color='k')
plt.show()

# coding: utf-8

# In[1]:


###Standard Import
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# use seaborn plotting defaults
import seaborn as sns; sns.set()


# In[5]:


import pylab as plt
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
iris = load_iris()


# In[3]:





k = range(1,11)

clusters = [KMeans(n_clusters = c,init = 'k-means++').fit(iris.data) 
            for c in k]
centr_lst = [cc.cluster_centers_ for cc in clusters]

k_distance = [cdist(iris.data, cent, 'euclidean') for cent in centr_lst]
clust_indx = [np.argmin(kd,axis=1) for kd in k_distance]
distances = [np.min(kd,axis=1) for kd in k_distance]
avg_within = [np.sum(dist)/iris.data.shape[0] for dist in distances]

kidx = 2

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k, avg_within, 'g*-')
ax.plot(k[kidx], avg_within[kidx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for KMeans clustering (IRIS Data)')
#with_in_sum_square = [np.sum(dist ** 2) for dist in distances]
#to_sum_square = np.sum(pdist(iris.data) ** 2)/iris.data.shape[0]
#bet_sum_square = to_sum_square - with_in_sum_square


# In[4]:


####Clustering for IRIS data =====#Unsupervised======
from sklearn import cluster, datasets
from sklearn import metrics
iris = datasets.load_iris()
k_means = cluster.KMeans(n_clusters=3)
kmf=k_means.fit(iris.data)

y_kmeans = kmf.predict(iris.data)

plt.scatter(iris.data[:, 0], iris.data[:, 1], c=y_kmeans, s=50,
            cmap='rainbow');

from sklearn import metrics
ms=metrics.silhouette_score(iris.data, y_kmeans,
                                      metric='euclidean')
print("silhouette_score=%f",ms)


# The data sets contains 150 points of 3 different labels(known).
# Use KNN to classify them.(same train and test sets)

import numpy as np
import matplotlib.pyplot as plt
import math
# Define a function to load and process data.
# Return: data - each data point in numpy array form. # label - each label.
def loadData(infile):
    f = open(infile,'r') sourceInLine = f.readlines() data = []
    label = []
    for line in sourceInLine:
        temp = line.strip('\n').split(',')
        label.append(temp[-1])
        temp = [float(x) for x in temp[:-1]]
        data.append(temp)
    return np.array(data), np.array(label)
data, label = loadData('./data.txt')
N = len(label) # Total number of data points

# Check if the datasets are balanced. It looks like our data is well balanced.
for i in range(3):
    print('percentage of label %s in training set: %.2f %%'%(chr(ord("A") + i), 100 * sum(label == chr(ord("A") + i)) / N))

# Distances with Euclidean and Manhattan measurement.
# Define Euclidean distance
# x1 ,x2 needs to be numpy array.
def distance_Euc(x1, x2):
    return math.sqrt(sum((x1 - x2) ** 2))
# Define Manhattan distance
# x1 ,x2 needs to be numpy array.
def distance_Manhattan(x1, x2):
    return sum(abs((x1 - x2)))

# Calculate the Euclidean and Manhattan ditance for each pair of data points.
dis_Euc = np.array(list(map(lambda x: [distance_Euc(x, y) for y in data], data)))
dis_Manhattan = np.array(list(map(lambda x: [distance_Manhattan(x, y) for y in data], data)))

# Another way to get Euclidean distance. Note that ||x1 − x2 ||2 = ||x1 ||2 + ||x2 ||2 − 2 ∗ xT1 · x2 . I can use matrix algebra to calculate Euclidean distance matrix directly.
# When the dataset is huge, this method can save a lot of time. # Same with above result running this chunk.
# Euclidean Norm of each data points: shape of (N * 1)
norm = np.sum(data ** 2, axis = 1).reshape(N, 1)
norm_matrix = norm.dot(np.ones(N).reshape(1, N)) # (N * N)
dis_Euc = np.sqrt(norm_matrix + norm_matrix.T - 2 * data.dot(data.T))

#Normalization
# Attributes have the potential to dominate in the contribution to the distance measure, so I normalize the data and calculate two distance again for further use. Data are substracted by mean of each attribute and divided by standard error of each attribute.
# Centralization and Normalization
data_normlized = (data - np.mean(data,axis=0)) / np.std(data,axis=0)
dis_Euc_normlized = np.array(list(map(lambda x: [distance_Euc(x, y) for y in data_normlized], data_normlized)))
dis_Manhattan_normlized = np.array(list(map(lambda x: [distance_Manhattan(x, y) for y in data_normlized], data_normlized)))

# KNN main function
# Define the function that implement KNN algorithm.
# Input: K - number of nearest neighbors.
#        data - data points matrix in numpy form.
#        label - data label vector in numpy form.
#        distance - matrix of diatances of all pair of data points in numpy form.
# Output: K most analogous data distances, indexes, labels(in numpy matrix form)
#         and prediction accuracy.
def KNN(K, data, label, distance):
    N = len(label)
    K_cloest_dis = []
    K_cloest_index = []
    K_cloest_label = []
    totalerror = 0
    for i in range(N):
        # K most analogous data indexes. Note the index begins with 0 in python.
        K_cloest_index_i = np.argsort(distance[i])[1 : K + 1] # Discard the point itself.
        K_cloest_index.append(K_cloest_index_i)
        # K most analogous data instances.
        K_cloest_dis.append([distance[i][j] for j in K_cloest_index_i])
        # K most analogous data labels.
        K_cloest_label_i = label[K_cloest_index_i]
        K_cloest_label.append(K_cloest_label_i)
        count_i = [list(K_cloest_label_i).count(chr(ord("A") + i)) for i in range(3)]
        # #data points mis-predicted.
        totalerror += (chr(ord("A") + np.argmax(count_i)) != label[i])
    return K_cloest_dis, K_cloest_index, K_cloest_label, 1 - float(totalerror / N)

# Regression: revise the model
# regression: in the summarization of the closest instances use the mean
# Actually I do not quite follow your idea here. I revise the model by calculate # mean ditance to each label group
# amoong K nearest neighbors for each data point.
def KNNrevise(K, data, label, distance):
    N = len(label)
    K_cloest_dis = []
    K_cloest_index = []
    K_cloest_label = []
    totalerror = 0
    for i in range(N):
        meandis = [float('inf')] * 3 # Initialize the mean distance
        # K most analogous data indexes. Note the index begins with 0 in python.
        K_cloest_index_i = np.argsort(distance[i])[1 : K + 1] # Discard the point itself.
        K_cloest_index.append(K_cloest_index_i)
        # K most analogous data instances.
        K_cloest_dis_i = [distance[i][j] for j in K_cloest_index_i]
            K_cloest_dis.append(K_cloest_dis_i)
        # K most analogous data labels.
        K_cloest_label_i = label[K_cloest_index_i]
        K_cloest_label.append(K_cloest_label_i)
        count_i = [list(K_cloest_label_i).count(chr(ord("A") + i)) for i in range(3)]
        # #data points mis-predicted.
        for m in range(3):
            if count_i[m]:
                meandis[m] = np.mean([K_cloest_dis_i[n] for n in range(K) if K_cloest_label_i[n] == chr(ord("A") + m)])
        totalerror += (chr(np.argmin(meandis) + ord("A")) != label[i])
    return K_cloest_dis, K_cloest_index, K_cloest_label, 1 - totalerror / N


#Results Visualization
K_total = range(1, 150)
accu = []
for K in K_total:
    accu.append(KNN(K, data, label, dis_Euc)[3])
plt.plot(K_total, accu)
plt.title('Accuracy v.s. K under Euclidean Distance')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

#It looks that normalization does not help a lot. This may because the last attribute, which is relatively small in scale, does not contain much information. Manhattan distance and Euclidean distance are similar in results. All the model behave well when K < 50, and this is because model lose local structure information when K is too large. It is worth notice that, revised model gives good result for any K, suggesting mean regression brings robustness to the KNN algorithm.

# Numerical
# Find the 10 most analogous data instances of each data, show their index, label and Euclidean distances.
K = 10
dis, ind, lab, acc = KNN(K, data, label, dis_Euc)
for i in [0, 1, 2, 50, 51, 52, 100, 101, 102]:
    for j in range(K):
        print(' %d analogous data points for data %d(%s): %d(%s) with distnace %.2f '% (K, i, label[i], ind[i][j], lab[i][j], dis[i][j]))


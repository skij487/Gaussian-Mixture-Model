import numpy as np

def initiate(points, K):
    points = np.random.permutation(points)
    means = points[0:K]
    return means

def ClosestMean(points, means): # This way index of means does not shuffle
    K, D = np.shape(means)
    length = len(points)
    distances = np.sum((points - np.reshape(means,(K,1,D))) ** 2, 2)
    zs = np.argmin(distances, 0)
    cost = np.sum(distances[zs,range(length)])
    return zs, cost

def CalculateMeans(points, zs, K):
    N, D = np.shape(points)
    means = np.zeros((K, D))
    for i in range(K):
        curr_points = points[zs == i]
        length = len(curr_points)
        means[i] = np.sum(curr_points, axis=0) / length
    return means

def KMeans(points, K):
    length = len(points)
    means = initiate(points, K)
    prev_zs = np.zeros(length)
    zs = np.ones(length)
    zs, cost = ClosestMean(points, means)
    while not np.array_equal(zs, prev_zs):
        prev_zs = zs
        means = CalculateMeans(points, zs, K)
        zs, cost = ClosestMean(points, means)
    return zs, cost, means

def KMeansMin(points, K):
    min_zs, min_cost, min_means = KMeans(points, K)
    same = False
    while not same:
        zs, cost, means = KMeans(points, K)
        if np.array_equal(zs, min_zs):
            same = True
        elif cost < min_cost:
            min_zs = zs
            min_cost = cost
            min_means = means
    return zs, cost, means
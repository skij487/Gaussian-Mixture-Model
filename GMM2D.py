import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as pat
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pandas as pd
from scipy.stats import multivariate_normal
import math

def GMM2D(data=None, K=3, constraints="full", nit=100):
    import KMeans
    kzs, Js, kmus = KMeans.KMeansMin(data, K)
    mus, sigma, pi, r, N, LL = GMM(data, K, constraints, nit)
    dist = [[np.sqrt((kmus[n][0] - mus[k][0])**2 + (kmus[n][1] - mus[k][1])**2) for k in range(K)] for n in range(K)]
    zs = assign(data, K, mus, sigma)
    n_mus = [[] for _ in range(K)]
    n_sigma = [[] for _ in range(K)]
    n_r = np.empty_like(r)
    n_zs = np.empty_like(zs)
    idx = np.argmin(dist,axis=1)
    for k in range(len(idx)):
        n_mus[idx[k]] = mus[k]
        n_sigma[idx[k]] = sigma[k]
        for n in range(len(data)):
            n_r[n][idx[k]] = r[n][k]
            if zs[n] == k:
                n_zs[n] = idx[k]
    mus = n_mus
    sigma = n_sigma
    r = n_r
    zs = n_zs
    '''
    nsamp = 100
    t = np.linspace(0,2*np.pi,nsamp)
    uicrc = np.array([np.cos(t), np.sin(t)])    
    color_c = [(1,0.2,0.2),(0.2,1,0.2),(0.2,0.2,1)]
    for k in range(K):
        L, U = np.linalg.eig(sigma[k])
        covelipse = mus[k] + (U @ np.diag(np.sqrt(L)) @ uicrc).T
        plt.plot(covelipse[:,0],covelipse[:,1], color=color_c[k])
    for k in range(K):
        plt.plot(kmus[k,0],kmus[k,1],'k*', markersize=16)
        plt.plot(mus[k][0],mus[k][1],'ko', fillstyle='none',markeredgewidth=3,markersize=16)
    '''
    return mus, sigma, r, zs
    

def sigma_cons(sigma, constraints='full'):
    if constraints=='isotropic':
        sigma = [np.mean(np.diag(sigma_k),0) * np.eye(2) for sigma_k in sigma]
    elif constraints=='diagonal':
        sigma = [np.diag(np.diag(sigma_k)) for sigma_k in sigma]
    elif constraints=='full':
        sigma = sigma
    return sigma

def GMMInit(data=None, K=None, constraints=None):
    import KMeans
    zs, Js, mus = KMeans.KMeansMin(data, K)
    sigma = [[] for _ in range(K)]
    for k in range(K):
        sigma[k] = np.cov(data[zs == k].T, bias=False)
    return mus, sigma_cons(sigma, constraints)

def GMM(data=None, K=1, constraints="full", nit=25):
    mus, sigma = GMMInit(data,K,constraints) # k dimension, k dimension    
    LL = []
    pi = [1/K for _ in range(K)] # k dimension
    for i in range(nit):
        r, N = e_step(data, K, mus, sigma, pi) # n*k dimension, k dimension
        mus, sigma, pi = m_step(data, K, constraints, r, N)
        print(i)
        LLn = []
        for n in range(len(data)):
            loglk = []
            for k in range(K):                
                loglk.append(pi[k]*multivariate_normal.pdf(data[n], mus[k], sigma[k]))
            LLn.append(np.sum(loglk))
        LL.append(np.sum(np.log(LLn)))
    return mus, sigma, pi, r, N, LL

def e_step(data, K, mus, sigma, pi):
    r = np.zeros((len(data), K))
    for n in range(len(data)):
        for k in range(K):
            r[n][k] = pi[k] * multivariate_normal.pdf(data[n], mus[k], sigma[k])
            r[n][k] /= np.sum([pi[j] * multivariate_normal.pdf(data[n], mus[j], sigma[j]) for j in range(K)])
    N = np.sum(r, axis=0)
    return r, N

def m_step(data, K, constraints, r, N):
    mus = np.zeros((K, len(data[0])))    
    for k in range(K):
        for n in range(len(data)):
            mus[k] += r[n][k] * data[n]
        mus[k] = 1/N[k]*mus[k]
    sigma = [np.zeros((len(data[0]), len(data[0]))) for _ in range(K)]
    for k in range(K):
        for n in range(len(data)):
            sigma[k] += r[n][k] * np.array(np.outer(data[n]-mus[k], data[n]-mus[k]))
        sigma[k] = 1/N[k]*sigma[k]
    pi = [N[k]/len(data) for k in range(K)]
    return mus, sigma_cons(sigma, constraints), pi

def assign(data, K, mus, sigma):
    probas = []
    for n in range(len(data)):
        probas.append([multivariate_normal.pdf(data[n], mus[k], sigma[k]) for k in range(K)])
    cluster = []
    for proba in probas:
        cluster.append(np.argmax(proba))
    return cluster
import numpy as np
import matplotlib.pyplot as plt
import GMM2D


POIs = np.loadtxt('data/POIs')
mus, sigma, r, zs = GMM2D.GMM2D(POIs, 3, 'full', 25)
color_r = [r[n]/np.sum(r[n]) for n in range(len(POIs))]
for n in range(len(POIs)):
    plt.plot(POIs[n][0], POIs[n][1], '.', color=(color_r[n][0],color_r[n][1],color_r[n][2]), markersize=4)
plt.savefig('images/resposibilities.svg')
plt.show()
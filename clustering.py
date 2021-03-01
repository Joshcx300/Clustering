import numpy as np
import pdb
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.cm as colormaps


def main():
    ##Set parameter used for clustering##
    k = 2

    ##Load data points, results in array with shape: (1500,3)##
    txt = np.loadtxt("data.txt")

    ##Seperate Data as np.loadtxt combines labels with points##
    labels = txt[:,2]
    x = txt[:,[0,1]]

    ##Create classifiers##
    model1 = AgglomerativeClustering(n_clusters = k, linkage='single')
    model2 = AgglomerativeClustering(n_clusters = k, linkage='complete')
    model3 = KMeans(n_clusters = k)

    y1 = model1.fit_predict(x)
    y2 = model2.fit_predict(x)
    y3 = model3.fit_predict(x)




    plots = [y1, y2, y3]
    titles = ["Agglomerative Clustering with single linkage", "Agglomerative Clustering with Complete Linkage", "K-Means Clustering"]

    for i in range(len(plots)):
        plt.figure(1)
        plt.title(titles[i])
        plt.clf()
        s1 = plt.scatter(x[:, 0], x[:, 1], s=6, c = plots[i], cmap='rainbow')
        ax = plt.gca()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        #pdb.set_trace()
        plt.show()



    pdb.set_trace()




    



if __name__ == '__main__':
    main()
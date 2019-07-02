import csv
import numpy as np
import random
import time
import matplotlib.pyplot as plt


class Data(object):

    def __init__(self):
        self.data = csv.reader(open('../dataset/Frogs_MFCCs.csv', 'r'))

        self.X = []
        self.Y = []
        next(self.data)
        count = 0
        for item in self.data:
            if count % 7 == 0:
                self.X.append(item[:-1])
                self.Y.append(item[-1])
            count += 1
        self.data_size = len(self.Y)
        self.dimension = len(self.X[0])

        self.C = {}
        for i in range(self.data_size):
            if self.Y[i] in self.C.keys():
                self.C[self.Y[i]].add(i)
            else:
                self.C[self.Y[i]] = set()

    def writeback(self, mark, cluster):
        writer = csv.writer(open('KMeans_PCA.csv', 'w', newline=''))
        line = []
        for i in range(len(cluster)):
            line.append(str(i) + 'class')
            line.append(len(cluster))
        writer.writerow(line)

        for i in range(self.data_size):
            tmp = list(self.X[i])
            tmp.append(self.Y[i])
            tmp.append(mark[i])
            writer.writerow(tmp)

    def PCA(self, threshold):

        # shape (data.data_size, self.dimension)
        x = np.array(self.X).astype(np.float)

        # 进行中心化
        # shape (self.dimension, )
        ave_x = np.mean(x, axis=0)
        x = x - ave_x # 广播

        # 计算协方差矩阵
        cov_x = np.cov(x.T)

        # 计算特征值
        eigen_value, eigen_vector = np.linalg.eig(cov_x)

        # 计算特征值从大到小排序的索引
        eigen_sort_index = np.argsort(- eigen_value)

        # 计算特征值的和，用于判断threshold
        sum_eigen_value = np.sum(eigen_value)

        # 计算所取到特征值的序数（从大到小） m
        current_sum_eigen_value = 0
        m = 0
        for i in range(len(eigen_value)):
            current_sum_eigen_value += eigen_value[i]
            m = i
            if current_sum_eigen_value / sum_eigen_value > threshold:
                break

        # 计算投影矩阵 W
        W = np.array(eigen_vector.T[eigen_sort_index[:m + 1]]).T

        # 得到低位结果 (W.T * x.T).T
        self.X = np.dot(x, W)
        self.dimension = len(self.X[0])
        print(self.X.shape)


class Statistic(object):

    def __init__(self):
        pass

    def purity(self, k, cluster, data):
        purity = 0
        for i in range(k):
            max = 0
            for item_c in data.C.values():
                intersec_len = len(item_c & cluster[i])
                if intersec_len > max:
                    max = intersec_len

            purity += max

        purity /= data.data_size

        return purity

    def RI(self, data, mark):
        a, b, c, d = 0, 0, 0, 0
        pred1 = mark[np.newaxis, :].repeat(data.data_size, axis=0)
        pred2 = mark[:, np.newaxis].repeat(data.data_size, axis=1)
        real = np.array(data.Y)
        real1 = real[np.newaxis, :].repeat(data.data_size, axis=0)
        real2 = real[:, np.newaxis].repeat(data.data_size, axis=1)
        a = np.sum(np.triu(((pred1 == pred2) == (real1 == real2)).astype(np.int), 1))
        b = np.sum(np.triu(((pred1 == pred2) == (real1 != real2)).astype(np.int), 1))
        c = np.sum(np.triu(((pred1 != pred2) == (real1 == real2)).astype(np.int), 1))
        d = np.sum(np.triu(((pred1 != pred2) == (real1 != real2)).astype(np.int), 1))

        return (a + d) / (a + b + c + d)

    def visualize(self, data, cluster):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        key2color = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'skyblue', 7: 'purple', 8: 'black',
                     9: 'grey'}
        count = 1
        for clu in cluster:
            coord_x = []
            coord_y = []
            for index in clu:
                coord_x.append(data.X[index][0])
                coord_y.append(data.X[index][1])
            ax.scatter(coord_x, coord_y, s=1, c=key2color[count])
            count += 1

        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('HC graph.png')
        plt.show()


def HC(k, data):

    # 初始化：把每个点设为一个类
    cluster = [{i} for i in range(data.data_size)]
    cluster_calculate = np.array([[data.X[i]] for i in range(data.data_size)]).astype(np.float)
    cluster_center = np.mean(cluster_calculate, axis=1)
    cluster_center_weight = np.ones(data.data_size)

    Cl = cluster_center[np.newaxis, :].repeat(data.data_size, axis=0)
    Cr = cluster_center[:, np.newaxis].repeat(data.data_size, axis=1)

    M = np.sqrt(np.sum(np.square(Cl - Cr), axis=2)) + np.identity(data.data_size) * 100

    # 设置当前聚类簇数
    q = data.data_size

    # 开始聚类
    while q > k:
        nearest = np.argmin(M)
        nearest_x, nearest_y = nearest // q, nearest % q

        # 合并簇的中心，合并簇，删除对应矩阵行列
        cluster_center[nearest_x] = (cluster_center[nearest_x] * cluster_center_weight[nearest_x]
                                    + cluster_center[nearest_y] * cluster_center_weight[nearest_y]) / \
                                    (cluster_center_weight[nearest_x] + cluster_center_weight[nearest_y])
        cluster_center = np.delete(cluster_center, nearest_y, axis=0)
        cluster_center_weight[nearest_x] = cluster_center_weight[nearest_x] + cluster_center_weight[nearest_y]
        cluster_center_weight = np.delete(cluster_center_weight, nearest_y, axis=0)
        cluster[nearest_x] = cluster[nearest_x] | cluster[nearest_y]
        cluster.pop(nearest_y)
        M = np.delete(M, nearest_y, axis=0)
        M = np.delete(M, nearest_y, axis=1)

        # 更新和nearest_x 相关簇的距离
        tmpM = np.sqrt(np.sum(np.square(cluster_center - cluster_center[nearest_x]), axis=1))
        tmpM[nearest_x] = 100
        M[:, nearest_x] = tmpM
        M[nearest_x] = tmpM

        q -= 1

    comm = Statistic()
    # 计算 mark
    mark = np.zeros(data.data_size).astype(np.int)
    for i in range(k):
        for index in cluster[i]:
            mark[index] = i
    purity = comm.purity(k, cluster, data)
    RI = comm.RI(data, mark)
    comm.visualize(data, cluster)

    return (purity, RI)




if __name__ == '__main__':

    # 运行kmeans
    print('begin train')
    start = time.perf_counter()
    data = Data()
    data.PCA(0.5)
    purity, RI = HC(8, data)
    print('purity:', purity, '  RI:', RI)
    print('finished train in', time.perf_counter() - start, 's')



import csv
import numpy as np
import random
import time

# 宏
dimension = 22


class Data(object):

    def __init__(self):
        self.data = csv.reader(open('../dataset/Frogs_MFCCs.csv', 'r'))

        self.X = []
        self.Y = []
        next(self.data)
        for item in self.data:
            self.X.append(item[:-1])
            self.Y.append(item[-1])
        self.data_size = len(self.Y)

        self.C = {}
        for i in range(self.data_size):
            if self.Y[i] in self.C.keys():
                self.C[self.Y[i]].add(i)
            else:
                self.C[self.Y[i]] = set()


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


def kmeans(k, data):

    # 随机选取 k 个样本作为初始均值向量
    rand = set()
    while len(rand) != k:
        rand = set()
        for i in range(k):
            rand.add(random.randint(0, data.data_size))

    ave_vec = np.zeros((k, dimension))
    for i in range(k):
        ave_vec[i] = np.array(data.X[i])

    # 测试
    # print(ave_vec)

    # repeat 迭代开始
    cluster = [set() for i in range(k)]
    mark = np.zeros((data.data_size, ))
    judge_ave_vec_update = True
    while judge_ave_vec_update:
        judge_ave_vec_update = False

        cluster = [set() for i in range(k)]
        cluster_calculate = [[] for i in range(k)]
        x = np.zeros((data.data_size, k, dimension))
        u = np.zeros((data.data_size, k, dimension))
        for i in range(data.data_size):
            x[i, :] = data.X[i]
        for i in range(k):
            u[:, i] = ave_vec[i]

        # 计算距离
        d = np.array(np.sqrt(np.sum(np.square(x - u), axis=2)))

        # 计算标记 shape(data.data_size, )
        mark = np.argmin(d, axis=1)

        # 划入相应的cluster
        for i in range(data.data_size):
            cluster_calculate[mark[i]].append(data.X[i])
            cluster[mark[i]].add(i)

        # 计算新的均值向量，并且更新
        new_ave_vec = ave_vec
        for i in range(k):
            new_ave_vec[i] = np.sum(np.array(cluster_calculate[i]).astype(np.float), axis=0) / len(cluster_calculate[i])

            if (new_ave_vec[i] - ave_vec[i]).any():
                new_ave_vec[i] = ave_vec[i]
                judge_ave_vec_update = True
            else:
                pass
    # end repeat

    comm = Statistic()
    purity = comm.purity(k, cluster, data)
    # print('finished purity')
    RI = comm.RI(data, mark)
    return purity, RI


if __name__ == '__main__':
    data = Data()
    # k的取值
    genre = set()
    for item in data.Y:
        if item not in genre:
            genre.add(item)
    k = len(genre)

    # 运行kmeans
    print('begin train')
    start = time.perf_counter()
    purity, RI = kmeans(k, data)
    print('purity:', purity, '  RI:', RI)
    print('finished train in', time.perf_counter() - start, 's')



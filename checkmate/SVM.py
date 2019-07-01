import osqp
from scipy import sparse
import csv
import numpy as np
import copy
import time

sigma = 2
C = 2
gap = 4

class Data(object):

    '''数据加载器'''
    def __init__(self):
        self.data = {'train': csv.reader(open('dataset/trainset.csv', 'r')),
                     'test': csv.reader(open('dataset/testset.csv', 'r'))}
        self.coord = []
        next(self.data['train'])
        for item in self.data['train']:
            self.coord.append(self.transformer(item))

        # 取其中部分数据
        tmpcoord = []
        for i in range(0, len(self.coord), gap):
            tmpcoord.append(self.coord[i])
        self.coord = tmpcoord

        self.x = []
        for item in self.coord:
            self.x.append(self.toArray(item))

        # 计算核矩阵


    def transformer(self, item):
        a = ord(item[0]) - ord('a') + 1
        b = int(item[1])
        c = ord(item[2]) - ord('a') + 1
        d = int(item[3])
        e = ord(item[4]) - ord('a') + 1
        f = int(item[5])

        return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'class': item[6]}

    def toArray(self, item):
        tmpx = np.zeros((6))
        for k in range(6):
            tmpx[k] = item[chr(k + ord('a'))]
        return tmpx


# 评估函数
def macroF1(TP, FP, TN, FN):
    F = {}
    P = {}
    R = {}
    F1 = 0
    for key in TP.keys():
        if TP[key] + FP[key] == 0:
            P[key] = 1
        else:
            P[key] = float(TP[key])/(TP[key] + FP[key])
        if TP[key] + FN[key] == 0:
            R[key] = 1
        else:
            R[key] = float(TP[key])/(TP[key] + FN[key])
        if P[key] + R[key] == 0:
            F[key] = 2 * P[key] * R[key]
        else:
            F[key] = (2 * P[key] * R[key]) / (P[key] + R[key])
        F1 += F[key]
    return F1 / len(TP.keys())


def microF1(TP, FP, TN, FN):
    TP_total = 0
    FP_total = 0
    TN_total = 0
    FN_total = 0
    for key in TP.keys():
        TP_total += TP[key]
        FP_total += FP[key]
        TN_total += TN[key]
        FN_total += FN[key]
    print(TP_total, FP_total, TN_total, FN_total)
    if TP_total + FP_total == 0:
        P = 1
    else:
        P = float(TP_total) / (TP_total + FP_total)
    if TP_total + FN_total == 0:
        R = 1
    else:
        R = float(TP_total) / (TP_total + FN_total)
    if P + R == 0:
        F1 = 2 * P * R
    else:
        F1 = (2 * P * R) / (P + R)
    return F1


def kernel(xi, xj, sigma):
    if sigma == 0:
        return np.dot(xi, xj)
    else:
        return np.exp(- (np.linalg.norm(xi - xj) ** 2) / (sigma ** 2))


data = Data()
length = len(data.coord)
Kl = np.zeros((length, length, 6))
Kr = np.zeros((length, length, 6))
for i in range(len(data.coord)):
    Kl[i, :] = np.array(data.x[i])
for i in range(len(data.coord)):
    Kr[:, i] = np.array(data.x[i])
yl = {}
yr = {}




def softSVM(SVMclass, sigma, C):

    # 计算出P
    P = np.zeros((length, length))

    yl[SVMclass] = np.zeros((length, length))
    yr[SVMclass] = np.zeros((length, length))
    for i in range(len(data.coord)):
        if SVMclass == data.coord[i]['class']:
            yl[SVMclass][i] = 1
        else:
            yl[SVMclass][i] = -1
    for i in range(len(data.coord)):
        if SVMclass == data.coord[i]['class']:
            yr[SVMclass][:, i] = 1
        else:
            yr[SVMclass][:, i] = -1

    if sigma == 0:
        P = yl[SVMclass] * yr[SVMclass] * np.sum(Kl * Kr, axis=2)
    else:
        P = yl[SVMclass] * yr[SVMclass] * np.exp(- (np.sum(np.square(Kl - Kr), axis=2)) / sigma ** 2)

    # for i in range(len(data.coord)):
    #     print(i)
    #     for j in range(len(data.coord)):
    #         start = time.clock()
    #         mul = 1
    #         xi = data.x[i]
    #         xj = data.x[j]
    #         time1 = time.clock()
    #         mul *= kernel(xi, xj, sigma)
    #         time2 = time.clock()
    #         if SVMclass == data.coord[i]['class']:
    #             mul *= 1
    #         else:
    #             mul *= -1
    #         if SVMclass == data.coord[j]['class']:
    #             mul *= 1
    #         else:
    #             mul *= -1
    #         time3 = time.clock()
    #         P[i, j] = mul
    #         time4 = time.clock()
    #         # print(time1 - start, time2 - time1, time3 - time2, time4 - time3)
    print('transfer')
    P = sparse.csc_matrix(P)
    # 计算 q
    q = - np.ones((length))

    l = np.zeros((length + 1))

    u = np.ones((length + 1)) * C
    u[-1] = 0

    A = np.identity((length + 1))
    A = np.delete(A, -1, axis=1)
    A[-1] = yr[SVMclass][-1]
    A = sparse.csc_matrix(A)

    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=l, u=u)

    print('begin solve')
    result = m.solve()

    return result

def calculateb(SVMclass, a):
    b = yr[SVMclass][0, 0]
    a_np = np.array(a)
    if sigma == 0:
        b -= np.sum(a_np * yr[SVMclass][0] * np.sum(Kl[0] * Kr[0], axis=1))
    else:
        b -= np.sum(a_np * yr[SVMclass][0] * np.exp(- (np.sum(np.square(Kl[0] - Kr[0]), axis=1)) / sigma ** 2))
    return b

def calculatef(SVMclass, a, b, x):
    a_np = np.array(a)
    if sigma == 0:
        f = np.sum(a_np * yr[SVMclass][0] * np.sum(x * Kr[0], axis=1)) + b
    else:
        f = np.sum(a_np * yr[SVMclass][0] * np.exp(- (np.sum(np.square(x - Kr[0]), axis=1)) / sigma ** 2)) + b
    return f

def multiclassSVM():
    acc_items = 0
    total_items = 0
    TP = {'draw': 0, 'zero': 0, 'one': 0, 'two': 0, 'three': 0, 'four': 0, 'five': 0, 'six': 0, 'seven': 0, 'eight': 0,
          'nine': 0, 'ten': 0, 'eleven': 0, 'twelve': 0, 'thirteen': 0, 'fourteen': 0, 'fifteen': 0, 'sixteen': 0}
    FP = copy.deepcopy(TP)
    FN = copy.deepcopy(TP)
    TN = copy.deepcopy(TP)


    # 构造多个SVM
    result = {}
    print('Begin train')
    for SVMclass in TP.keys():
        result[SVMclass] = softSVM(SVMclass, sigma, C)
        print('finished SVM: SVM' + SVMclass)

    print('Begin test')
    # 开始测试
    next(data.data['test'])
    for item in data.data['test']:
        sample = data.transformer(item)
        # sample = data.coord[300]
        # 计算多种分类的得分f
        f = {}
        for SVMclass in TP.keys():
            a = result[SVMclass].x
            b = calculateb(SVMclass, a)
            f[SVMclass] = calculatef(SVMclass, a, b, data.toArray(sample))

        # 计算预测值 ypred
        ypred = None
        max = -100
        for SVMclass in TP.keys():
            if f[SVMclass] > max:
                max = f[SVMclass]
                ypred = SVMclass

        # 统计评估
        if ypred == sample['class']:
            acc_items += 1
        else:
            print(total_items + 1, ypred, sample['class'])
        for key in TP.keys():
            if key == ypred:
                if key == sample['class']:
                    TP[key] += 1
                else:
                    FP[key] += 1
            else:
                if key == sample['class']:
                    FN[key] += 1
                else:
                    TN[key] += 1

        total_items += 1
        if total_items % 100 == 0:
            accuracy = float(acc_items) / total_items
            MacroF1 = macroF1(TP, FP, TN, FN)
            MicroF1 = microF1(TP, FP, TN, FN)
            print('total:', total_items, 'Accuracy:', accuracy, 'Macro F1:', MacroF1, 'Micro F1:', MicroF1)

    # 计算准确率
    accuracy = float(acc_items) / total_items
    MacroF1 = macroF1(TP, FP, TN, FN)
    MicroF1 = microF1(TP, FP, TN, FN)

    # 输出
    print('total:', total_items, 'Accuracy:', accuracy, 'Macro F1:', MacroF1, 'Micro F1:', MicroF1)




if __name__ == '__main__':
    multiclassSVM()




import osqp
from scipy import sparse
import csv
import numpy as np

class Data(object):

    '''数据加载器'''
    def __init__(self):
        self.data = {'train': csv.reader(open('dataset/trainset.csv', 'r')),
                     'test': csv.reader(open('dataset/testset.csv', 'r'))}
        self.coord = []
        next(self.data['train'])
        for item in self.data['train']:
            self.coord.append(self.transformer(item))

    def transformer(self, item):
        a = ord(item[0]) - ord('a') + 1
        b = int(item[1])
        c = ord(item[2]) - ord('a') + 1
        d = int(item[3])
        e = ord(item[4]) - ord('a') + 1
        f = int(item[5])

        return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'f': f, 'class': item[6]}

def kernel(xi, xj, sigma):
    if sigma == 0:
        return np.dot(xi, xj)
    else:
        return np.exp(- (np.linalg.norm(xi - xj) ** 2) / (sigma ** 2))

def QPsolver(data, QPclass, sigma, C):

    # 计算出P
    P = sparse.lil_matrix((len(data.coord), len(data.coord)))
    for i in range(0, len(data.coord), 64):
        print(i)
        for j in range(0, len(data.coord), 64):
            mul = 1
            xi = np.array((6))
            xj = np.array((6))
            for k in range(6):
                xi[k] = data.coord[i][chr(k + ord('a'))]
                xj[k] = data.coord[j][chr(k + ord('a'))]
            mul *= kernel(xi, xj, sigma)
            if QPclass == data.coord[i]['class']:
                mul *= 1
            else:
                mul *= -1
            if QPclass == data.coord[j]['class']:
                mul *= 1
            else:
                mul *= -1
            P[i, j] = mul
    P = sparse.csc_matrix(P)

    # 计算 q
    q = - np.ones((len(data.coord)))

    l = np.zeros((len(data.coord) + 1))

    u = np.ones((len(data.coord) + 1)) * C
    u[-1] = 0

    A = np.identity((len(data.coord) + 1))
    A[-1] = 1
    A = sparse.csc_matrix(A)

    m = osqp.OSQP()
    m.setup(P=P, q=q, A=A, l=l, u=u)

    print('begin solve')
    result = m.solve()

    return result

def calculateb(x)


def multiclassSVM(data):
    result = QPsolver(data, 'draw', 1, 2)
    b = calculateb(result.x)

if __name__ == '__main__':
    data = Data()
    multiclassSVM(data)




import csv
import numpy as np
import copy
import time
from graphviz import Graph


class Node(object):

    def __init__(self):
        self.leave = False
        self.child = {}
        self.attr = None
        self.leaveclass = None


class Data(object):

    def __init__(self):
        self.data = {'train': csv.reader(open('../dataset/trainset.csv', 'r')),
                     'test': csv.reader(open('../dataset/testset.csv', 'r'))}
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


def Ent(D):
    sum = {'draw': 0, 'zero': 0, 'one': 0, 'two': 0, 'three': 0, 'four': 0, 'five': 0, 'six': 0, 'seven': 0, 'eight': 0,
          'nine': 0, 'ten': 0, 'eleven': 0, 'twelve': 0, 'thirteen': 0, 'fourteen': 0, 'fifteen': 0, 'sixteen': 0}
    total_item = len(D)
    for item in D:
        sum[item['class']] += 1
    ent = 0
    for key in sum.keys():
        pk = sum[key] / total_item
        if pk == 0:
            continue
        ent += - pk * np.log2(pk)
    return ent


def Gain(D, a):
    gain = 0
    gain += Ent(D)
    Dv_count = {x: 0 for x in range(1, 9)}
    Dv = {x: [] for x in range(1, 9)}
    for sample in D:
        Dv[sample[a]].append(sample)
        Dv_count[sample[a]] += 1
    total = len(D)
    for v in range(1, 9):
        if Dv_count[v] == 0:
            continue
        gain += - Dv_count[v] / total * Ent(Dv[v])
    return gain


def chooseBestFeature(dataset, A):
    max = 0
    bestChoice = None
    for attr in A:
        if attr == 'class':
            continue
        increase = Gain(dataset, attr)
        if increase > max:
            bestChoice = attr
            max = increase
    return bestChoice


dot = Graph(comment='Tree', format='png', engine='sfdp')
dot.graph_attr['nodesep'] = str(0.02)
dot.graph_attr['ranksep'] = str(0.02)
dot.attr('node', shape='plaintext')
dot.attr('node', color='none')
dot.attr('node', margin='0')
dot.attr('node', width='0')
dot.attr('node', height='0')
dot.graph_attr['overlap'] = 'prism10' # 5
dot.graph_attr['overlap_shrink'] = 'true'
dot.graph_attr['concentrate'] = 'true'
# dot.graph_attr['splines'] = 'curved'
dot.graph_attr['fontsize'] = str(1.0)
name_dot = {}
key2color = {1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'skyblue', 7: 'purple', 8: 'black'}
class2num = {'draw': 'x', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
             'seven': '7', 'eight': '8', 'nine': '9', 'ten': '/a', 'eleven': '/b', 'twelve': '/c', 'thirteen': '/d',
             'fourteen': '/e', 'fifteen': '/f', 'sixteen': '/g'}
num_draw = 0

def treeScan(node):
    # global num_draw
    # num_draw += 1
    # if num_draw % 10000 == 0:
    #     dot.view()
    #     input()
    if node.leave:
        if not (node.leaveclass in name_dot.keys()):
            name_dot[node.leaveclass] = 1
        else:
            name_dot[node.leaveclass] += 1
        name = node.leaveclass + str(name_dot[node.leaveclass])
        dot.node(name, class2num[node.leaveclass])
        return name
    else:
        if not (node.attr in name_dot.keys()):
            name_dot[node.attr] = 1
        else:
            name_dot[node.attr] += 1
        name = node.attr + str(name_dot[node.attr])
        dot.node(name, node.attr)
        for key in node.child.keys():
            point = treeScan(node.child[key])
            dot.edge(name, point, **{'color': key2color[key]})
        return name



def treeGenerate(D, A):
    node = Node()

    # 判断是否同一类别
    oneclass = None
    diff = False
    for item in D:
        if oneclass == None:
            oneclass = item['class']
        if oneclass != item['class']:
            diff = True
            break
    if not diff:
        node.leave = True
        node.leaveclass = oneclass
        return node

    # 判断是否样本相同
    same = False
    oneAttr = {}
    if A == set():
        same = True
    else:
        diff = False
        for item in D:
            if oneAttr == {}:
                for attr in A:
                    oneAttr[attr] = item[attr]
            for attr in A:
                if oneAttr[attr] != item[attr]:
                    diff = True
            if diff:
                break
        if not diff:
            same = True
    if same:
        # 标记叶节点 且类别为最多的类
        node.leave = True
        classes = {}
        for sample in D:
            if sample['class'] in classes.keys():
                classes[sample['class']] += 1
            else:
                classes[sample['class']] = 1
        max = 0
        chooseClass = None
        for m in classes.keys():
            if classes[m] > max:
                max = classes[m]
                chooseClass = m
        node.leaveclass = chooseClass
        return node


    # 选择最优划分
    bestFeature = chooseBestFeature(D, A)
    node.attr = bestFeature

    for i in range(1, 9):
        Dv = []
        for sample in D:
            if sample[bestFeature] == i:
                Dv.append(sample)
        if Dv == []:
            # 分支节点标记为叶节点 且类别为最多的类
            newnode = Node()
            node.child[i] = newnode
            newnode.leave = True
            classes = {}
            for sample in D:
                if sample['class'] in classes.keys():
                    classes[sample['class']] += 1
                else:
                    classes[sample['class']] = 1
            max = 0
            chooseClass = None
            for m in classes.keys():
                if classes[m] > max:
                    max = classes[m]
                    chooseClass = m
            newnode.leaveclass = chooseClass
        else:
            Av = copy.deepcopy(A)
            Av.remove(bestFeature)
            node.child[i] = treeGenerate(Dv, Av)
    return node


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
        print('here1')
    else:
        P = float(TP_total) / (TP_total + FP_total)
    if TP_total + FN_total == 0:
        R = 1
        print('here2')
    else:
        R = float(TP_total) / (TP_total + FN_total)
    if P + R == 0:
        F1 = 2 * P * R
        print('here3')
    else:
        F1 = (2 * P * R) / (P + R)
    return F1


def walkTree(root, sample):
    node = root
    while not node.leave:
        node = node.child[sample[node.attr]]
    return node.leaveclass


def createTree():

    # 建树
    check = Data()
    A = {'a', 'b', 'c', 'd', 'e', 'f'}
    D = check.coord
    root = treeGenerate(D, A)
    print('finished create tree')
    treeScan(root)
    print('finished scan tree')
    # dot.view()
    # print('finished show')

    # 测试
    acc_items = 0
    total_items = 0
    TP = {'draw': 0, 'zero': 0, 'one': 0, 'two': 0, 'three': 0, 'four': 0, 'five': 0, 'six': 0, 'seven': 0, 'eight': 0,
          'nine': 0, 'ten': 0, 'eleven': 0, 'twelve': 0, 'thirteen': 0, 'fourteen': 0, 'fifteen': 0, 'sixteen': 0}
    FP = copy.deepcopy(TP)
    FN = copy.deepcopy(TP)
    TN = copy.deepcopy(TP)

    next(check.data['test'])
    for testitem in check.data['test']:
        sample = check.transformer(testitem)
        ypred = walkTree(root, sample)

        # 统计正确预测的个数
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
        print(ypred, sample['class'], ':', )
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
    start = time.perf_counter()
    print('Train start')
    createTree()
    total_time = time.perf_counter() - start
    print('Train finished after:', total_time)
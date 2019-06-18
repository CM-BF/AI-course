import csv
import numpy as np
import time
import copy


class Data(object):

    def __init__(self):
        self.data = {'train': csv.reader(open('dataset/trainset.csv', 'r')),
                     'test': csv.reader(open('dataset/testset.csv', 'r'))}
        self.coord = []
        next(self.data['train'])
        for item in self.data['train']:
            self.coord.append(self.transformer(item))

    def transformer(self, item):
        x = np.sqrt(np.square(ord(item[0]) - ord(item[2])) + np.square(int(item[1]) - int(item[3])))
        y = np.sqrt(np.square(ord(item[0]) - ord(item[4])) + np.square(int(item[1]) - int(item[5])))
        z = np.sqrt(np.square(ord(item[2]) - ord(item[4])) + np.square(int(item[3]) - int(item[5])))
        return {'x': x, 'y': y, 'z': z, 'class': item[6]}

    def distance(self, k, raw_item):
        dis = []
        value_range = 1
        Flag = True
        item = self.transformer(raw_item)
        x = item['x']
        y = item['y']
        z = item['z']
        while Flag:
            for n in range(len(self.coord)):
                tmp = np.sqrt(np.square(self.coord[n]['x'] - x + self.coord[n]['y'] - y)
                              + np.square(self.coord[n]['x'] - x + self.coord[n]['z'] - z)
                              + np.square(self.coord[n]['z'] - z + self.coord[n]['y'] - y))
                if tmp < value_range:
                    dis.append([tmp, self.coord[n]['class']])
            if len(dis) >= k:
                Flag = False
            else:
                value_range *= 2
                dis = []
        ddis = []
        for i in range(k):
            min_key = -1
            min = 10000
            for j in range(len(dis)):
                if dis[j][0] < min:
                    min_key = j
                    min = dis[j][0]
            ddis.append(dis[min_key])
            dis.pop(min_key)
        return ddis[0:k]


def macroF1(TP, FP, TN, FN):
    F = {}
    P = {}
    R = {}
    F1 = 0
    for key in TP.keys():
        P[key] = float(TP[key])/(TP[key] + FP[key])
        R[key] = float(TP[key])/(TP[key] + FN[key])
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
    P = float(TP_total) / (TP_total + FP_total)
    R = float(TP_total) / (TP_total + FN_total)
    F1 = (2 * P * R) / (P + R)
    return F1



def test(k):
    acc_items = 0
    total_items = 0
    TP = {'draw': 0, 'zero': 0, 'one': 0, 'two': 0, 'three': 0, 'four': 0, 'five': 0, 'six': 0, 'seven': 0, 'eight': 0,
          'nine': 0, 'ten': 0, 'eleven': 0, 'twelve': 0, 'thirteen': 0, 'fourteen': 0, 'fifteen': 0, 'sixteen': 0}
    FP = copy.deepcopy(TP)
    FN = copy.deepcopy(TP)
    TN = copy.deepcopy(TP)


    next(check.data['test'])
    for testitem in check.data['test']:
        knear = check.distance(k, testitem)
        classes = {}
        # 统计k近邻结果
        for i in range(k):
            if knear[i][1] in classes.keys():
                classes[knear[i][1]] += 1
            else:
                classes[knear[i][1]] = 1
        max_key = None
        max_value = 0
        for key in classes.keys():
            if classes[key] > max_value:
                max_key = key
                max_value = classes[key]

        # 统计正确预测的个数
        if max_key == testitem[6]:
            acc_items += 1
        for key in TP.keys():
            if key == max_key:
                if key == testitem[6]:
                    TP[key] += 1
                else:
                    FP[key] += 1
            else:
                if key == testitem[6]:
                    TN[key] += 1
                else:
                    FN[key] += 1

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



if __name__ == "__main__":
    check = Data()
    start = time.clock()
    print('Train start')
    test(5)
    total_time = time.clock() - start
    print('Train finished after:', total_time)




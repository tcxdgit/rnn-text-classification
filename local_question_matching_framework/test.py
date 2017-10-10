# coding:utf-8
# author:Tian
import sys
sys.path.append("..")
from rnn_text_classification.classify import Classify_CN
import os
import pandas as pd
from pprint import pprint
from sklearn.metrics import accuracy_score

module_path = './runs/people2014'
classify = Classify_CN(module_path)

# test_data_path = '../work_space/people2014/dataset/test_set/userchat1_neg.txt'
test_data_path = '../work_space/suning/dataset/suning_biu_topic_data'

input = []
output_actu = []
for parent,dirnames,filenames in os.walk(test_data_path):
    # print(filenames)
    for filename in filenames:
        if filename[-1] == '~':
            continue
        else:
            pass
        for line in open(os.path.join(parent, filename)):
            line_clean = line.strip()
            input.append(line_clean)
            # output_actu.append(filename)
# for line in open(test_data_path, 'r'):
#     line_clean = line.strip()
#     input.append(line_clean)
    # output_actu.append(filename)

output_pred = []
# time_cost = []
count = 0
count_sort = 0
count_shuffle = 0
for sentence in input:
    # result, time = classify.getCategory(sentence)
    result = classify.getCategory(sentence)
    output_pred.append(result['value'])

    print(sentence)
    print(result['probability'])
    print('Pred label: ' + result['value']+'\n')

    if result['value'] == '1':
        count_sort += 1
    else:
        count_shuffle += 1

    count += 1
    # if result['value'] != output_actu[count]:
    #     print(sentence)
    #     print(result)
    #     print('Pred label: ' + result['value'])
    #     print('Actual label: ' + output_actu[count] + '\n')
    # count += 1
    # time_cost.append(time)
    # time_aver = sum(time_cost)/len(time_cost)

# y_actu = pd.Series(output_actu, name='Actual')
# y_pred = pd.Series(output_pred, name='Predicted')
# df_confusion = pd.crosstab(y_actu, y_pred, margins=True)
# accuracy = accuracy_score(y_actu, y_pred)
#
# print('plot confusion matrix:\n')
# pprint(df_confusion)
# print('Accuracy: {}'.format(accuracy))
# print('Average time:{}'.format(time_aver))

accuracy = float(count_sort) / count

print('总共有{}条数据，其中{}条判断为正类，{}条判断为负类，accuracy为{}'.format(count,count_sort,count_shuffle,accuracy))
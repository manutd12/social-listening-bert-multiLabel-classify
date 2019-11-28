#-*- coding : utf-8 -*-
# coding: utf-8
'''
1、将数据集 进行处理（去除停用词，数字，特殊字符等）
2、分割成训练集和开发集
3、将训练集保存到trainData.xls中
   将开发集保存到devData.xls中
   将开发集写成bert所需的example格式，保存到devData.pkl中
'''


from data_util import data
from bert import tokenization
import pickle
import xlwt
import xlrd
import os
from sklearn.model_selection import train_test_split

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def save_excel(fileName, texts,labels):
    writeexcel = xlwt.Workbook()    # 创建工作表
    sheet1 = writeexcel.add_sheet(u"Sheet1", cell_overwrite_ok = True)    # 创建sheet
    #生成其他行
    row = 0
    for x,y in zip(texts,labels):
        sheet1.write(row, 0, x)
        sheet1.write(row, 1, y)
        row+=1
    # 保存文件
    writeexcel.save(fileName)

def read_excel(input_file, col_number=None, sheet_name=None):
    excel = xlrd.open_workbook(input_file)
    sheet_names = excel.sheet_names()
    if len(sheet_names) > 1:
        if sheet_name is not None:
            sheet = excel.sheet_by_name(sheet_name)
            return sheet.col_values(col_number)
        else:
            print('please identify the sheet name!')
    else:
        sheet = excel.sheet_by_index(0)
        return sheet.col_values(col_number)

def count_label_number(lables):
    #统计各个类别的数据量
    label2num = defaultdict(lambda: 0)
    for label in lables:
        for label_ in label.split(","):
            label2num[label_] += 1
    return label2num

filePath = "../data/Product_10000.xlsx"  # 原数据
allData_save_path = "../data/allData_arterProcess.xls"  # 处理后的所有数据保存路径
trainData_save_path = "../data/trainData.xls"  # 训练集保存路径
devData_save_excel_path = "../data/devData.xls"  # 开发集(计算阈值使用)保存路径（excel）
devData_save_pkl_path = "../data/devData.pkl"  # 开发集(计算阈值使用)保存路径（pkl）
dataProcess = data()
cut_label = ['others']  # 去除某些类别的数据
stopWord_file = "../user_dict/null.txt"  # 处理数据时去除停用词
if not os.path.exists(trainData_save_path) or not os.path.exists(devData_save_excel_path):
    label_list, lower2originalLabel = dataProcess.get_label(filePath,cut_label)
    data_text, data_label = dataProcess.get_data(filePath, label_list, stopWord_file)

    # 去除低频字 认为字频为1的字是错别字，应该去除
    from collections import defaultdict
    dico=defaultdict(lambda: 0)
    for content in data_text:
        for i in content:
            dico[i] +=1
    lowfrquence_words = [word for word,frequence in dico.items() if frequence==1]
    for text in data_text:
        for word in text:
            if word in lowfrquence_words:
                text.remove(word)

    X = []
    Y = []
    for (text, label) in zip(data_text, data_label):
        text_a = " ".join(text)
        text_a = tokenization.convert_to_unicode(text_a)
        label = ",".join(label)
        label = tokenization.convert_to_unicode(label)
        X.append(text_a)
        Y.append(label)

    zipped = set(zip(X,Y))
    X, Y = zip(*zipped)

    # save all data into excel
    save_excel(allData_save_path, X, Y)

    # 切割训练集和开发集
    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=0.2)

    # save train data and dev data into excel
    save_excel(trainData_save_path, X_train, y_train)
    save_excel(devData_save_excel_path, X_dev, y_dev)

    # 将开发集写成bert所需的数据格式，并保存在pkl中，方便在eval时直接加载
    test_example = []
    for i, text_a in enumerate(X_dev):
        guid = "test-%d" % i
        label = X_dev[i]
        test_example.append(
            InputExample(guid=guid, text_a=text_a, label=label))
    print('________eval data len:', len(test_example))
    pickle.dump(test_example, open(devData_save_pkl_path, 'wb'))


else:
    texts = read_excel(trainData_save_path,0)
    labels = read_excel(trainData_save_path,1)
    zipped = set(zip(texts,labels))
    texts, labels = zip(*zipped)
    twohundred_texts = []
    twohundred_labels = []
    # 打印每个类别的数据量
    label2num = count_label_number(labels)
    for key,value in label2num.items():
        print(key + ":" + str(value))
    for text,label in zip(texts, labels):
        isbreak = False
        for label_ in label.split(","):
            if label2num[label_]<200:
                twohundred_texts.append(text)
                twohundred_labels.append(label)




    # save_excel("../data/数据量少于200的类别.xls", twohundred_texts, twohundred_labels)






#-*- coding : utf-8 -*-
# coding: utf-8
'''
process social-listening data
'''
import xlrd
import jieba
import codecs
import os
import numpy as np
import xlwt
class data(object):
    def get_label(self,input_file, cut_labels):
        labels = self._read_excel(input_file,2,sheet_name="Taxonomy")
        labelmap = dict()
        lower2originalLabel = dict() #小写后的标签对应的原始标签
        for i,label_ in enumerate(labels):
            if i ==0 or label_.lower() in cut_labels:
                continue
            if label_ not in lower2originalLabel:
                lower2originalLabel[label_.lower()] = label_
            labelmap[label_.lower()] = 1
        return list(labelmap.keys()), lower2originalLabel
    def remove(self,text):
        import re
        remove_chars = '[0-9’!"$%&\'()*+,-./:;<=>?，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
        return re.sub(remove_chars, '', text)
    def process_text(self,content,stopwords):
        self.remove(content)
        content = content.replace("[", "").replace(".]", "").replace("]", "").replace("<ob>", "").replace("</ob>", "").replace("<ng>", "").replace("</ng>", "").replace("<po>", "").replace("</po>", "").replace("<nu>", "").replace("</nu>", "")
        content = content.replace("=","").replace("&", "").replace('"', "").replace("-", "").replace("{", "").replace("}", "")
        content = content.replace("xxx", "").replace("<", "").replace(">", "").replace("//", "").replace("(", "")
        content = content.replace(")", "").replace("+", "").replace("^", "").replace("|", "").replace("...", "")
        content = content.replace("......", "").replace(".", "").replace(":", "").replace(",", "").replace(";", "").replace("!", "").replace("?", "")
        content_list = content.lower().strip().split()
        content_list = [content_.strip() for content_ in content_list if len(content_) <= 15 and (not self.contain_other(content_.strip())) and not self.contain_chinese(content_.strip()) and not self.contain_number(content_.strip()) and content_.strip() not in stopwords and not self.contain_specialChar(content_.strip())]
        return content_list
    def get_data(self,input_file, all_labels, stopwords_file, mode):
        stopwords = []
        if mode == 'train':
            stopwords = codecs.open(stopwords_file, 'r', encoding='UTF-8').readlines()
            stopwords = set([word.strip('\n\r') for word in stopwords])
        
        text = self._read_excel(input_file, 0)
        labels = self._read_excel(input_file, 1)
        # text = self._read_excel(input_file,2,sheet_name="5000")[1:]
        # labels = self._read_excel(input_file,0,sheet_name="5000")[1:]
        # text_2 = self._read_excel(input_file,2,sheet_name="Sheet1")[1:]
        # labels_2 = self._read_excel(input_file,0,sheet_name="Sheet1")[1:]
        # text.extend(text_2)
        # labels.extend(labels_2)
        docs = []
        labs = []
        original_docs = []
        for i,content in enumerate(text):
            content_list = self.process_text(content,stopwords)
            if len(content_list)==0 or len(content_list)==1:
                continue
            label_list = labels[i].replace("\n","").strip().strip(",").split(",")
            label_list = [label_.strip().lower() for label_ in label_list if label_.strip().lower() in all_labels]
            if len(label_list) ==0:
                continue
            docs.append(content_list)
            labs.append(label_list)
            original_docs.append(content)
        return docs,labs,original_docs

    def create_all_result_excel(self,fileName, eval_examples, y_pred, y_pre_correct, id2label):
        writeexcel = xlwt.Workbook()    # 创建工作表
        sheet1 = writeexcel.add_sheet(u"Sheet1", cell_overwrite_ok = True)    # 创建sheet
        row0 = ["语句","real_label", "预测出的label"]
        # 生成第一行
        for index,content in enumerate(row0):
            sheet1.write(0, index, content)
        #生成其他行
        for row,pre in enumerate(eval_examples):
            y_pre_correct_index = [index for index, y in enumerate(y_pre_correct[row]) if y==1]
            sheet1.write(row+1, 0, eval_examples[row].original_text)
            sheet1.write(row+1, 1, eval_examples[row].text_a)
            sheet1.write(row+1, 2, eval_examples[row].label)
            str1 = ''
            for j,index in enumerate(y_pre_correct_index):
                str1 +="（ " + id2label[index] + ":" + str(float('%.4f' % y_pred[row][index]))+" ）"
                str1 += ","
            sheet1.write(row+1, 3, str1)
            
            maxindex  = np.argmax(y_pred[row])
            str2 ="（ " + id2label[maxindex] + ":" + str(float('%.4f' % y_pred[row][maxindex]))+" ）"
            str2 += ","
            sheet1.write(row+1, 4, str2)
        # 保存文件
        writeexcel.save(fileName)

    def create_error_result_excel(self,input,output):
        original_text = self._read_excel(input,0)[1:]
        text = self._read_excel(input,1)[1:]
        real_label = self._read_excel(input,2)[1:]
        pre = self._read_excel(input,3)[1:]
        max_prob = self._read_excel(input,4)[1:]
        error_list = []
        i=0
        for label, probs in zip(real_label,pre):
            all_real_label = label.split(",")
            all_pre = probs.split(",")[:-1]
            if len(all_pre) ==0:
                error_list.append((original_text[i],text[i],label,probs,max_prob[i]))
            else:
                for j in all_pre:
                    if j.replace("（","").replace("）","").split(":")[0].strip() not in  all_real_label:
                        error_list.append((original_text[i],text[i],label,probs,max_prob[i]))
                        break
            i+=1
        writeexcel = xlwt.Workbook()    # 创建工作表
        sheet1 = writeexcel.add_sheet(u"Sheet1", cell_overwrite_ok = True)    # 创建sheet
        #生成其他行
        for row,pre in enumerate(error_list):
            sheet1.write(row, 0, error_list[row][0])
            sheet1.write(row, 1, error_list[row][1])
            sheet1.write(row, 2, error_list[row][2])
            sheet1.write(row, 3, error_list[row][3])
            sheet1.write(row, 4, error_list[row][4])
        writeexcel.save(output)

    #判读字符串是否包含中文，TRUE是包含中文
    def contain_chinese(self,check_str):
        for char in check_str: #判断每个字符
            if self.is_chinese(char):
                return True
        return False

    #判断字符串是否只有中文，True是只有中文
    def only_contain_chinese(self,check_str):
        for char in check_str:
            if not self.is_chinese(char):
                return False
        return True

    #判读字符串是否包含数字，TRUE是包含数字
    def contain_number(self,check_str):
        for char in check_str:
            if self.is_number(char):
                return True
        return False

    def only_contain_number(self,check_str):
        for char in check_str:
            if not self.is_number(char):
                return False
        return True

    #判读字符串是否包含英文，TRUE是包含英文
    def contain_english(self,check_str):
        for char in check_str:
            if self.is_english(char):
                return True
        return False
    #判断字符串是否包含特殊字符（如：@ #）
    def contain_specialChar(self,check_str):
        if "@" in check_str or "#" in check_str or "https" in check_str:
            return True
        return False



    #判读字符串是否包含乱码（即不包含英文、数字和标点），TRUE是包含乱码
    def contain_other(self,check_str):
        if (not self.contain_english(check_str)) and (not self.contain_number(check_str)) and not self.isPuntion(check_str):
            return True
        return False

    def isPuntion(self,check_str):
        if check_str =="," or check_str =="." or check_str =="," or check_str =="?" or check_str =="!" or check_str ==":" or check_str ==";" :
            return True
        return False


    def is_chinese(self,char):
        if u'\u4e00' <= char <= u'\u9fff':
            return True
        else:
            return False

    def is_english(self,char):
        if (u'\u0041' <= char <=u'\u005a') or (u'\u0061' <= char <=u'\u007a'):
            return True
        else:
            return False

    def is_number(self,char):
        if u'\u0030' <= char <=u'\u0039':
            return True
        else:
            return False



    def _read_excel(cls, input_file, col_number=None, sheet_name=None):
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







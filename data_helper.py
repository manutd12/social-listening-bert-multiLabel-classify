# -*-coding:utf-8 -*-

import csv


def load_label(file_path):
    """
        Load sub_intent and corresponding intent&domain code.
    """
    sub_intent_to_domain = {}
    sub_intent_to_intent = {}
    sub_intent_to_cn = {}

    # Read label
    with open(file_path, "r", encoding="utf-8") as q_csv:
        reader = csv.reader(q_csv)
        for line in reader:
            if reader.line_num == 1:
                continue
            sub_intent = line[5].strip().lower()
            sub_intent_to_domain[sub_intent] = line[3].strip().lower()
            sub_intent_to_intent[sub_intent] = line[4].strip().lower()
            sub_intent_to_cn[sub_intent] = line[2]

    return sub_intent_to_domain, sub_intent_to_intent, sub_intent_to_cn


def load_eval_data(file_path):
    """
        Load evaluation data set.
    """
    sentence = []
    sub_intent = []
    with open(file_path, "r", encoding="utf-8") as q_csv:
        reader = csv.reader(q_csv)
        for line in reader:
            if reader.line_num == 1:
                continue
            sentence.append(line[0])
            sub_intent.append(line[1])

    return sentence, sub_intent


class DataProcess(object):

    def process_data(self,sentence):
        """
            对一条预测数据进行处理：去除特殊字符，去除中文，去除数字，去除标点，去除乱码错别字等
            注意：预测时不去除停用词
        """
        self.remove(sentence)
        content = sentence.replace("[", "").replace(".]", "").replace("]", "").replace("<ob>", "").replace("</ob>", "").replace("<ng>", "").replace("</ng>", "").replace("<po>", "").replace("</po>", "").replace("<nu>", "").replace("</nu>", "")
        content = content.replace("=","").replace("&", "").replace('"', "").replace("-", "").replace("{", "").replace("}", "")
        content = content.replace("xxx", "").replace("<", "").replace(">", "").replace("//", "").replace("(", "")
        content = content.replace(")", "").replace("+", "").replace("^", "").replace("|", "").replace("...", "")
        content = content.replace("......", "").replace("XXXXXX", "").replace("XXX", "").replace(".", "").replace(":", "").replace(",", "").replace(";", "").replace("!", "").replace("?", "")
        content_list = content.lower().strip().split()
        content_list = [content_.strip() for content_ in content_list if len(content_) <= 15 and (not self.contain_other(content_.strip())) and not self.contain_chinese(content_.strip()) and not self.contain_number(content_.strip())  and not self.contain_specialChar(content_.strip())]
        return " ".join(content_list)


    def remove(self,text):
        import re
        remove_chars = '[0-9’!"$%&\'()*+,-./:;<=>?，。?★#、…【】《》？“”‘’！[\\]^_`{|}~]+'
        return re.sub(remove_chars, '', text)

    # 判读字符串是否包含中文，TRUE是包含中文
    def contain_chinese(self,check_str):
        for char in check_str: #判断每个字符
            if self.is_chinese(char):
                return True
        return False

    # 判断字符串是否只有中文，True是只有中文
    def only_contain_chinese(self,check_str):
        for char in check_str:
            if not self.is_chinese(char):
                return False
        return True

    # 判读字符串是否包含数字，TRUE是包含数字
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

    # 判读字符串是否包含英文，TRUE是包含英文
    def contain_english(self,check_str):
        for char in check_str:
            if self.is_english(char):
                return True
        return False

    # 判断字符串是否包含特殊字符（如：@ #）
    def contain_specialChar(self,check_str):
        if "@" in check_str or "https" in check_str:
            return True
        return False

    # 判读字符串是否包含乱码（即不包含英文、数字），TRUE是包含乱码
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


# -*-coding:utf-8 -*-

import logging
import pickle
from multiLabel_bert_model import MultiLabelBert
from data_helper import DataProcess


class Classifier:
    def __init__(self, model_path, lable_path, thresholds_path, label_lower2original_path):
        """
        Initialization of classifier.
        """
        # Load data
        logging.info("Load label...")

        with open(label_lower2original_path, "rb") as f_label_lower2original:
            self.label_lower2original = pickle.load(f_label_lower2original)  # 小写label对应原始label
        with open(thresholds_path, "rb") as f_thresholds:
            self.thresholds, self.label2threshold = pickle.load(f_thresholds)
        with open(lable_path,"rb") as f_label_map:
            self.label2id, self.id2label = pickle.load(f_label_map)

        # Model initialization
        logging.info("Init BERT model...")
        self.multiLabelBert = MultiLabelBert(model_path, self.label2id)

    def predict(self, sentence):
        """
        Predict sub_intent of a question.
        """
        process = DataProcess()
        sentence_process = process.process_data(sentence)
        prob = self.multiLabelBert.run(sentence_process)
        result = [(self.label_lower2original[self.id2label[i]], prob_) for i, prob_ in enumerate(prob) if prob_ >= self.label2threshold[self.id2label[i]]]

        return result

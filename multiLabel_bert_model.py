#-*- coding : utf-8 -*-
# coding: utf-8
import numpy as np
import os
import tensorflow as tf
import bert.modeling as modeling
import bert.tokenization as tokenization
import bert.run_classifier_train_multiLabel as rc





class MultiLabelBert:
    BERT_CONFIG_FILE = "uncased_L-12_H-768_A-12/bert_config.json"
    VOCAB_FILE = "uncased_L-12_H-768_A-12/vocab.txt"

    do_lower_case = True
    max_seq_length = 256   # 可以修改成句子长度
    batch_size = 1
    is_training = False
    use_one_hot_embeddings = False

    def __init__(self, model_path, label2id):
        """
        Creates graphs, sessions and restore models.
        """
        config_file = os.path.join(model_path, self.__class__.BERT_CONFIG_FILE)
        vocab_file = os.path.join(model_path, self.__class__.VOCAB_FILE)

        bert_config = modeling.BertConfig.from_json_file(config_file)
        self.tokenizer = tokenization.FullTokenizer(vocab_file, self.__class__.do_lower_case)

        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True

        # modify
        self.label2id = label2id
        self.graph = tf.Graph()

        with self.graph.as_default() as g:
            self.input_ids_p = tf.placeholder(tf.int32, [self.__class__.batch_size,
                                                         self.__class__.max_seq_length], name="input_ids")
            self.input_mask_p = tf.placeholder(tf.int32, [self.__class__.batch_size,
                                                          self.__class__.max_seq_length], name="input_mask")
            self.label_ids_p = tf.placeholder(tf.int32, [self.__class__.batch_size,
                                                         len(self.label2id)], name="label_ids")  # mock
            self.segment_ids_p = tf.placeholder(tf.int32, [self.__class__.max_seq_length], name="segment_ids")

            _, _, _, self.probabilities = rc.create_model(bert_config, self.__class__.is_training,
                                                          self.input_ids_p, self.input_mask_p, self.segment_ids_p,
                                                          self.label_ids_p, len(self.label2id),
                                                          self.__class__.use_one_hot_embeddings)
            saver = tf.train.Saver()
            graph_init_op = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.graph, config=gpu_config)
        self.sess.run(graph_init_op)

        with self.sess.as_default() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))

    @staticmethod
    def convert_line(line, label2id, max_seq_length, tokenizer):
        """
        Function to convert a line that should be predicted into BERT input features.
        """
        label = tokenization.convert_to_unicode("email")  # Mock label
        text_a = tokenization.convert_to_unicode(line)
        example = rc.InputExample(guid=0, text_a=text_a, text_b=None, label=label)
        feature = rc.convert_single_example(example, max_seq_length, tokenizer, label2id)

        input_ids = np.reshape([feature.input_ids], (1, max_seq_length))
        input_mask = np.reshape([feature.input_mask], (1, max_seq_length))
        segment_ids = np.reshape([feature.segment_ids], max_seq_length)
        label_ids = [feature.label_id]

        return input_ids, input_mask, segment_ids, label_ids

    def run(self, line):
        """
        Function to run the inference
        """
        input_ids, input_mask, segment_ids, label_ids = self.__class__.convert_line(line.lower(), self.label2id,
                                                                                    self.__class__.max_seq_length,
                                                                                    self.tokenizer)
        with self.graph.as_default() as g:
            with self.sess.graph.as_default():
                feed_dict = {self.input_ids_p: input_ids, self.input_mask_p: input_mask,
                             self.segment_ids_p: segment_ids, self.label_ids_p: label_ids}
                prob = self.sess.run([self.probabilities], feed_dict)
                prob = list(prob[0][0])
                # scores = softmax(prob)
                # if top_k > 1:
                #     labels_ids = list(sorted(enumerate(softmax(prob)), key=lambda s: s[1], reverse=True))
                #     labels = [(label_id, self.label_list[label_id], score) for label_id, score in labels_ids][:top_k]
                #     return labels
                # else:
                #     label_predict_id = np.argmax(prob)
                #     label_predict = self.label_list[label_predict_id]
                #     top_score = np.amax(scores)
                return prob

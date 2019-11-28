# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""multi-label-BERT finetuning runner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import optimization, tokenization, modeling
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle
import os
import xlrd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score,precision_score, recall_score
from data_util import data
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
root_path = os.path.abspath(os.path.dirname(os.getcwd()))+os.sep
root_path = os.getcwd() + os.sep
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", root_path+"data",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", os.path.join(root_path+"model/multilingual_L-12_H-768_A-12",'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'social_listening', "The name of the task to train.")

flags.DEFINE_string("vocab_file", os.path.join(root_path+"model/multilingual_L-12_H-768_A-12","vocab.txt"),
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", root_path+"my_output",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", os.path.join(root_path+"model/multilingual_L-12_H-768_A-12/bert_model.ckpt"),
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", None,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", None, "Whether to run training.")

flags.DEFINE_bool("do_eval", None, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool("do_calThresholds", None, "Whether to calculate thresholds of every class.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", None,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 500,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float("cut_off", 0.5, "Cut off in the probabilities, in case output the classes")

flags.DEFINE_string("cutoff_type", "dynamic", "Dynamic calculate the  cutoff")

flags.DEFINE_bool("isMultiLabel", True, "is multi-label task")

flags.DEFINE_string("stopWord_file", root_path+"user_dict//social-listening-stopwords.txt", "stopwords_en")

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


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, original_text, text_b=None, label=None):
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
        self.original_text = original_text


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
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


class social_listening_Prpcessor(DataProcessor):
    def __init__(self):
        self.language = "zh"
        self.dataProcess = data()
        self.trainFilePath = os.path.join(FLAGS.data_dir, "trainData.xls")
        self.devFilePath = os.path.join(FLAGS.data_dir, "devData.xls")
        self.testFilePath = os.path.join(FLAGS.data_dir, "testData.xls")
        self.cut_label = ['others']  #todo: 要去除的类别
    def write_eval_data(self, example):
        # Write result
        with open(root_path+"/metrics/eval/eval_split.csv", "w", encoding='utf-8-sig', newline="") as e_csv:
            writer = csv.writer(e_csv)
            head = ["sentence", "intent"]
            writer.writerow(head)

            for e in example:
                writer.writerow([e.text_a, e.label])

    def get_train_examples(self, data_dir):
        """See base class."""
        original_docs = []
        train_file = os.path.join(FLAGS.data_dir,"trainData.pkl")
        if os.path.exists(train_file):
            with open(train_file,"rb") as f:
                result = pickle.load(f)
                data_text = result[0]
                data_label = result[1]
        else:
            data_text, data_label, original_docs = self.dataProcess.get_data(self.trainFilePath,self.label_list,FLAGS.stopWord_file,mode='train')

            #去除低频字
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

            pickle.dump((data_text,data_label),open(train_file,"wb"))

        X = []
        Y = []
        for (text, label) in zip(data_text,data_label):
            text_a = " ".join(text)
            text_a = tokenization.convert_to_unicode(text_a)
            label = ",".join(label)
            label = tokenization.convert_to_unicode(label)
            X.append(text_a)
            Y.append(label)

        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        X_train = X
        y_train = Y
        train_example = []
        for i, text_a in enumerate(X_train):
            guid = "train-%d" % i
            label = y_train[i]
            train_example.append(
                InputExample(guid=guid, text_a=text_a, label=label, original_text=original_docs[i]))
        print('________train data len:___', len(train_example))

        # if len(X_test) > 0:
        #     test_example = []
        #     for i, text_a in enumerate(X_test):
        #         guid = "test-%d" % i
        #         label = y_test[i]
        #         test_example.append(
        #             InputExample(guid=guid, text_a=text_a, label=label))
        #     print('________eval data len:', len(test_example))
        #
        #     f = open(os.path.join(data_dir, 'eservice_dataset_eval.pkl'), 'wb')
        #     pickle.dump(test_example, f)
        #     f.close()
        #     self.write_eval_data(test_example)

        return train_example

    def get_dev_examples(self, data_dir):
        """See base class."""
        data_text, data_label ,original_docs = self.dataProcess.get_data(self.devFilePath, self.label_list, FLAGS.stopWord_file,mode='predict')
        X = []
        Y = []
        for (text, label) in zip(data_text,data_label):
            text_a = " ".join(text)
            text_a = tokenization.convert_to_unicode(text_a)
            label = ",".join(label)
            label = tokenization.convert_to_unicode(label)
            X.append(text_a)
            Y.append(label)

        X_dev = X
        y_dev = Y
        dev_example = []
        for i, text_a in enumerate(X_dev):
            guid = "dev-%d" % i
            label = y_dev[i]
            dev_example.append(
                InputExample(guid=guid, text_a=text_a, label=label, original_text=original_docs[i]))
        print('________dev data len:___', len(dev_example))
        return dev_example

    def get_eval_examples(self, data_dir):
        data_text,data_label,original_docs = self.dataProcess.get_data(self.testFilePath, self.label_list, FLAGS.stopWord_file,mode='predict')
        X = []
        Y = []
        for (text, label) in zip(data_text,data_label):
            text_a = " ".join(text)
            text_a = tokenization.convert_to_unicode(text_a)
            label = ",".join(label)
            label = tokenization.convert_to_unicode(label)
            X.append(text_a)
            Y.append(label)

        X_dev = X
        y_dev = Y
        test_example = []
        for i, text_a in enumerate(X_dev):
            guid = "dev-%d" % i
            label = y_dev[i]
            test_example.append(
                InputExample(guid=guid, text_a=text_a, label=label, original_text=original_docs[i]))
        print('________test data len:___', len(test_example))
        return test_example


    def get_labels(self, data_dir):

        self.label_list, lower2originalLabel = self.dataProcess.get_label(os.path.join(FLAGS.data_dir, "Product_10000.xlsx"),self.cut_label)
        label2id = {}
        id2label = {}
        for (i, label) in enumerate(self.label_list):
            label2id[label] = i
            id2label[i] = label
        pickle.dump((label2id,id2label),open(os.path.join(FLAGS.data_dir,"label_map.pkl"),'wb'))
        pickle.dump(lower2originalLabel,open(os.path.join(FLAGS.data_dir,"lower2originalLabel.pkl"),'wb'))
        return self.label_list, lower2originalLabel, label2id, id2label



def convert_single_example(example, max_seq_length,
                           tokenizer, label2id):
    """Converts a single `InputExample` into a single `InputFeatures`.
    组装一条样本的输入数据格式，包括：iunputid,inputmask,segmentid,labelid
    example:输入的一条样本
    label2id:标签对应的id"""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    #modify by songming
    label_id = None
    # multi_label: multi-hot
    label_id_list = [label2id[label_] for label_ in example.label.split(",")]
    label_id=[0 for l in range(len(label2id))]
    for index in (label_id_list):
        label_id[index] = 1

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file, label2id):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)
    #读取每一条样本
    for (ex_index, example) in enumerate(examples):
        feature = convert_single_example(example, max_seq_length, tokenizer, label2id)

        def create_int_feature(values):

            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        #modify by songming
        if isinstance(feature.label_id,list):
            label_ids = feature.label_id
        else:
            label_ids = [feature.label_id]
        #modify end
        features["label_ids"] = create_int_feature(label_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,num_labels,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([num_labels], tf.int64),#modify by songming: ADD TO A FIXED length:num_labels
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.7)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        # multi-label：
        probabilities = tf.nn.sigmoid(logits)
        labels = tf.cast(labels,tf.float32)
        tf.reduce_sum(labels * probabilities, axis=-1)
        #         per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits)

        # label smooth
        per_example_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels,logits=logits,label_smoothing=0.1)

        loss = tf.reduce_mean(per_example_loss)

        #single-label：
        # probabilities = tf.nn.softmax(logits,axis=-1)
        # one_hot_labels = tf.one_hot(labels,depth=num_labels,dtype=tf.float32)
        # per_example_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels = one_hot_labels)
        # loss = tf.reduce_mean(per_example_loss)
        #
        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            #modify 训练过程中打印loss
            logging_hook = tf.train.LoggingTensorHook({"loss": total_loss}, every_n_iter=10)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):


                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)


                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,

                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                predictions={"Prediction": tf.argmax(logits, axis=-1, output_type=tf.int32)},
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities,"label_ids":label_ids},
                scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn





def main(_):

    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        'social_listening':social_listening_Prpcessor,

    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list, lower2originalLabel, label2id, id2label = processor.get_labels(FLAGS.data_dir)


    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None


    #     threshold_examples = random.choices(train_examples, k=int(len(train_examples)*0.8))
    #     threshold_examples = random.sample(train_examples, k=int(len(train_examples)*0.8))
    #     random.shuffle(train_examples)
    #     threshold_examples = train_examples[:int(len(train_examples)*0.2)]
    #     train_examples = train_examples[int(len(train_examples)*0.2):]
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    # 进行训练
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        # 将数据写入tf.record
        file_based_convert_examples_to_features(
            train_examples, FLAGS.max_seq_length, tokenizer, train_file, label2id)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        # 读取一个batch数据
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            num_labels = len(label_list),
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    # 利用开发集计算阈值
    if FLAGS.do_calThresholds:
        dev_examples = processor.get_dev_examples(FLAGS.data_dir)
        thresholds_data_path = os.path.join(FLAGS.output_dir, "threshold.tf_record")
        file_based_convert_examples_to_features(
            dev_examples, FLAGS.max_seq_length, tokenizer, thresholds_data_path,label2id)
        thresholds_pre_result = []

        if os.path.exists(os.path.join(FLAGS.output_dir,"thresholds_data_pre_result.pkl")):
            with open(os.path.join(FLAGS.output_dir,"thresholds_data_pre_result.pkl"),"rb") as f_thresholds_data_pre_result:
                thresholds_pre_result = pickle.load(f_thresholds_data_pre_result)
            print("成功加载开发集预测结果")
        else:
            print("计算开发集的预测结果")
            train_sample_input_fn = file_based_input_fn_builder(
                input_file=thresholds_data_path,
                num_labels=len(label_list),
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder= True if FLAGS.use_tpu else False)
            pre_results = estimator.predict(input_fn=train_sample_input_fn)
            thresholds_pre_result=[predictions for i, predictions in enumerate(pre_results)]
            pickle.dump(thresholds_pre_result, open(FLAGS.output_dir+"//thresholds_data_pre_result.pkl", "wb"))

        # 计算每个类别的最佳阈值
        label2threshold = dict()
        print("start calculate best thresholds of every class")
        y_true = []
        y_pred = []
        for _, prediction in enumerate(thresholds_pre_result):
            y_true.append(prediction["label_ids"])  # 二值化后的label
            y_pred.append(prediction["probabilities"])
        thresholds = []
        for i in range(len(label_list)):
            best_threshold = 0
            best_threshold_f1 = 0
            score = 0
            y_true_class = np.array(y_true)[:, i]
            prediction_probabilies = np.array(y_pred)[:, i]
            for t in np.arange(0.5, 1, 0.01):  # todo:从0.5开始微调概率阈值
                y_pred_class = [1 if x>= t else 0 for x in prediction_probabilies]
                if len([i for i in y_pred_class if i == 0]) == 0 or len([i for i in y_pred_class if i == 1]) == 0:
                    continue
                score = f1_score(y_true_class, y_pred_class)
                if score > best_threshold_f1:
                    best_threshold_f1 = score
                    best_threshold = t
            if best_threshold == 0:
                thresholds.append(0.5)
                label2threshold[id2label[i]] = 0.5
            else:
                thresholds.append(best_threshold)
                label2threshold[id2label[i]] = best_threshold
        # todo: label2threshold：每个类别对应的最佳阈值。thresholds：所有类别的最佳阈值的列表
        pickle.dump((thresholds,label2threshold),open(FLAGS.output_dir + "//thresholds.pkl", "wb"))
        for key,value in label2threshold.items():
            print(str(key)+":"+str(value))

        # 计算阈值在开发集上的PRF
        pred_samples = []
        for (i, probabilities) in enumerate(y_pred):
            doc_pred = [1 if prob >= label2threshold[id2label[index]] else 0 for index, prob in enumerate(probabilities)]
            pred_samples.append(doc_pred)   # 将每条样本的每个类别预测值设为1/0
        # samples
        samples = [0 for i in range(len(y_true[0]))]
        for i in range(len(y_true[0])):
            samples[i] = len([j for j in range(len(y_true)) if y_true[j][i] == 1])  # 在每个类别上真实值为1的数目，用于计算每个类别的计算权重
        f1_mean = []
        pre_mean = []
        recall_mean = []
        output_threshold_file = os.path.join(FLAGS.output_dir, "thresholds_results.txt")
        with tf.gfile.GFile(output_threshold_file, "w") as writer:
            tf.logging.info("***** threshold cal results *****")
            for i in range(len(y_true[0])):
                y_true_label = np.array([y_true[j][i] for j in range(len(y_true))])  # 第i个类别上的所有样本真实值0/1
                y_pred_label = np.array([pred_samples[j][i] for j in range(len(pred_samples))])  # 第i个类别上的所有样本预测值0/1
                score_f1 = f1_score(y_true_label, y_pred_label)
                score_precision = precision_score(y_true_label, y_pred_label)
                score_recall = recall_score(y_true_label, y_pred_label)
                f1_mean.append(score_f1)
                pre_mean.append(score_precision)
                recall_mean.append(score_recall)
                writer.write(str(label_list[i]) + "\t" + "f1:" + str(score_f1) + "\t" + str(samples[i])+"\n")
                writer.write(str(label_list[i]) + "\t" + "precision:" + str(score_precision) + "\t" + str(samples[i])+"\n")
                writer.write(str(label_list[i]) + "\t" + "recall:" + str(score_recall) + "\t" + str(samples[i])+"\n")
            f1_macro = np.average(f1_mean, weights=[samples[i] / float(sum(samples)) for i in range(len(samples))])
            precison_macro = np.average(pre_mean, weights=[samples[i] / float(sum(samples)) for i in range(len(samples))])
            recall_macro = np.average(recall_mean, weights=[samples[i] / float(sum(samples)) for i in range(len(samples))])
            print("F1_threshold:"+str(f1_macro))
            print("precison_threshold:"+str(precison_macro))
            print("recall_threshold:"+str(recall_macro))
            writer.write("num_threshold_data: \t" + str(len(dev_examples)) + "\n")
            writer.write("F1_threshold: \t" + str(f1_macro) + "\n")
            writer.write("precison_threshold: \t" + str(precison_macro) + "\n")
            writer.write("recall_threshold: \t" + str(recall_macro) + "\n")
            for key,value in label2threshold.items():
                writer.write(str(key)+": \t"+str(value)+ "\n")
        data_process = data()
        data_process.create_all_result_excel(os.path.join(FLAGS.output_dir,"threshold_result.xls"), dev_examples, y_pred, pred_samples, id2label)
        data_process.create_error_result_excel(os.path.join(FLAGS.output_dir, "threshold_result.xls"),os.path.join(FLAGS.output_dir, "threshold_error_result.xls"))



    # 利用测试集进行测试
    if FLAGS.do_eval:
        eval_examples = processor.get_eval_examples(FLAGS.data_dir)
        thresholds = None
        label2threshold = None
        with open(FLAGS.output_dir + "//thresholds.pkl","rb") as f_thresholds: # todo：加载类别阈值
            thresholds,label2threshold = pickle.load(f_thresholds)
        if not os.path.exists(os.path.join(FLAGS.output_dir,"eval_results.pkl")):

            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            file_based_convert_examples_to_features(
                eval_examples, FLAGS.max_seq_length, tokenizer, eval_file, label2id)

            eval_drop_remainder = True if FLAGS.use_tpu else False
            eval_input_fn = file_based_input_fn_builder(
                input_file=eval_file,
                num_labels=len(label_list),
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=eval_drop_remainder)

            result = estimator.predict(input_fn=eval_input_fn)
            y_true = []
            y_pred = []
            for (i, prediction) in enumerate(result):
                y_true.append(prediction["label_ids"])  # 二值化后的label
                y_pred.append(prediction["probabilities"])
            pickle.dump((y_true,y_pred),open(os.path.join(FLAGS.output_dir,"eval_results.pkl"),"wb"))
        else:
            y_true, y_pred = pickle.load(open(os.path.join(FLAGS.output_dir,"eval_results.pkl"),"rb"))

        # 每个label根据各自的概率阈值进行计算
        if FLAGS.cutoff_type == "dynamic":
            # binary calculation
            pred_samples = []
            for (i, probabilities) in enumerate(y_pred):
                doc_pred = [1 if x >= y else 0 for x, y in zip(probabilities,thresholds)]
                pred_samples.append(doc_pred)  # 将每条样本的每个类别预测值设为1/0

        # 所有label根据一个固定的概率阈值进行计算
        elif FLAGS.cutoff_type == "static":
            print("static cal")
            # binary calculation
            pred_samples = []
            for (i, prediction) in enumerate(y_pred):
                probabilities = prediction
                doc_pred = [1 if x >= FLAGS.cut_off else 0 for x in probabilities]
                pred_samples.append(doc_pred)

        # 根据每个类别的实际数据量，用于计算每个类别的权重
        samples = [0 for i in range(len(y_true[0]))]
        for i in range(len(y_true[0])):
            samples[i] = len([j for j in range(len(y_true)) if y_true[j][i] == 1])

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt") # todo:测试记录的结果（加权宏平均prf和微平均prf）
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")

            # # auc
            # writer.write("AUC: \n")
            # auc_mean = []
            # for i in range(len(y_true[0])):
            #     y_true_label = np.array([y_true[j][i] for j in range(len(y_true))])
            #     y_pred_label = np.array([y_pred[j][i] for j in range(len(y_pred))])
            #     try:
            #         score = roc_auc_score(y_true_label, y_pred_label)
            #         writer.write(str(label_list[i]) + "\t" + str(score) + "\n")
            #     except:
            #         print(y_true_label, y_pred_label)
            #         writer.write(str(label_list[i]) + "\t0.5\n")
            #     auc_mean.append(score)
            # writer.write("Mean AUC: \t" + str(np.mean(auc_mean)) + "\n")
            # writer.write("--------\n")
            #
            # # hamming loss
            # writer.write("\nHamming Loss: \t" + str(hamming_loss(np.array(y_true), np.array(pred_samples))) + "\n")

            # macro-PRF：计算出每个类别上的PRF，再进行加权平均
            writer.write("--------\n")
            writer.write("F1 score: \n")
            f1_mean = []
            pre_mean = []
            recall_mean = []
            for i in range(len(y_true[0])):
                y_true_label = np.array([y_true[j][i] for j in range(len(y_true))])#第i个类别上的所有样本真实值0/1
                y_pred_label = np.array([pred_samples[j][i] for j in range(len(pred_samples))]) #第i个类别上的所有样本预测值0/1
                score_f1 = f1_score(y_true_label, y_pred_label)
                score_precision = precision_score(y_true_label, y_pred_label)
                score_recall = recall_score(y_true_label, y_pred_label)
                f1_mean.append(score_f1)
                pre_mean.append(score_precision)
                recall_mean.append(score_recall)
                writer.write(str(label_list[i]) + "\t" + "f1:" + str(score_f1) + "\t" + str(samples[i])+"\n")
                writer.write(str(label_list[i]) + "\t" + "precision:" + str(score_precision) + "\t" + str(samples[i])+"\n")
                writer.write(str(label_list[i]) + "\t" + "recall:" + str(score_recall) + "\t" + str(samples[i])+"\n")

            # macro-weighted-average 加权宏平均
            f1_macro = np.average(f1_mean, weights=[samples[i] / float(sum(samples)) for i in range(len(samples))])
            precison_macro = np.average(pre_mean, weights=[samples[i] / float(sum(samples)) for i in range(len(samples))])
            recall_macro = np.average(recall_mean, weights=[samples[i] / float(sum(samples)) for i in range(len(samples))])
            writer.write("weighted-Macro F1: \t" + str(f1_macro) + "\n")
            writer.write("weighted-Macro Precision: \t" + str(precison_macro) + "\n")
            writer.write("weighted-Macro recall: \t" + str(recall_macro) + "\n")
            writer.write("--------\n")
            print("weighted-Macro Precision:" + str(precison_macro))

        # micro-PRF
        data_process = data()
        tp = 0
        ts = 0
        fs = 0
        for y_pre,y_true in zip(pred_samples,y_true):
            for pre_,true_ in zip(y_pre,y_true):
                if pre_ == 1:
                    ts+=1
                if true_ ==1:
                    fs +=1
                if pre_ ==1 and true_ ==1:
                    tp +=1
        pre_micro = tp/ts
        recall_micro = tp/fs
        print("micro precision:" + str(pre_micro))
        print("micro recall:" + str(recall_micro))
        writer.write("micro precision: \t" + str(pre_micro) + "\n")
        writer.write("micro recall: \t" + str(recall_micro) + "\n")
        writer.close()

        # 将所有预测结果和预测错误结果分别保存excel
        all_result_file_name = "all_evaluate_result.xls"
        error_result_file_name = "error_result.xls"
        data_process.create_all_result_excel(os.path.join(FLAGS.output_dir, all_result_file_name), eval_examples, y_pred, pred_samples, id2label)
        data_process.create_error_result_excel(os.path.join(FLAGS.output_dir, all_result_file_name), os.path.join(FLAGS.output_dir, error_result_file_name))



if __name__ == "__main__":
    tf.app.run()

if __name__ == "__main__":
    # flags.mark_flag_as_required("data_dir")
    # flags.mark_flag_as_required("task_name")
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()

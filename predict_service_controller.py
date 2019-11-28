# -*-coding:utf-8 -*-

import logging
import configparser
from flask import Flask, request, jsonify

from classification_handler import Classifier

# Logging config
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(filename)s[line:%(lineno)d][%(levelname)s] %(message)s",
                    datefmt="%a, %d %b %Y %H:%M:%S",
                    filename="log/classification.log",
                    filemode="a")

# Configuration
conf = configparser.ConfigParser()
conf.read("config.ini")

# Constants from configuration
LABEL_FILE = conf.get("path", "multiLabel_label")  # label列表和label对应的id
MODEL_FILE = conf.get("path", "multiLabel_model")  # 模型文件
Thresholds_FILE = conf.get("path", "multiLabel_thresholds")  # 训练好的类别对应的阈值
label_lower2original = conf.get("path", "multiLabel_label_lower2original")  # 小写label 对应 大写label
APP_PORT = conf.getint("app", "port")
APP_DEBUG = conf.getboolean("app", "debug")

# Flask app
app = Flask("multi_label_classification")

# Classifier
classifier = Classifier(MODEL_FILE, LABEL_FILE, Thresholds_FILE, label_lower2original)


@app.route("/subintent_classification", methods=['POST'])
def get_prediction():
    """
    Handle POST request and return classification result.
    """
    result = {}
    result_list = []

    try:
        logging.info(str(request.data))
        content = request.get_json()

        sentence = content["q"]

        result = classifier.predict(sentence)  # 预测sentence的label
        print(len(result))
                # result['subintentcode'] = subintent_code
                # result['score'] = '%.2f' % score
                # result['domain'] = domain_code
                # result['intentcode'] = intent_code
                # result_list.append(result)


        # logging.info("Result:" + str(result_list))
        return jsonify(result_list)

    except Exception as e:
        logging.error(e)
        return jsonify([])


if __name__ == "__main__":
    app.run(
        # host="0.0.0.0",
        host="127.0.0.1",
        port=APP_PORT,
        debug=APP_DEBUG
    )

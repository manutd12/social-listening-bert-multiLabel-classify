# social-listening-bert-multiLabel-classify
social数据 多标签分类
使用的是bert-fine-tune模型
数据分为训练集、开发集（dev)、测试集（eval）
开发集用于计算label的概率阈值
评测指标采用 加权micro-prf 和 macro-prf

项目整个流程：
1、数据增强：反向翻译、EDA
2、数据预处理：去除停用词、去除特殊字符、去除中文、去除错别字（字频为1的字）、去除标点、去除数字、去除乱码等
3、训练
4、开发集上计算概率阈值
5、测试
6、预测服务

# *_*coding:utf-8 *_*
import pickle

class config(object):
    # 数据集参数
    train_list = "./data/train_text_file.txt"
    test_list = "./data/train_text_file.txt"
    num_workers = 4
    batchsize = 32
    imgH = 32
    imgW = 100
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    num_classes = len(alphabet) + 1
    manualseed = 0

    # 训练参数
    num_epochs = 100
    nc = 1
    nh = 256
    lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    output = "output"
    pretrained = ""
    interval = 10
    valinterval = 30





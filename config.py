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
    random_sample = True
    h_rate = 0.1
    n_test_disp = 10

    # demo
    img_path = "./images/20456343_4045240981.jpg"
    encoder = "./output/pretrained/encoder_9.pth"
    decoder = "./output/pretrained/decoder_9.pth"
    cuda = True


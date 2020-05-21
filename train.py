# *_*coding:utf-8 *_*
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
import random
import src.utils as utils
import src.dataset as dataset

from config import config as opt
import models.crnn as crnn



#
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def val(crnn, data_loader, criterion, converter, device, max_iter=100):
    print('Start val')

    crnn.eval()
    val_iter = iter(data_loader)

    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for _ in range(max_iter):
        data = val_iter.next()
        cpu_images, text, length, cpu_texts = data
        image = cpu_images.to(device)
        batch_size = cpu_images.size(0)

        with torch.no_grad():
            preds = crnn(image)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length)
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchsize)
    print('Test loss: %f, accuray: %.2f%%' % (loss_avg.val(), accuracy * 100))


def trainBatch(crnn, train_iter, criterion, optimizer, device):
    crnn.train()
    data = train_iter.next()
    image, text, length, _ = data
    image = image.to(device)
    image.requires_grad_()
    batch_size = image.size(0)
    preds = crnn(image)
    preds = torch.clamp(preds, min=-50.0)
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)
    cost = criterion(preds.log_softmax(2), text, preds_size, length)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

def main():

    if not os.path.exists(opt.output):
        os.makedirs(opt.output)

    converter = utils.strLabelConverter(opt.alphabet)

    collate = dataset.AlignCollate()
    train_dataset = dataset.TextLineDataset(text_file=opt.train_list, transform=dataset.ResizeNormalize(100, 32), converter=converter)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True,
                                               num_workers=opt.num_workers, collate_fn=collate)
    test_dataset = dataset.TextLineDataset(text_file=opt.train_list, transform=dataset.ResizeNormalize(100, 32), converter=converter)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=opt.batchsize,
                                              num_workers=opt.num_workers, collate_fn=collate)

    criterion = nn.CTCLoss()

    import models.crnn as crnn

    crnn = crnn.CRNN(opt.imgH, opt.nc, opt.num_classes, opt.nh)
    crnn.apply(utils.weights_init)
    if opt.pretrained != '':
        print('loading pretrained model from %s' % opt.pretrained)
        crnn.load_state_dict(torch.load(opt.pretrained), strict=False)
    print(crnn)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    crnn = crnn.to(device)
    criterion = criterion.to(device)


    # setup optimizer
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr)

    for epoch in range(opt.num_epochs):

        loss_avg = 0.0
        i = 0
        while i < len(train_loader):

            time0 = time.time()
            # 训练
            train_iter = iter(train_loader)

            cost = trainBatch(crnn, train_iter, criterion, optimizer, device) # 一个批次，一个批次训练
            loss_avg += cost
            i += 1

            if i % opt.interval == 0:
                print('[%d/%d][%d/%d] Loss: %f Time: %f s' %
                      (epoch, opt.num_epochs, i, len(train_loader), loss_avg,
                       time.time() - time0))
                loss_avg = 0.0



        if (epoch + 1) % opt.valinterval == 0:
            val(crnn, test_loader, criterion, converter=converter, device=device, max_iter=100)
            # torch.save(crnn.state_dict(), '{0}/crnn-{1}.pth'.format(opt.output, epoch))

if __name__ == '__main__':
    main()


from data_pipe import get_train_loader,  get_test_loader
import torch
import torch.nn.functional as F
from torch import optim
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import AverageMeter, make_folder_if_not_exist
import os
import pprint
from resnet import resnet18


os.environ['CUDA_VISIBLE_DEVICES']='7'

class face_learner(object):
    def __init__(self, conf, inference=False):
        pprint.pprint(conf)

        self.model = resnet18(conf.model.use_senet, conf.model.embedding_size, conf.model.drop_out, conf.model.se_reduction)
        self.model = torch.nn.DataParallel(self.model).cuda()

        if not inference:
            self.loader = get_train_loader(conf)
            self.optimizer = optim.SGD(list(self.model.parameters()), lr=conf.train.lr, momentum=conf.train.momentum)
            print(self.optimizer)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, conf.train.milestones, gamma=conf.train.gamma)
            print('optimizers generated')
            self.print_freq = len(self.loader)//2
            print('print_freq: %d'%self.print_freq)
        else:
            self.test_loader = get_test_loader(conf)

    def save_state(self, save_path,  epoch):
        torch.save(self.model.state_dict(), save_path+'//'+'epoch={}.pth'.format(str(epoch)))

    def get_model_input_data(self, imgs):
        return torch.cat((imgs[0], imgs[1], imgs[2]), dim=1).cuda()  # for rgb+depth+ir

    def get_model_input_data_for_test(imgs):
        input0 = torch.cat((imgs[0], imgs[1], imgs[2]), dim=1).cuda()
        input1 = torch.cat((imgs[3], imgs[4], imgs[5]), dim=1).cuda()
        return input0, input1  # for rgb+depth+ir

    def train(self, conf):
        self.model.train()
        xent1_losses = AverageMeter()
        xent2_losses = AverageMeter()
        losses = AverageMeter()

        save_path = os.path.join(conf.save_path, conf.exp)
        make_folder_if_not_exist(save_path)

        for e in range(conf.train.epoches):
            self.scheduler.step()
            print('exp {}'.format(conf.exp))
            print('epoch {} started'.format(e))
            print('learning rate: {}, {}'.format(len(self.scheduler.get_lr()), self.scheduler.get_lr()[0]))
            for batch_idx, (imgs, labels) in enumerate(self.loader):
                input = self.get_model_input_data(imgs)
                labels = labels.cuda()
                output1, output2 = self.model(input)
                output = (output1+output2) / 2.0
                loss_xent1 = conf.train.criterion_xent1(output1, labels)
                loss_xent2 = conf.train.criterion_xent2(output2, labels)
                loss = loss_xent1 + loss_xent2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item(), labels.size(0))
                xent1_losses.update(loss_xent1.item(), labels.size(0))
                xent2_losses.update(loss_xent2.item(), labels.size(0))
                predictions = output.data.max(1)[1]
                correct = (predictions == labels.data).sum()
                acc_cur = correct * 100. / labels.size(0)
                if (batch_idx + 1) % self.print_freq == 0:
                    print("Batch {}/{} Loss {:.6f} ({:.6f}) Acc {:.6f} Xent1_Loss {:.6f} ({:.6f}) Xent2_Loss {:.6f} ({:.6f})" \
                          .format(batch_idx + 1, len(self.loader), losses.val, losses.avg, acc_cur, xent1_losses.val,
                                  xent1_losses.avg, xent2_losses.val, xent2_losses.avg))
            self.save_state(save_path, e)

    def test(self, conf):
        for epoch in range(conf.test.epoch_start, conf.test.epoch_end, conf.test.epoch_interval):
            save_listpath = os.path.join(conf.test.pred_path, conf.exp, 'epoch={}.txt'.format(str(epoch)))
            make_folder_if_not_exist(os.path.dirname(save_listpath))
            fw = open(save_listpath, 'w')
            model_path = os.path.join(conf.save_path, conf.exp, 'epoch={}.pth'.format(str(epoch)))
            print(model_path)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

            with torch.no_grad():
                for batch_idx, (imgs, names) in enumerate(self.test_loader):
                    input1, input2 = self.get_model_input_data_for_test(imgs)
                    output11, output12 = self.model(input1)
                    output21, output22 = self.model(input2)
                    output11 = F.softmax(output11, dim=1)[:, 1]
                    output12 = F.softmax(output12, dim=1)[:, 1]
                    output21 = F.softmax(output21, dim=1)[:, 1]
                    output22 = F.softmax(output22, dim=1)[:, 1]
                    output = (output11 + output12 + output21 + output22) / 4.0
                    for k in range(len(names[0])):
                        write_str = names[0][k] + ' ' + names[1][k] + ' ' + names[2][k] + ' ' + '%.12f' % output[k] + '\n'
                        fw.write(write_str)
                        fw.flush()
            fw.close()


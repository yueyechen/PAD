from data_pipe import get_train_loader,  get_test_loader
import torch
from torch import optim
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import AverageMeter, make_folder_if_not_exist
import os
import pprint
from resnet import resnet18


os.environ['CUDA_VISIBLE_DEVICES']='1'

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

    def get_model_input_data_for_test(self, imgs):
        input0 = torch.cat((imgs[0], imgs[1], imgs[2]), dim=1).cuda()
        input1 = torch.cat((imgs[3], imgs[4], imgs[5]), dim=1).cuda()
        return input0, input1  # for rgb+depth+ir

    def train(self, conf):
        self.model.train()
        SL1_losses = AverageMeter()
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
                labels = labels.cuda().float().unsqueeze(1)
                output = self.model(input)
                loss_SL1 = conf.train.criterion_SL1(output, labels)
                loss = loss_SL1
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.update(loss.item(), labels.size(0))
                SL1_losses.update(loss_SL1.item(), labels.size(0))
                if (batch_idx + 1) % self.print_freq == 0:
                    print("Batch {}/{} Loss {:.6f} ({:.6f}) SL1_Loss {:.6f} ({:.6f})" \
                          .format(batch_idx + 1, len(self.loader), losses.val, losses.avg, SL1_losses.val,
                                  SL1_losses.avg))
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
                    output1 = self.model(input1)
                    output2 = self.model(input2)
                    output = (output1 + output2 ) / 2.0
                    for k in range(len(names[0])):
                        write_str = names[0][k] + ' ' + names[1][k] + ' ' + names[2][k] + ' ' + '%.12f' % output[k] + '\n'
                        fw.write(write_str)
                        fw.flush()
            fw.close()


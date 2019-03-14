from easydict import EasyDict as edict
from torchvision import transforms as trans
import torch.nn as nn

def get_config(training = True):
    conf = edict()
    conf.model = edict()
    conf.train = edict()
    conf.eval = edict()

    conf.data_folder = '/home2/xuejiachen/data/huoti/align/256x256_v1' #data root for training, and testing
    conf.result_path = '/home2/xuejiachen/PAD/work_space/result'
    conf.log_path = '/home2/xuejiachen/PAD/work_space/log' #path for saving loggers in training process
    conf.save_path = '/home2/xuejiachen/PAD/work_space/save' #path for save model in training process
    conf.train_list =  '/home2/xuejiachen/data/huoti/align/256x256_v1/quarter_face_train_list.txt' #training list
    conf.test_list = '/home2/xuejiachen/data/huoti/test_list.csv'
    conf.batch_size = 128
    conf.exp = 'commit'

    conf.model.input_size = [112,112]
    conf.model.random_offset = [16,16] #for random crop
    conf.model.use_senet = True
    conf.model.se_reduction = 16
    conf.model.drop_out = 0.7
    conf.model.embedding_size = 1024

    conf.pin_memory = True
    conf.num_workers = 3

#--------------------Training Config ------------------------
    if training:
        conf.train.lr = 0.01
        conf.train.milestones = [80, 140, 180]
        conf.train.epoches = 200
        conf.train.momentum = 0.9
        conf.train.gamma = 0.1
        conf.train.criterion_SL1 = nn.SmoothL1Loss()

        conf.train.transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

#--------------------Inference Config ------------------------
    else:
        conf.test = edict()
        conf.test.epoch_start = 150
        conf.test.epoch_end = 200
        conf.test.epoch_interval = 8
        conf.test.pred_path = '/home2/xuejiachen/PAD_NEW/work_space/test_pred' #path for save predict result
        conf.test.transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    return conf

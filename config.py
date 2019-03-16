from easydict import EasyDict as edict
from torchvision import transforms as trans
import torch.nn as nn

def get_config(training = True):
    conf = edict()
    conf.model = edict()
    conf.train = edict()
    conf.eval = edict()

    conf.data_folder = 'path to data root' #data root for training, and testing
#     conf.result_path = '/home2/xuejiachen/PAD/work_space/result'
#     conf.log_path = '/home2/xuejiachen/PAD/work_space/log' #path for saving loggers in training process
    conf.save_path = './work_space/save' #path for save model in training process
    conf.train_list =  'quarter_face_train_list.txt' #path where training list is 
    conf.test_list = 'test_public_list.txt' #path where test list is
    conf.batch_size = 128
    conf.exp = 'commit'

    conf.model.input_size = [112,112] #the input size of our model
    conf.model.random_offset = [16,16] #for random crop
    conf.model.use_senet = True #senet is adopted in our resnet18 model
    conf.model.se_reduction = 16 #parameter concerning senet
    conf.model.drop_out = 0.7 #we add dropout layer in our resnet18 model
    conf.model.embedding_size = 1024 #feature size of our resnet18 model

    conf.pin_memory = True
    conf.num_workers = 3

#--------------------Training Config ------------------------
    if training:
        conf.train.lr = 0.01 # the initial learning rate
        conf.train.milestones = [80, 140, 180] #epoch milestones decreased by a factor of 10
        conf.train.epoches = 200 #we trained our model for 200 epoches
        conf.train.momentum = 0.9 #parameter in setting SGD
        conf.train.gamma = 0.1 #parameter in setting lr_scheduler
        conf.train.criterion_SL1 = nn.SmoothL1Loss() #we use SmoothL1Loss in training stage

        conf.train.transform = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

#--------------------Inference Config ------------------------
    else:
        conf.test = edict()
        conf.test.epoch_start = 150
        conf.test.epoch_end = 200
        conf.test.epoch_interval = 8 #we set a range of epoches for testing
        conf.test.pred_path = '/home2/xuejiachen/PAD_NEW/work_space/test_pred' #path for save predict result
        conf.test.transform = trans.Compose([ #convert input from PIL.Image to Tensor and normalized
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    return conf

# coding=utf-8
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import csv

def get_train_loader(conf):
    print('train dataset: {}'.format(conf.train_list))
    ds = MyDataset_huoti(conf)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=True, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader

def get_test_loader(conf):
    print('val dataset: {}'.format(conf.test_list))
    ds = MyDataset_huoti_test(conf)
    loader = DataLoader(ds, batch_size=conf.batch_size, shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers)
    return loader

def default_loader_rgb(path):
    img = Image.open(path).convert('RGB')
    return img

def default_loader_gray(path):
    img = Image.open(path).convert('L')
    return img

class MyDataset_huoti(Dataset):
    def __init__(self, conf, target_transform=None, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        fh = open(conf.train_list,'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], words[1], words[2], int(words[3])))

        self.imgs = imgs
        self.transform = conf.train.transform
        self.target_transform = target_transform
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray
        self.root = conf.data_folder
        self.input_size = conf.model.input_size
        self.random_offset = conf.model.random_offset

    def __getitem__(self, index):
        fn1, fn2, fn3, label = self.imgs[index]
        img1 = self.loader_rgb(os.path.join(self.root,fn1))
        img2 = self.loader_gray(os.path.join(self.root,fn2))
        img3 = self.loader_gray(os.path.join(self.root,fn3))

        offset_x = random.randint(0, self.random_offset[0])
        offset_y = random.randint(0, self.random_offset[1])
        img1 = img1.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))
        img2 = img2.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))
        img3 = img3.crop((offset_x, offset_y, offset_x + self.input_size[0], offset_y + self.input_size[1]))

        # random horizantal flip
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            img3 = img3.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return [img1,img2,img3], label

    def __len__(self):
        return len(self.imgs)

class MyDataset_huoti_test(Dataset):
    def __init__(self, conf, loader_rgb=default_loader_rgb, loader_gray=default_loader_gray):
        with open(conf.test_list) as f:
            f_csv = csv.reader(f)
            _ = next(f_csv)
            imgs = []
            for row in f_csv:
                imgs.append((row[0], row[1], row[2]))

        self.imgs = imgs
        self.transform = conf.test.transform
        self.loader_rgb = loader_rgb
        self.loader_gray = loader_gray
        self.root = conf.data_folder

    def __getitem__(self, index):
        fn1, fn2, fn3= self.imgs[index]
        img11 = self.loader_rgb(os.path.join(self.root,fn1))
        img12 = self.loader_gray(os.path.join(self.root,fn2))
        img13 = self.loader_gray(os.path.join(self.root,fn3))
        size_c = (8, 8, 120, 120)
        img11 = img11.crop(size_c)
        img12 = img12.crop(size_c)
        img13 = img13.crop(size_c)

        img21 = self.loader_rgb(os.path.join(self.root, fn1)).transpose(Image.FLIP_LEFT_RIGHT)
        img22 = self.loader_gray(os.path.join(self.root, fn2)).transpose(Image.FLIP_LEFT_RIGHT)
        img23 = self.loader_gray(os.path.join(self.root, fn3)).transpose(Image.FLIP_LEFT_RIGHT)
        img21 = img21.crop(size_c)
        img22 = img22.crop(size_c)
        img23 = img23.crop(size_c)

        if self.transform is not None:
            img11 = self.transform(img11)
            img12 = self.transform(img12)
            img13 = self.transform(img13)
            img21 = self.transform(img21)
            img22 = self.transform(img22)
            img23 = self.transform(img23)
        return [img11,img12,img13,img21,img22,img23], [fn1, fn2, fn3]

    def __len__(self):
        return len(self.imgs)
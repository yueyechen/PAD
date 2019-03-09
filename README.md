# ReadME

## 1.数据集
本项目的数据来源于 Chalearn CASIA-SURF[1](https://competitions.codalab.org/competitions/20853#learn_the_details) 

对CASIA数据集进行基本的预处理，并将数据上传至百度云盘[下载地址](https://pan.baidu.com/s/1uKopv4uTpALKKPqzuFBOyg);提取码：d27c 



数据集下载后，将训练数据和测试数据解压到`data`目录下，

## 2.环境配置
### 2.1 环境描述

本项目运行所使用系统为ubuntu 16.04 LST，编程语言为python3，pytorch 0.41；
硬件环境为：
> Nvidia GPU支持（GPU：GeForce GTX 1080Ti）
> 
> CPU型号是 `Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz`

### 2.2 软件环境
该项目的requirements.txt中包含项目运行所必须的python 库。
打开终端，跳转到该项目下，执行以下命令，安装所需的python类库。
```
pip3 install -r requirements.txt
```



## 3. 训练模型

**Step1**: 修改配置文件

对config.py 的变量进行以下修改：

```python
conf.data_folder = '<训练数据所在的路径>' 
```

``` 
conf.train_list = '<train_list.txt的路径>'
```

**Step2**: 训练模型

运行 `train.py`文件

```
python3 train.py # 训练脚本
```



程序运行完毕以后，会在save目录下生成训练模型，在log目录下记录训练情况。

## 4. 测试

**Step1**:  修改配置文件

对config.py 的变量进行以下修改：

```python

conf.val_list = '<test_list.txt>' # 测试文件索引
conf.save_path = '/test_pred' # 指定生成预测文件的根目录，默认在当前文件的test_pred 目录下
conf.exp = '' # 指定生成预测文件所在的目录

```


**Step2**： 运行测试脚本
为了方便测试，可以先下载已经训练好的模型，上传至百度云盘链接：https://pan.baidu.com/s/1WXY_agWo22qMLw6BZL_hKw 
提取码：0mhv 

打开终端，切换到该项目下，执行以下命令：
```
python3 test.py
```

运行以下命令进行测试，该脚本会针对每个epoch，在目录`${conf.save_path}/${conf.exp}`生成以 `epoch={%d}.txt` 命名的文件。 


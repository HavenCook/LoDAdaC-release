from src.DistDataModel import DistDataModel
import os
import yaml
from yaml import Loader
from src.Compressor import *

compressor_map = {'none': NoneCompressor(),
                   'topk30': TopKCompressor(0.3),
                   'topk40': TopKCompressor(0.4),
                   'topk50': TopKCompressor(0.5),
                   'topk60': TopKCompressor(0.6),
           'qsgd': QSGDCompressor(4)}

# First we construct our data distributed neural network.
# RANK options: [1,4,9,16], must match slurm script if running on multiple nodes
RANK = 16

# dataset options: ["FashionMNIST","CIFAR10","Shakespeare","OpenWebText"]
dataset = ["CIFAR10"]

# compress_method options: ["none","topk30","topk40","topk50","topk60","qsgd"]
compress_method = ["topk40"]

# variety_type options: ["index","label"]
variety_type = ["index"]

# optimizers format:
# (optimizer_name, comm_set, learning_rate)
# optimizer_name options: ["DistributedAdam","DistributedAdaGrad","DistributedAMSGrad",'HeavyBall","CDProxSGT","DoCoM","SQuARM-SGD"]
# comm_set options: [['x','g'],['x']]
# learning_rate: positive float
optimizers = [("DistributedAMSGrad",['x','g'],0.001)]

# lr_decay options: ["none","cosine"]
lr_decay = ["none"]

# model options: nanoGPT, fixup_resnet20, LeNet5
model = "fixup_resnet20"
bs = 32
dev = 'gpu'
nv = False
models = []
names = []
k_list = [20]

for data in dataset:
    for k in k_list:
        for compress in compress_method:
            for opt,comm_set,lr in optimizers:
                for variety in variety_type:
                    for decay in lr_decay:
                        if "topk" in compress and (opt == "NewAlg" or opt == "NewAlg2"):
                            comm_set = ['x_bar']
                        if "topk" in compress and (opt == "DistributedAdam" or opt == "DistributedAdaGrad" or opt == "HeavyBall" or opt == "CDProxSGT"):
                            comm_set = ['x_bar','g_bar']
                        models.append(DistDataModel(model=model,dataset=data,topology="ring",optimizer=opt,comm_set=comm_set,
                                    batch_size=bs,device=dev,track=True,seed=1337,compressor=compressor_map[compress],
                                    variety=variety,lr_decay=decay,lr=lr,nvlink=nv,k=k))
                        names.append(data+"-"+opt+"-"+compress+"-Rank-"+str(RANK)+ "-LR_Decay-"+decay+"-Variety-"+variety+"-ring"+"-K-"+str(k))

for i in range(len(models)):
    e_model = models[i]
    e_model.epochs=250*int(names[i].split("-")[-1])
    print(f"Model initialized with the following parameters: {names[i]}")
    print("Now Starting training....")
    names[i] = names[i]+"-"+str(e_model.epochs)
    training_history = e_model.train(verbose=True,output_file=names[i])
    print("training finished...")
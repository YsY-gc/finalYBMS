
from importlib import import_module
from torch.utils.data import DataLoader



class Data():
    def __init__(self):
        self.loader_train = None  
        module_train = import_module('dataset.' + 'dual_unsupervised')
        trainset = module_train.TrainSetLoader('E:\\finallYBMS\\peoplePic\\DATASETS_X')
        self.loader_train = DataLoader(
            trainset,
            batch_size=1,# 必须为1
            num_workers=3,
            shuffle=True,

            )

        module_test = import_module('dataset.' +  'dual_unsupervised')
        testset = module_test.TestSetLoader('E:\\finallYBMS\\peoplePic\\DATASETS_X')
        self.loader_test = DataLoader(
            testset,
            batch_size=1,# 必须为1
            num_workers=3,
            shuffle=False,

        )
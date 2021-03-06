import torch.utils.data as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        if dataset_name == 'GoproHDF5SingleTest2Bin':
            from data.GoproHDF5SingleTest2Bin import GoProHDF5
            dataset = GoProHDF5(opt, is_for_train=is_for_train)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)
        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt, is_for_train):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._opt = opt
        self._is_for_train = is_for_train

    @property
    def name(self):
        return self._name

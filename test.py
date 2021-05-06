import time
from opts.test_opt import TestOpt
from data.dataloader import CustomDatasetDataLoader
from models.models import ModelsFactory
from collections import OrderedDict
import os
import shutil
import torch
from networks.networks import NetworksFactory

import time
from opts.test_opt import TestOpt
from data.dataloader import CustomDatasetDataLoader
from models.models import ModelsFactory
from collections import OrderedDict
from utils.cv_utils import save_image
import os
import shutil
import torch
from networks.networks import NetworksFactory
from utils import cv_utils, criterion

class Test:
    def __init__(self):
        self._opt = TestOpt().parse()
        self._opt.distributed=False
        print(self._opt)
        self.cal_time = 0.0

        self.output_dir = os.path.expanduser(self._opt.output_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        print(self.output_dir)

        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)

        self._dataset_test = data_loader_test.load_data()

        self._dataset_test_size = len(data_loader_test)
        print('#test images = %d' % self._dataset_test_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)

        self._test()

    def _test(self):
        # train epoch
        self._test_epoch()

    def _test_epoch(self):

        self._model.set_eval()

        val_errors = OrderedDict()

        val_start_time = time.time()

        f = open(os.path.join(self.output_dir, 'results.txt'), 'w')

        measurement = {}
        for i_val_batch, val_batch in enumerate(self._dataset_test):

            self._model.set_input(val_batch)
            with torch.no_grad():
                self._model.forward()

            loss_dict = self._model.get_current_scalars()

            message = ''
            f.write(val_batch['dataname'][0] + ' ')
            for k, v in loss_dict.items():
                if k in measurement:
                    measurement[k] += v
                else:
                    measurement[k] = v
                f.write(k + ': ' + str(v) + ' ')
                message+= k + ': ' + str(v) + ' '
            f.write('\n')

            self._model.save_output(self.output_dir, output_gt=True, output_blur=True)

            print("{}/{}; '{}'".format(i_val_batch, self._dataset_test_size, message))

        if not self._opt.qualitative:
            t = time.time() - val_start_time
            
            for k in measurement.keys():
                measurement[k] /= self._dataset_test_size
                f.write(k + ': ' + str(measurement[k]) + ' ')
            f.write('\n')

            message = '(Total time: %.4f s)' % (t)

            print(message)
            print(measurement)

if __name__ == "__main__":
    Test()

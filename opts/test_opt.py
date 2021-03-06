from opts.base_opt import BaseOpt

class TestOpt(BaseOpt):
    def initialize(self):
        BaseOpt.initialize(self)
        self.parser.add_argument('--test_data_dir', type=str, default='/data/05_Event/tmp_dvs')
        self.parser.add_argument('--data_dir', type=str, default='/data/05_Event/tmp_dvs')
        # augmentation.
        self.parser.add_argument('--test_batch_size', type=int, default=4, help='input batch size')
        self.parser.add_argument('--output_dir', type=str, default='./output', help='output_path')

        self.is_train = False

        self.parser.add_argument('--output_gt', action='store_true', help='Save gt')
        self.parser.add_argument('--output_blur', action='store_true', help='Save blur')
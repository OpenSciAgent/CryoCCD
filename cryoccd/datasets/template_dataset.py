from cryoccd.datasets.base_dataset import BaseDataset, get_transform
import logging

logger = logging.getLogger(__name__)


class TemplateDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = 'temp'    # needs to be a string
        data_A = None    # needs to be a tensor
        data_B = None    # needs to be a tensor
        return {'data_A': data_A, 'data_B': data_B, 'path': path}

    def __len__(self):
        return len(self.image_paths)

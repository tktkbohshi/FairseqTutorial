import os
from .data_utility import read_image_ids, ImageCaptionDataset, CaptionDataset, ImageDataset

from fairseq.data import Dictionary, data_utils
from fairseq.tasks import FairseqTask, register_task

@register_task('captioning')
class CaptioningTask(FairseqTask):
    @staticmethod
    def add_args(parser):   
        parser.add_argument('--features-dir', default='output',
                            help='image features directory')
        parser.add_argument('--captions-dir', default='output',
                            help='image captions directory')
        parser.add_argument('--captions-lang', default='en', choices=['en'],
                            help='caption language')
        parser.add_argument('--max-source-positions', default=64, type=int, metavar='N', help='max number of objects in the source image')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N', help='max number of tokens in the target sequence')
        parser.add_argument('--sentencepiece-model', default='data/sp')
        parser.add_argument('--sentencepiece-dict', default='data/sp')

    @classmethod
    def setup_task(cls, args, **kwargs):
        # load a caption's dict
        #captions_dict_file = os.path.join(args.captions_dir, f'dict.{args.captions_lang}.txt')
        #captions_dict = Dictionary.load(captions_dict_file)
        captions_dict = Dictionary()
        return CaptioningTask(args, captions_dict)

    def __init__(self, args, captions_dict):
        self.args = args
        self.captions_dict = captions_dict
        super().__init__(args)

    def load_dataset(self, split, **kwargs):
        # read image path
        # file path -> list[image_path]
        #features_dir = os.path.join(self.args.features_dir, f'{split}-features-{self.args.features}')
        image_ids_file = os.path.join(self.args.captions_dir, f'{split}.src')
        image_ids = read_image_ids(image_ids_file)
        image_ds = ImageDataset(image_ids, self.args.max_source_positions)

        # read caption data
        captions_file = os.path.join(self.args.captions_dir, f'{split}.dst')
        captions_ds = CaptionDataset(captions_file, self.args.sentencepiece_model)

        self.datasets[split] = ImageCaptionDataset(img_ds=image_ds,
                                                        cap_ds=captions_ds,
                                                        cap_dict=self.captions_dict,
                                                        shuffle=True)

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return self.captions_dict
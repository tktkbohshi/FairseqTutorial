from atexit import register
import torch
import torch.nn.functional as F

import torchvision.models as models
from fairseq.models import FairseqEncoder, BaseFairseqModel, FairseqDecoder
from fairseq.models import register_model, register_model_architecture, transformer
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# FairseqEncoderクラスを継承しておくのが無難
class CNNEncoder(FairseqEncoder):
    def __init__(self, embed_size=256):
        # Load the pretrained ResNet-152 and replace top fc layer.
        super(CNNEncoder, self).__init__(dictionary=None)
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images, src_lengths):
        #Extract feature vectors from input images
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        print(features.size())
        return features

# FairseqDecoderクラスを継承しておくのが無難
class LSTMDecoder(FairseqDecoder):
    def __init__(self, dictionary, encoder_hidden_dim=256, embed_dim=256, hidden_dim=256):
        super().__init__(dictionary)

        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=encoder_hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=False,
        )

        # Define the output projection.
        self.output_projection = nn.Linear(hidden_dim, len(dictionary))

    def forward(self, prev_output_tokens, encoder_out):
        print(prev_output_tokens)
        bsz, tgt_len = prev_output_tokens.size()
        x = prev_output_tokens
        output, _ = self.lstm(
            x,  # convert to shape `(tgt_len, bsz, dim)`
        )
        x = output.transpose(0, 1)  # convert to shape `(bsz, tgt_len, hidden)`
        x = self.output_projection(x)

        return x, None

from fairseq.models import FairseqEncoderDecoderModel, register_model

@register_model('image_caption')
class ImageCaptionModel(FairseqEncoderDecoderModel):

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the encoder embeddings',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the encoder hidden state',
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.
        # In this case we'll just return a SimpleLSTMModel instance.

        # Initialize our Encoder and Decoder.
        encoder = CNNEncoder(
            embed_size=args.encoder_embed_dim
        )
        decoder = LSTMDecoder(
            dictionary=task.target_dictionary,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
        )
        model = ImageCaptionModel(encoder, decoder)

        # Print the model architecture.
        print(model)

        return model

from fairseq.models import register_model_architecture

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'simple_lstm'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.

@register_model_architecture('image_caption', 'image_caption')
def tutorial_simple_caption(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
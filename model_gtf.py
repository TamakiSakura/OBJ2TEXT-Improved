import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.autograd import Variable
import transformer.Constants as Constants
from transformer.Utils import *
from transformer.Modules import BottleLinear as Linear
from transformer.Layers import EncoderLayer, DecoderLayer

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained VGG16 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.vgg16_feat = vgg16.features
        self.vgg16_clf = nn.Sequential(*list(vgg16.classifier.children())[:-1])
        self.vgg16_feat.eval()
        self.vgg16_clf.eval()
        self.linear = nn.Linear(4096, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.vgg16_feat(images)
        features = features.view(features.size(0), -1)
        features = self.vgg16_clf(features)
        features = self.bn(self.linear(features))
        return features


class LayoutEncoder(nn.Module):
    def __init__(self, layout_encoding_size, hidden_size, vocab_size, num_layers, 
                 n_head=8, d_k=64, d_v=64, d_inner_hid=1024, 
                 dropout=0.1):
        """Set the hyper-parameters and build the layers."""
        super(LayoutEncoder, self).__init__()
        d_word_vec = layout_encoding_size
        d_model = layout_encoding_size
        
        self.label_encoder = nn.Embedding(vocab_size, layout_encoding_size)
        self.location_encoder = nn.Linear(4, layout_encoding_size)
        
        self.init_weights()

        self.layer_stack = nn.ModuleList([
                           EncoderLayer(
                           d_model, d_inner_hid, n_head, 
                           d_k, d_v, dropout=dropout)
                           for _ in range(num_layers)])

    def init_weights(self):
        """Initialize weights."""
        self.label_encoder.weight.data.uniform_(-0.1, 0.1)
        self.location_encoder.weight.data.uniform_(-0.1, 0.1)
        self.location_encoder.bias.data.fill_(0)

    def forward(self, label_seqs, location_seqs, lengths):
        """Encode the Layout"""
        # encode label sequences
        label_encoding = self.label_encoder(label_seqs)
        
        # encode location sequences
        location_encoding = self.location_encoder(location_seqs.view(-1, 4))
        location_encoding = location_encoding.view(label_encoding.size(0), -1, location_encoding.size(1))

        # layout encoding - batch_size x max_seq_len x embed_size
        layout_encoding = label_encoding + location_encoding

        enc_output = layout_encoding

        enc_slf_attn_mask = get_attn_padding_mask(label_seqs, 
                                                  label_seqs)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=enc_slf_attn_mask)

        return enc_output
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers,
                 n_max_seq=50, 
                 n_head=8, d_k=64, d_v=64, d_inner_hid=1024, 
                 dropout=0.1):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.tgt_word_emb = nn.Embedding(vocab_size, embed_size, Constants.PAD)
        
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
  
        d_model = embed_size
        d_word_vec = embed_size

        self.d_model = d_model

        self.position_enc = nn.Embedding(
            n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(num_layers)])

        self.linear = nn.Linear(embed_size, vocab_size)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)         

    def forward(self, src_seq, tgt_seq, enc_output, length):
        """Decode the input into sentence"""
        # Word embedding look up
        dec_input = self.tgt_word_emb(tgt_seq)

        # Position Encoding addition
        # dec_input += self.position_enc(tgt_seq)
        # TODO: UNIMPLEMENTED

        # Decode
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)

        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)
        
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)
        
        output = self.linear(dec_output)
        output = pack(output, length, batch_first=True).data
        
        return output
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        # TODO INCORRECT
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted).unsqueeze(1)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        return sampled_ids.squeeze()

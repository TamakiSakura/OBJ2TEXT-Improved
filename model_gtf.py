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
from transformer.Beam import Beam

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

    def forward(self, src_seq, tgt_seq, enc_output, length, flat=True):
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
        if flat:
            output = pack(output, length, batch_first=True).data
        
        return output
    
    def sample(self, src_seq, enc_output):
        """Samples captions for given image features (Greedy search)."""
        beam_size = 2
        batch_size = src_seq.size(0)

        self.softmax = nn.LogSoftmax()        
        self.tt = torch.cuda if torch.cuda.is_available() else torch
         
        # Repeat Data
        src_seq = Variable(
            src_seq.data.repeat(1, beam_size).view(
                src_seq.size(0) * beam_size, src_seq.size(1)))

        enc_output = Variable(
            enc_output.data.repeat(1, beam_size, 1).view(
                enc_output.size(0) * beam_size, enc_output.size(1), enc_output.size(2)))

        #--- Prepare beams
        beams = [Beam(beam_size, torch.cuda.is_available()) for _ in range(batch_size)]
        beam_inst_idx_map = {
            beam_idx: inst_idx for inst_idx, beam_idx in enumerate(range(batch_size))}
        n_remaining_sents = batch_size

        #- Decode
        for i in range(20):
            len_dec_seq = i + 1

            # -- Preparing decoded data seq -- #
            # size: batch x beam x seq
            dec_partial_seq = torch.stack([
                b.get_current_state() for b in beams if not b.done])
            # size: (batch * beam) x seq
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            # wrap into a Variable
            dec_partial_seq = Variable(dec_partial_seq, volatile=True)

            if torch.cuda.is_available():
                dec_partial_seq = dec_partial_seq.cuda()

            # -- Decoding -- #
            dec_output = self(
                src_seq, dec_partial_seq, enc_output, 
                [len_dec_seq] * batch_size, False)
            dec_output = dec_output[:, -1, :] # (batch * beam) * d_model
            out = self.softmax(dec_output)

            # batch x beam x n_words
            word_lk = out.view(n_remaining_sents, beam_size, -1).contiguous()

            active_beam_idx_list = []
            
            for beam_idx in range(batch_size):
                if beams[beam_idx].done:
                    continue

                inst_idx = beam_inst_idx_map[beam_idx]
                if not beams[beam_idx].advance(word_lk.data[inst_idx]):
                    active_beam_idx_list += [beam_idx]

            if not active_beam_idx_list:
                # all instances have finished their path to <EOS>
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_inst_idxs = self.tt.LongTensor(
                [beam_inst_idx_map[k] for k in active_beam_idx_list])

            # update the idx mapping
            beam_inst_idx_map = {
                beam_idx: inst_idx for inst_idx, beam_idx in enumerate(active_beam_idx_list)}

            def update_active_seq(seq_var, active_inst_idxs):
                ''' Remove the src sequence of finished instances in one batch. '''

                inst_idx_dim_size, b = seq_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = inst_idx_dim_size, b

                # select the active instances in batch
                original_seq_data = seq_var.data.view(n_remaining_sents, -1)
                active_seq_data = original_seq_data.index_select(0, active_inst_idxs)
                active_seq_data = active_seq_data.view(*new_size)

                return Variable(active_seq_data, volatile=True)

            def update_active_enc_info(enc_info_var, active_inst_idxs):
                ''' Remove the encoder outputs of finished instances in one batch. '''

                inst_idx_dim_size, b, c = enc_info_var.size()
                inst_idx_dim_size = inst_idx_dim_size * len(active_inst_idxs) // n_remaining_sents
                new_size = inst_idx_dim_size, b, c
                
                # select the active instances in batch
                original_enc_info_data = enc_info_var.data.view(
                    n_remaining_sents, -1, self.d_model)
                active_enc_info_data = original_enc_info_data.index_select(0, active_inst_idxs)
                active_enc_info_data = active_enc_info_data.view(*new_size)

                return Variable(active_enc_info_data, volatile=True)

            src_seq = update_active_seq(src_seq, active_inst_idxs)
            enc_output = update_active_enc_info(enc_output, active_inst_idxs)

            #- update the remaining size
            n_remaining_sents = len(active_inst_idxs)

        #- Return useful information
        all_hyp, all_scores = [], []
        n_best = 1

        for beam_idx in range(batch_size):
            scores, tail_idxs = beams[beam_idx].sort_scores()
            all_scores += [scores[:n_best]]

            hyps = [beams[beam_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        
        return all_hyp

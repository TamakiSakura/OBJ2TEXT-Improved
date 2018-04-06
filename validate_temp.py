import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_val_loader_temp import get_loader
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN, LayoutEncoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from nltk.translate.bleu_score import *
from nltk import word_tokenize
import json


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def compute_bleu(reference_tokenized, predicted_sentence):
    """
    Given a reference sentence, and a predicted sentence, compute the BLEU similary between them.
    """
    predicted_tokenized = word_tokenize(predicted_sentence.lower())
    return sentence_bleu(reference_tokenized, 
                         predicted_tokenized,
                         smoothing_function = SmoothingFunction().method1)

def validation(layout_encoder,decoder, args,vocab,transform, batch_size,encoder=None):
    yolo=False
    # Build data loader
    data_loader_val = get_loader(args.image_dir_val, args.caption_path_val, vocab, 
                                 args.MSCOCO_result_val, args.coco_detection_result_val,
                                 transform, batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 dummy_object=99,
                                 yolo=yolo)
    bleu_score_all = 0
    bleu_score_batch = 0
    n = 0
    for i, (img_id, images, captions, 
            label_seqs, location_seqs, visual_seqs, 
            layout_lengths) in enumerate(data_loader_val):
        # Set mini-batch dataset
        images = to_var(images)
        # Modify This part for using visual features or not
        if yolo:
            visual_seqs = to_var(visual-ses)
        else:
            visual_seqs = None
         
        # features = encoder(images)
        layout_encoding = layout_encoder(label_seqs, location_seqs, layout_lengths)
        # comb_features = features + layout_encoding
        comb_features = layout_encoding
        sampled_ids = decoder.sample(comb_features)

        sampled_ids = sampled_ids.cpu().data.numpy()
        for j in range(len(sampled_ids)):
            # Decode word_ids to words
            sampled_caption = []
            sampled_id = sampled_ids[j][1:]
            for word_id in sampled_id:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                sampled_caption.append(word)

            predicted_sentence = ' '.join(sampled_caption)
            print("predict: "+ predicted_sentence)
            ref_caption = captions[j][0]
            reference_sentence = ' '.join(ref_caption)
            print("reference: "+ reference_sentence)
            #print(captions[j])
            score = compute_bleu(captions[j], predicted_sentence)
            bleu_score_all += score
            bleu_score_batch += score
        
        print("Validation step %d, avg bleu: %f"%(i,bleu_score_batch/batch_size))
        bleu_score_batch = 0

    print("Total number of Validation Images : %d, overall avg bleu: %f"%(i,bleu_score_all/(i*batch_size)))



def save_output(layout_encoder,decoder, args,vocab,transform, batch_size,encoder=None):
    yolo=False
    # Build data loader
    data_loader_val = get_loader(args.image_dir_val, args.caption_path_val, vocab, 
                                 args.MSCOCO_result_val, args.coco_detection_result_val,
                                 transform, batch_size,
                                 shuffle=True, num_workers=args.num_workers,
                                 dummy_object=99,
                                 yolo=yolo)
    bleu_score_all = 0
    bleu_score_batch = 0
    n = 0
    output_vector = []
    for i, (img_id, images, captions, 
            label_seqs, location_seqs, visual_seqs, 
            layout_lengths) in enumerate(data_loader_val):
        # Set mini-batch dataset
        images = to_var(images)
        # Modify This part for using visual features or not
        if yolo:
            visual_seqs = to_var(visual-ses)
        else:
            visual_seqs = None
         
        # features = encoder(images)
        layout_encoding = layout_encoder(label_seqs, location_seqs, layout_lengths)
        # comb_features = features + layout_encoding
        comb_features = layout_encoding
        sampled_ids = decoder.sample(comb_features)

        sampled_ids = sampled_ids.cpu().data.numpy()
        for j in range(len(sampled_ids)):
            # Decode word_ids to words

            sampled_caption = []
            sampled_id = sampled_ids[j][1:]
            for word_id in sampled_id:
                word = vocab.idx2word[word_id]
                if word == '<end>':
                    break
                sampled_caption.append(word)

            predicted_sentence = ' '.join(sampled_caption)
            # Save as json type
            data = {}
            data["image_id"] = img_id[j]
            data["caption"] = predicted_sentence
            output_vector.append(data)
        print("Validation step %d"%(i))
    return output_vector


def main(args):
	# Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build Models
    # encoder = EncoderCNN(args.embed_size)
    # encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    layout_encoder = LayoutEncoder(args.layout_embed_size, args.embed_size, 100, args.num_layers)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
    

    # Load the trained model parameters
    layout_encoder.load_state_dict(torch.load(args.layout_encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    
    # If use gpu
    if torch.cuda.is_available():
        layout_encoder.cuda()
        decoder.cuda()

    # validation(layout_encoder,decoder, args,vocab,transform,args.batch_size)
    out = save_output(layout_encoder,decoder, args,vocab,transform,args.batch_size)
    with open('bsl_output.txt', 'w') as outfile:
        json.dump(out, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--encoder_path', type=str, default='./models/encoder-5-3000.pkl',
    #                     help='path for trained encoder')
    parser.add_argument('--layout_encoder_path', type=str, default='./models/layout_encoding-4-11000.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/decoder-4-11000.pkl',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    # from train.py
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--image_dir_val', type=str, default='./data/resized2014_val',
                        help='directory for resized validation images')
    parser.add_argument('--caption_path_val', type=str,
                        default='./data/annotations/captions_val2014.json',
                        help='path for validation annotation json file')
    parser.add_argument('--MSCOCO_result_val', type=str,
                        default='./data/annotations/instances_val2014.json',
                        help='path coco object detection result file')
    parser.add_argument('--coco_detection_result_val', type=str,
                        default='./data/annotations/instances_val2014.json',
                        help='path coco object detection result file for valiadation')

    parser.add_argument('--embed_size', type=int , default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--layout_embed_size', type=int, default=256,
                        help='layout encoding size')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')

    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=123, help='random generator seed')

    
    args = parser.parse_args()
    main(args)

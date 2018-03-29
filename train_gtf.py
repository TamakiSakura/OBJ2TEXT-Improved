import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from build_vocab import Vocabulary
from model_gtf import EncoderCNN, DecoderRNN, LayoutEncoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torchvision import transforms
import transformer.Constants as Constants

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
        # transforms.RandomCrop(args.crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.Scale(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    Constants.set_constant(vocab)

    yolo = True
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             args.MSCOCO_result, args.coco_detection_result,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers,
                             dummy_object=99,
			     yolo=yolo)
    # Build the models
    encoder = EncoderCNN(args.embed_size)
    # the layout encoder hidden state size must be the same with decoder input size
    layout_encoder = LayoutEncoder(args.layout_embed_size, args.embed_size, 100, args.num_layers)
    decoder = DecoderRNN(args.embed_size, args.hidden_size,
                         len(vocab), args.num_layers)

    if torch.cuda.is_available():
        encoder.cuda()
        layout_encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(layout_encoder.parameters()) + list(decoder.parameters()) + \
	     list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Train the Models
    total_step = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths,
                label_seqs, location_seqs, visual_seqs,
                layout_lengths) in enumerate(data_loader):
            for idx_length in range(len(lengths)):
                lengths[idx_length] -= 1

            # Set mini-batch dataset
            images = to_var(images)
            targets = to_var(captions)
            label_seqs = to_var(label_seqs)
            location_seqs = to_var(location_seqs)
            if yolo:
                visual_seqs = to_var(visual_seqs)
            else:
                visual_seqs = None
            targets = pack(targets[:,1:], lengths, batch_first=True)[0]

            for idx_length in range(len(lengths)):
                captions[idx_length][lengths[idx_length]] = 0
            captions = to_var(captions)
            # Forward, Backward and Optimize
            # decoder.zero_grad()
            # layout_encoder.zero_grad()
            # encoder.zero_grad()

            # Modify This part for using visual features or not

            # vgg_features = encoder(images)
            # visual_seqs = None
            layout_encoding = layout_encoder(label_seqs, location_seqs, visual_seqs,
                                             layout_lengths)
            # comb_features = vgg_features + layout_encoding
            comb_features = layout_encoding

            outputs = decoder(label_seqs, captions, comb_features, lengths)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      % (epoch, args.num_epochs, i, total_step,
                         loss.data[0], np.exp(loss.data[0])))

                # Save the models
            if (i + 1) % args.save_step == 0:
                torch.save(decoder.state_dict(),
                           os.path.join(args.model_path,
                                        'gtf-decoder-%d-%d.pkl' % (epoch + 1, i + 1)))
                torch.save(encoder.state_dict(),
                           os.path.join(args.model_path,
                                        'gtf-encoder-%d-%d.pkl' % (epoch + 1, i + 1)))
                torch.save(layout_encoder.state_dict(),
                           os.path.join(args.model_path,
                                        'gtf-layout_encoder-%d-%d.pkl' % (epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/',
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized2014',
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--MSCOCO_result', type=str,
                        default='./data/annotations/instances_train2014.json',
                        help='path coco object detection result file')
    parser.add_argument('--coco_detection_result', type=str,
                        default='./data/train2014_layouts.json',
                        help='path coco object detection result file')
    parser.add_argument('--log_step', type=int, default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000,
                        help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256,
                        help='dimension of word embedding vectors')
    parser.add_argument('--layout_embed_size', type=int, default=256,
                        help='layout encoding size')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of layers in google transformer')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=123, help='random generator seed')
    args = parser.parse_args()
    print(args)
    main(args)

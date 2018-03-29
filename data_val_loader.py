import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO
import json
import h5py
# import matplotlib.pyplot as plt


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, coco_annotation, vocab, MSCOCO_result, coco_detection_result, yolo,
                 dummy_object, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            coco_annotation: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(coco_annotation)
        self.coco_obj = COCO(MSCOCO_result)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.yolo = yolo
        self.dummy_object = dummy_object

        self.ann_id_2_img_id = {}
        self.img_id_2_ann_id = {}
        self.img_ids = []

        for key in self.coco.anns.keys():
            img_id = self.coco.anns[key]['image_id']
            self.ann_id_2_img_id[key] = img_id
            target = self.img_id_2_ann_id.get(img_id)
            if target:
                target.append(key)
            else:
                self.img_id_2_ann_id[img_id] = [key]
            self.img_ids.append(img_id)

        if self.yolo:
            with open(coco_detection_result, 'r') as f:
                self.detection_results = json.load(f)
            # self.locations = {result['id']: result['bboxes'] for result in self.detection_results}
            # self.labels = {result['id']: result['full_categories'] for result in self.detection_results}

            self.locations = {}
            self.labels = {}
            self.widths = {}
            self.heights = {}
            for key in self.detection_results:
                layouts = self.detection_results[key]
                if layouts['bboxes'] != []:
                    self.locations[key] = layouts['bboxes']
                    self.labels[key] = layouts['categories']
                    self.widths[key] = layouts['width']
                    self.heights[key] = layouts['height']

            train_visual_features = h5py.File('./data/train2014_visual_features.hdf5', 'r')
            self.visual_features = {}
            for key in train_visual_features.keys():
                feature = list(train_visual_features[key])
                if feature != []:
                    self.visual_features[int(key)] = []
                    for arr in feature:
                        self.visual_features[int(key)].append(arr)
        else:
            self.locations = {}
            self.labels = {}
            for key in self.coco_obj.anns.keys():
                img_id = self.coco_obj.anns[key]['image_id']
                if self.labels.get(img_id):
                    self.labels[img_id].append(self.coco_obj.anns[key]['category_id'])
                    self.locations[img_id].append(self.coco_obj.anns[key]['bbox'])
                else:
                    self.labels[img_id] = [self.coco_obj.anns[key]['category_id']]
                    self.locations[img_id] = [self.coco_obj.anns[key]['bbox']]

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        img_id = self.img_ids[index]

        ann_ids = self.img_id_2_ann_id[img_id]
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # plt.imshow(image.permute(1,2,0).numpy())
        # plt.show()
        # Convert caption (string) to word ids.
        captions = []
        for ann_id in ann_ids:
            caption = coco.anns[ann_id]['caption']
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            if tokens[-1] == ".":
                tokens = tokens[:-1]
            captions.append(tokens)

        labels = self.labels.get(img_id)
        if labels is None:
            labels = []
            locations = []
            visuals = []
        else:
            if self.yolo:
                # locations = self.locations[img_id]
                locations = encode_location(self.locations[img_id], self.widths[img_id], self.heights[img_id])
                visuals = self.visual_features[img_id]
            else:
                details = self.coco_obj.loadImgs(img_id)[0]
                locations = encode_location(self.locations[img_id],
                                            details['width'], details['height'])
                visuals = []
        if len(labels) != len(locations):
            raise ValueError("number of labels nust be equal to number of locations")
        if len(labels) == 0:
            labels = [self.dummy_object]
            locations = encode_location([(0, 0, 100, 100)], 100, 100)
            visuals = []

        return image, captions, labels, locations, visuals

    def __len__(self):
        return len(self.img_ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, label_seqs, location_seqs, visual_seqs = zip(*data)
    assert len(label_seqs) > 0
    assert len(label_seqs) == len(location_seqs)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    label_seq_lengths = [len(label_seq) for label_seq in label_seqs]
    label_seq_data = torch.zeros(len(label_seqs), max(label_seq_lengths)).long()
    for i, label_seq in enumerate(label_seqs):
        label_seq_data[i, :len(label_seq)] = torch.LongTensor(label_seq[:len(label_seq)])

    location_seq_data = torch.zeros(len(location_seqs), max(label_seq_lengths), 4)
    for i, location_seq in enumerate(location_seqs):
        for j in range(len(location_seq)):
            coords = decode_location(location_seq[j])
            location_seq_data[i, j] = coords

    visual_seq_data = torch.zeros(len(visual_seqs), max(label_seq_lengths), 1024)
    for i, visual_seq in enumerate(visual_seqs):
        for j in range(len(visual_seq)):
            visual_seq_data[i, j] = torch.Tensor(visual_seq[j])

    # TODO visualize detection results on images
    return images, captions, label_seq_data, location_seq_data, visual_seq_data, label_seq_lengths


def encode_location(bboxs, img_w, img_h):
    locations = []
    for bbox in bboxs:
        x, y, w, h = bbox
        w_ratio = 608 / float(img_w)
        h_ratio = 608 / float(img_h)
        x = int(x * w_ratio)
        w = int(w * w_ratio)
        y = int(y * h_ratio)
        h = int(h * h_ratio)
        locations.append(x * 1e9 + y * 1e6 + w * 1e3 + h)
    return locations

def decode_location(location):
    x = location // 1e9
    y = (location % 1e9) // 1e6
    width = (location % 1e6) // 1e3
    height = location % 1e3
    return torch.Tensor((x / 608, y / 608, width / 608, height / 608))


def get_loader(root, coco_annotation, vocab,
               MSCOCO_result, coco_detection_result,
               transform, batch_size,
               shuffle, num_workers,
               dummy_object,
               yolo):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset

    coco = CocoDataset(root=root,
                       coco_annotation=coco_annotation,
                       vocab=vocab,
                       MSCOCO_result= MSCOCO_result,
                       coco_detection_result=coco_detection_result,
                       yolo=yolo,
                       dummy_object=dummy_object,
                       transform=transform
                      )

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

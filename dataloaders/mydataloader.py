from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random
from dataloaders.rawvideo_util2 import RawVideoExtractor

class My_DataLoader(Dataset):
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.framestart_name = 'frame1'
        self.frameend_name = 'frame' + str(max_frames)
        self.emostart_name = 'emo1'
        self.emoend_name = 'emo' + str(max_frames)
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids,startframe = 0,endframe = 12):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        video_path = os.path.join(self.features_path, "{}.mp4".format(choice_video_ids[0]))
        if os.path.exists(video_path) is False:
            video_path = video_path.replace(".mp4", ".webm")

        raw_video_data = self.rawVideoExtractor.get_video_data(video_path,start_time = startframe, end_time = endframe,frame_num = self.max_frames)
        video = raw_video_data['video']
        video = self.rawVideoExtractor.process_raw_data(video)
        max_video_length[0] = self.max_frames
        video = video.unsqueeze(0)
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['caption'].values[idx]
        action_label = self.data.loc[idx,"action"]
        text_label = -1
        if "text_label" in self.data.columns:
            text_label = self.data.loc[idx,"text_label"]
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids,startframe = int(self.data['start_frame'].values[idx]),endframe= int(self.data['end_frame'].values[idx]))
        return pairs_text, pairs_mask, pairs_segment, video, video_mask,torch.tensor([int(action_label)]),torch.tensor([int(text_label)])

class My_TrainDataLoader(Dataset):
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
    ):

        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.framestart_name = 'frame1'
        self.frameend_name = 'frame' + str(max_frames)
        self.emostart_name = 'emo1'
        self.emoend_name = 'emo' + str(max_frames)
        self.tokenizer = tokenizer
        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids,startframe = 0,endframe = 12):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float)

        video_path = os.path.join(self.features_path, "{}.mp4".format(choice_video_ids[0]))
        if os.path.exists(video_path) is False:
            video_path = video_path.replace(".mp4", ".webm")
        raw_video_data = self.rawVideoExtractor.get_video_data(video_path,start_time = startframe, end_time = endframe,frame_num = self.max_frames)
        video = raw_video_data['video']
        video = self.rawVideoExtractor.process_raw_data(video)
        max_video_length[0] = self.max_frames
        video = video.unsqueeze(0)
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['caption'].values[idx]

        frame_label = self.data.loc[idx,self.framestart_name:self.frameend_name].values
        scene_label = self.data.loc[idx,self.emostart_name:self.emoend_name].values
        attr_label = self.data.loc[idx,"attr_label1":"attr_label3"].values
        action_label = self.data.loc[idx,"action"]
        text_to_video = 0
        if "random_y" in self.data.columns:
            text_to_video = self.data.loc[idx,"random_y"]
        text_label = -1
        if "text_label" in self.data.columns:
            text_label = self.data.loc[idx, "text_label"]
        assert text_label >= 0

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids,startframe = int(self.data['start_frame'].values[idx]),endframe= int(self.data['end_frame'].values[idx]))
        return pairs_text, pairs_mask, pairs_segment, video, video_mask,(torch.tensor(list(frame_label)),
                torch.tensor(list(scene_label)),torch.tensor([int(attr_label[0])]),torch.tensor([int(attr_label[1])]),torch.tensor([int(attr_label[2])]),torch.tensor([int(action_label)])),torch.tensor([int(text_to_video)]),torch.tensor([int(text_label)])

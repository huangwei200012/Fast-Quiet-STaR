import torch
import os
import json, dill
import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List

@dataclass
class DataCollatorForLMDataset(object):

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels,labels_kl = tuple([instance[key].unsqueeze(0) for instance in instances] for key in ("input_ids", "labels", "labels_kl"))
        input_ids, labels = tuple([instance[key].unsqueeze(0) for instance in instances] for key in ("input_ids", "labels_kl"))
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        # labels_kl = torch.cat(labels_kl, dim=0)
        eos_indices = input_ids.argmin(dim=1) - 1
        max_position = eos_indices.max()
        if max_position < 0:
            return dict(
                input_ids=input_ids,
                labels=labels,
                # labels_kl=labels_kl,
            )
        return dict(
            input_ids=input_ids[:, :max_position+1],
            labels=labels[:, :max_position+1],
            # labels_kl=labels_kl[:, :max_position+1]
        )


def make_supervised_data_module(data_args) -> Dict:

    train_dataset = LMDataset(data_args)
    data_collator = DataCollatorForLMDataset()

    return dict(train_dataset=train_dataset, data_collator=data_collator)

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.filepath = filepath
        self.input_ids_file = os.path.join(filepath, 'input_ids')
        self.labels_file = os.path.join(filepath, 'labels')
        self.labels_kl_file = os.path.join(filepath, 'kl_labels')
        # self.input_ids_shape = np.load(self.input_ids_file, mmap_mode='r').shape
        # self.labels_shape = np.load(self.labels_file, mmap_mode='r').shape
        # self.labels_kl_shape = np.load(self.labels_kl_file, mmap_mode='r').shape

    def __getitem__(self, idx):

        input_ids = np.load(self.input_ids_file+"/input_ids_{}.npy".format(idx))[0,:]
        labels = np.load(self.labels_file+"/label_{}.npy".format(idx))[0,:]
        ones_array = np.ones_like(input_ids)
        labels_kl = np.load(self.labels_kl_file+"/logits_{}.npy".format(idx))[0,:]
        return {
            'input_ids': torch.tensor(input_ids),
            'labels': torch.tensor(labels),
            'attention_mask': torch.tensor(ones_array),
            'labels_kl': torch.tensor(labels_kl),
        }

    def __len__(self):
        return 8000




# train_dataset = LMSortDataset(data_args.train_file)


class LMPackDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.input_ids, self.attention_masks, self.labels, self.weights, self.nums = self.process_data(filepath)
        self.num_gpus = torch.cuda.device_count()
        
    def process_data(self, filepath):
        input_ids = torch.from_numpy(np.load(os.path.join(filepath, 'inputs_pack.npy')))
        labels = torch.from_numpy(np.load(os.path.join(filepath, 'labels_pack.npy')))
        weights = torch.from_numpy(np.load(os.path.join(filepath, 'weights_pack.npy')))
        attention_masks = json.load(open(os.path.join(filepath, 'attention_masks_pack.json')))
        num_gpus = torch.cuda.device_count()
        l = (input_ids.size(0) // num_gpus) * num_gpus
        input_ids, labels, weights, attention_masks = input_ids[:l, :], labels[:l, :], weights[:l, :], attention_masks[:l]
        nums = [weights[i*num_gpus:(i+1)*num_gpus, :].sum() for i in range(l//num_gpus)]
        return input_ids, attention_masks, labels, weights, nums

    def __getitem__(self, idx):
        if idx < 32: # reduce GPU memory usage during first few steps
            max_length_tmp = 32768
            attention_mask_tmp = []
            for pos in self.attention_masks[idx]:
                if pos < max_length_tmp:
                    attention_mask_tmp.append(pos)
            attention_mask_tmp.append(max_length_tmp)
            return {
                'input_ids': self.input_ids[idx, :max_length_tmp],
                'attention_mask': torch.tensor(attention_mask_tmp, dtype=torch.int32),
                'labels': (self.labels[idx, :max_length_tmp], self.weights[idx, :max_length_tmp]*2, self.nums[idx//self.num_gpus])
            }
        else:
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.int32),
                'labels': (self.labels[idx], self.weights[idx], self.nums[idx//self.num_gpus])
            }

    def __len__(self):
        return self.input_ids.size(0)

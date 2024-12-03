
import os
import torch
from torch.nn.utils.rnn import pad_sequence

def dynamic_padding(data):
    inputs = [item["input_ids"] for item in data]
    labels = [item["label"] for item in data]
    label_idxs = [item["label_idx"] for item in data]
    fpaths = [item["fpath"] for item in data]
    lengths = [item["length"] for item in data]
    review_lens = [item["review_len"] for item in data]
    inputs_padded = pad_sequence(inputs, batch_first=True,padding_value=0)
    labels = torch.tensor(labels,dtype=torch.float)
    review_lens = torch.tensor(review_lens)
    label_idxs = torch.tensor(label_idxs,dtype=torch.long)
    lengths = torch.tensor(lengths)
    return {"input_ids": inputs_padded,"review_lens":review_lens, "labels":labels, "fpaths": fpaths, "lengths": lengths, "label_idxs": label_idxs}

def start_recording(fname):
    # Start logging GPU memory usage
    os.system(f"""(while true; do echo "$(date +%Y-%m-%d\\ %H:%M:%S), $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> {fname}; sleep 1; done) &""")

def stop_recording():
    os.system("pkill -f 'nvidia-smi --query-gpu=memory.used'")
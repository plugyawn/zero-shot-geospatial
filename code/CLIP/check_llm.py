import torch, clip, csv, json

import numpy as np
import pandas as pd
import rasterio as rio
import torch.nn.functional as F

from rasterio.enums import Resampling
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import HammingDistance, Accuracy
import rasterio as rio
import transformers
from PIL import Image
import sentencepiece
import pdb



from tqdm import tqdm



device = "cuda:0"  # replace "cpu" with "cuda" to use your GPU

tokenizer = transformers.LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
model = transformers.LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").to(device)
classes = ["park", "desert", "parking lot", "open space"]

with torch.no_grad():
    batch = tokenizer(
    text = f"""
    I went to a book shop and got myself a...
    """,
    return_tensors="pt", 
    add_special_tokens=False
    )

    batch = {k: v.to(device) for k, v in batch.items()}
    # Generate output from the model
    output = model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        max_new_tokens = 20,
    )

    # Decode the output to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text)


    # Code translated from download.sh
    PRESIGNED_URL="" # replace with presigned url from email
    TARGET_FOLDER="./model/vi"
    model_sizes = ["7B"]
    N_SHARD_DICT = {
        "7B": 0
    }
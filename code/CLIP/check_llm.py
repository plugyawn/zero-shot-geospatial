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
model = transformers.LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").to(device, dtype=torch.float16)

classes = ["park", "desert", "parking lot", "open space"]


batch = tokenizer(
f"""
[park, desert]: "there is a park in the desert"
[parking lot, open space]: "there is a parking lot next to the open space"
[{classes}]: ?
""",
return_tensors="pt", 
add_special_tokens=False
)

batch = {k: v.to(device) for k, v in batch.items()}

generated = model.generate()

print(generated)



# Code translated from download.sh
PRESIGNED_URL="" # replace with presigned url from email
TARGET_FOLDER="./model/vi"
model_sizes = ["7B"]
N_SHARD_DICT = {
    "7B": 0
}
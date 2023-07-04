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
root_dir = "geolibs/data/BigEarthNet-S2"
bsize, psize = 512, 336
num_workers = 8

model_name = "ViT-L/14@336px"
df = pd.read_csv("/mnt/NVME2/geolibs/data/BigEarthNet-S2/vectors/random-split-6_2022_12_30-01_32_22/CSV/val.csv")
df.head()
def normalize(image):
    means = [0.48145, 0.45782, 0.40821]
    stds = [0.26863, 0.26130, 0.27578]

    image = image / 255
    for idx in range(3):
        image[idx,:,:] = (image[idx,:,:]-means[idx])/stds[idx]

    return image
class BigEarthNetS2InferenceDataset(Dataset):
    def __init__(self, root_dir):
        root_vec_dir = "/mnt/NVME2/geolibs/data/BigEarthNet-S2/vectors/random-split-6_2022_12_30-01_32_22/CSV"
        meta_file = "/mnt/NVME2/geolibs/data/BigEarthNet-S2/vectors/random-split-6_2022_12_30-01_32_22/CSV/metadata.json"
        rasters_root_dir = "/home/progyan/data/cogs/"
        with open(meta_file) as out:
            task_meta = json.load(out)

        classes = [lbl_meta["options"] for lbl_meta in task_meta["label:metadata"]][0]
        cls_idx_map = {cls: idx for idx, cls in enumerate(classes)}
        
        vec_df = pd.read_csv(f"{root_vec_dir}/test.csv")
        vec_df["image"] = vec_df["image:01"].apply(lambda x: f'{rasters_root_dir}{x.split("/")[-1]}')
        vec_df["image"] = vec_df["image"].apply(lambda x: f'{x.split(".")[0]}.tif')
        vec_df["label"] = vec_df["labels"].apply(lambda x: [cls_idx_map[l] for l in x.split("\t")])
        vec_df["label"] = vec_df["label"].apply(lambda x: [1 if i in x else 0 for i in range(len(classes))])
        vec_df.drop(['image-id','image:01','date:01','type','geometry','labels'],axis=1,inplace=True)

        self.vec_df = vec_df
        self.classes = classes

    def __len__(self):
        return len(self.vec_df)

    def __getitem__(self, idx):
        df_entry = self.vec_df.loc[idx]
        image = rio.open(df_entry["image"]).read(out_shape=[3,psize,psize], resampling=Resampling.bilinear)    
        smpl_map = {
            "image": normalize(image),
            "label": np.array(df_entry["label"], dtype=np.int16)
        }

        return smpl_map
                    
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(model_name, device=device)
device
templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]
context_templates = [
    "an overhead view of a {}",
    "a satellite view of a {}",
    "an aerial view of a {}",
    "a clear overhead view of a {}",
    "a blurry overhead view of a {}",
    "a low resolution overhead view of a {}",
    "a high resolution overhead view of a {}",
    "a clear satellite view of a {}",
    "a blurry satellite view of a {}",
    "a low resolution satellite view of a {}",
    "a high resolution satellite view of a {}",
    "a clear satellite view of a {}",
    "a blurry satellite view of a {}",
    "a low resolution satellite view of a {}",
    "a high resolution satellite view of a {}",
    "an overhead image of a {}",
    "a satellite image of a {}",
    "an aerial image of a {}",
    "a clear overhead image of a {}",
    "a blurry overhead image of a {}",
    "a low resolution overhead image of a {}",
    "a high resolution overhead image of a {}",
    "a clear satellite image of a {}",
    "a blurry satellite image of a {}",
    "a low resolution satellite image of a {}",
    "a high resolution satellite image of a {}",
    "a clear satellite image of a {}",
    "a blurry satellite image of a {}",
    "a low resolution satellite image of a {}",
    "a high resolution satellite image of a {}",  
]
def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights
root_dir
bens2_ds = BigEarthNetS2InferenceDataset(root_dir=root_dir)

classes = bens2_ds.classes
num_classes = len(classes)

texts = [f"an image of {cls}" for cls in bens2_ds.classes]
texts = clip.tokenize(texts).to(device)

text_features = model.encode_text(texts)

acc_inst_01 = Accuracy(task="multilabel", num_labels=num_classes).to(device)
acc_inst_02 = Accuracy(task="multilabel", num_labels=num_classes).to(device)
acc_inst_03 = Accuracy(task="multilabel", num_labels=num_classes).to(device)
acc_inst_04 = Accuracy(task="multilabel", num_labels=num_classes).to(device)
acc_inst_05 = Accuracy(task="multilabel", num_labels=num_classes).to(device)
acc_inst_06 = Accuracy(task="multilabel", num_labels=num_classes).to(device)
acc_inst_07 = Accuracy(task="multilabel", num_labels=num_classes).to(device)

bens2_dl = DataLoader(bens2_ds, batch_size=bsize, shuffle=False, num_workers=num_workers)
# acc_inst = MultilabelAccuracy(num_labels=num_classes).to(device)

# ham_dist_03 = HammingDistance(task="multilabel", num_labels=num_classes).to(device)
# ham_dist_04 = HammingDistance(task="multilabel", num_labels=num_classes).to(device)
# ham_dist_05 = HammingDistance(task="multilabel", num_labels=num_classes).to(device)
# ham_dist_06 = HammingDistance(task="multilabel", num_labels=num_classes).to(device)
# ham_dist_07 = HammingDistance(task="multilabel", num_labels=num_classes).to(device)

with torch.no_grad():
    for sample in tqdm(bens2_dl):
        images = sample["image"].to(device)
        target = sample["label"].to(device)
        
        im_features = model.encode_image(images)

        logits_per_image, logits_per_text = model(images, texts)
        
        preds_01 = torch.topk(logits_per_image,1).indices
        preds_02 = torch.topk(logits_per_image,2).indices
        preds_03 = torch.topk(logits_per_image,3).indices
        preds_04 = torch.topk(logits_per_image,4).indices
        preds_05 = torch.topk(logits_per_image,5).indices
        preds_06 = torch.topk(logits_per_image,6).indices
        preds_07 = torch.topk(logits_per_image,7).indices

        oh_preds_01 = F.one_hot(preds_01, num_classes).squeeze(1)
        oh_preds_02 = torch.sum(F.one_hot(preds_02, num_classes),1)
        oh_preds_03 = torch.sum(F.one_hot(preds_03, num_classes),1)
        oh_preds_04 = torch.sum(F.one_hot(preds_04, num_classes),1)
        oh_preds_05 = torch.sum(F.one_hot(preds_05, num_classes),1)
        oh_preds_06 = torch.sum(F.one_hot(preds_06, num_classes),1)
        oh_preds_07 = torch.sum(F.one_hot(preds_07, num_classes),1)

        acc_inf_01 = acc_inst_01(oh_preds_01, target)
        acc_inf_02 = acc_inst_02(oh_preds_02, target)
        acc_inf_03 = acc_inst_03(oh_preds_03, target)
        acc_inf_04 = acc_inst_04(oh_preds_04, target)
        acc_inf_05 = acc_inst_05(oh_preds_05, target)
        acc_inf_06 = acc_inst_06(oh_preds_06, target)
        acc_inf_07 = acc_inst_07(oh_preds_07, target)

        # ham_dist_inf_03 = ham_dist_03(oh_preds_03, target)
        # ham_dist_inf_04 = ham_dist_04(oh_preds_04, target)
        # ham_dist_inf_05 = ham_dist_05(oh_preds_05, target)
        # ham_dist_inf_06 = ham_dist_06(oh_preds_06, target)
        # ham_dist_inf_07 = ham_dist_07(oh_preds_07, target)
        #acc_inf= acc_inst(oh_preds, target)

    avg_acc_inst_01 = acc_inst_01.compute()
    avg_acc_inst_02 = acc_inst_02.compute()
    avg_acc_inst_03 = acc_inst_03.compute()
    avg_acc_inst_04 = acc_inst_04.compute()
    avg_acc_inst_05 = acc_inst_05.compute()
    avg_acc_inst_06 = acc_inst_06.compute()
    avg_acc_inst_07 = acc_inst_07.compute()

    # avg_ham_dist_03 = ham_dist_03.compute()
    # avg_ham_dist_04 = ham_dist_04.compute()
    # avg_ham_dist_05 = ham_dist_05.compute()
    # avg_ham_dist_06 = ham_dist_06.compute()
    # avg_ham_dist_07 = ham_dist_07.compute()
    print(f"top 1: {avg_acc_inst_01}")
    print(f"top 2: {avg_acc_inst_02}")   
    print(f"top 3: {avg_acc_inst_03}")
    print(f"top 4: {avg_acc_inst_04}")
    print(f"top 5: {avg_acc_inst_05}")
    print(f"top 6: {avg_acc_inst_06}")
    print(f"top 7: {avg_acc_inst_07}")

oh_preds_01.shape
import torch.nn.functional as F

target.shape
vec_df = pd.read_csv(f"/mnt/NVME2/{root_dir}/vectors/random-split-6_2022_12_30-01_32_22/CSV/test.csv")
vec_df.head(10)






image = preprocess(Image.open(bens2_ds.vec_df.loc[0]["image"])).unsqueeze(0).to(device)
text = clip.tokenize(bens2_ds.classes).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)

    idx_tp05 = torch.topk(logits_per_image, 5).indices.cpu().tolist()[0]
    idx_tp10 = torch.topk(logits_per_image, 10).indices.cpu().tolist()[0]
    idx_tp15 = torch.topk(logits_per_image, 15).indices.cpu().tolist()[0]
    idx_tp20 = torch.topk(logits_per_image, 20).indices.cpu().tolist()[0]

    cls_tp05 = [bens2_ds.classes[idx] for idx in idx_tp05]
    cls_tp10 = [bens2_ds.classes[idx] for idx in idx_tp10]
    cls_tp15 = [bens2_ds.classes[idx] for idx in idx_tp15]
    cls_tp20 = [bens2_ds.classes[idx] for idx in idx_tp20]

    gr_truth = vec_df.loc[8]['labels'].split('\t')
    
    print(f"Ground truth : {gr_truth}")
    print(f"\nTop 5: {cls_tp05}")
    print(f"\nTop 10: {cls_tp10}")
    print(f"\nTop 15: {cls_tp15}")
    print(f"\nTop 20: {cls_tp20}")

device = "cuda:0"  # replace "cpu" with "cuda" to use your GPU

tokenizer = transformers.LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = transformers.LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf").to(device)

classes = []

batch = tokenizer(
    f"""
    [park, desert]: "there is a park in the desert"
    [parking lot, open space]: "there is a parking lot next to the open space"
    [building, road {classes}]: ?
    """,
    return_tensors="pt", 
    add_special_tokens=False
)

batch = {k: v.to(device) for k, v in batch.items()}


generated = model.generate


print(generated)

zeroshot_weights = zeroshot_classifier(classes, context_templates)
bens2_ds.vec_df.loc[0]["image"]

bens2_dl = DataLoader(bens2_ds, batch_size=bsize, shuffle=False, num_workers=num_workers)
acc_inst = MultilabelAccuracy(num_labels=num_classes).to(device)

with torch.no_grad():
    for sample in tqdm(bens2_dl):
        images = sample["image"].to(device)
        target = sample["label"].to(device)

        im_features = model.encode_image(images)
        im_features /= im_features.norm(dim=-1, keepdim=True)
        logits = 1.* im_features @ zeroshot_weights
        preds = torch.sigmoid(logits)

        pdb.set_trace()

        acc_inf= acc_inst(preds, target)

    avg_acc_inf = acc_inst.compute()
    print(avg_acc_inf)
preds[0]
target[0]
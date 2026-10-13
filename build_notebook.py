"""
build_notebook.py
-----------------
Generates `image_captioning_kaggle.ipynb` from the cell definitions below.

Workflow (see MASTER_PROMPT.md):
    1. Edit the cell strings in this file.
    2. Run:  python build_notebook.py
    3. Upload the generated .ipynb to Kaggle (File -> Import Notebook).

Checks performed on build:
    - every code cell is syntax-checked with `ast` (lines starting with '!' or '%' are ignored)
    - every cell must be pure ASCII (guarantees no emojis / fancy symbols)
    - markdown cells must not contain bold markers or headings (style rule from the user)
"""
import ast
import json
import os

CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# PART 0. INTRODUCTION
md(r'''
Image Captioning with a Vision Transformer Encoder and a Transformer Decoder built from scratch

In this notebook we train a model that looks at an image and writes a short sentence describing it. We use the MS COCO 2017 dataset, which has about 118 thousand training images and 5 thousand validation images, and every image has around five captions written by people.

The model has two parts. The first part is a Vision Transformer (ViT) that was already trained on a very large image dataset. It turns the image into a set of feature vectors, one for each small patch of the image. The second part is a Transformer decoder that we write ourselves. It reads these patch features and produces the caption one word at a time.

Before running, open the notebook settings on Kaggle, choose a GPU (T4 or P100) as the accelerator, and turn the internet option on so the pretrained weights can be downloaded. Then use Add Input and add the dataset called COCO 2017 Dataset by awsaf49. After that the notebook can be run from top to bottom without changing anything.

For a quick test run you can set TRAIN_SUBSET to 5000 and EPOCHS to 1 in the configuration cell.
''')

# PART 1. SETUP
md(r'''
Step 1: Install the libraries that are not already available on Kaggle. pycocotools is used to read the COCO caption files, and the other packages are used to compute the evaluation scores.
''')
code(r'''
!pip install -q pycocotools rouge-score nltk pycocoevalcap
''')

md(r'''
Step 2: Import all the Python libraries used in the notebook.
''')
code(r'''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"   # avoids warnings from the tokenizer inside DataLoader workers

import re, glob, json, math, random, time, warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm.auto import tqdm

from pycocotools.coco import COCO
import transformers
from transformers import AutoTokenizer, ViTModel

import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
print("Imports done")
''')

md(r'''
Step 3: Download the WordNet data used by the METEOR score. If the download fails the notebook will still run, only METEOR will be skipped.
''')
code(r'''
for pkg in ["wordnet", "omw-1.4"]:
    try:
        ok = nltk.download(pkg, quiet=True)
        print(f"NLTK {pkg}: {'ok' if ok else 'failed'}")
    except Exception as e:
        print(f"NLTK {pkg}: could not download ({e})")
''')

md(r'''
Step 4: Fix the random seed so results can be reproduced, and check which GPU we are running on.
''')
code(r'''
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


print(f"PyTorch version      : {torch.__version__}")
print(f"Transformers version : {transformers.__version__}")
print(f"CUDA available       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU                  : {props.name}")
    print(f"GPU memory           : {props.total_memory / 1e9:.1f} GB")
else:
    print("WARNING: no GPU found. Turn on the GPU accelerator in the notebook settings.")
print(f"CPU cores            : {os.cpu_count()}")
''')

# PART 2. CONFIGURATION
md(r'''
Step 5: Find the dataset files. Kaggle can place the COCO dataset in different folders depending on how it was added, so instead of writing the path by hand we search for the caption files inside /kaggle/input.
''')
code(r'''
INPUT_ROOT = "/kaggle/input"


def glob_by_depth(rel_pattern, max_depth=6, want_dir=False):
    # search level by level so we never scan the huge image folders recursively
    for depth in range(max_depth + 1):
        pattern = os.path.join(INPUT_ROOT, *(["*"] * depth), rel_pattern)
        hits = sorted(h for h in glob.glob(pattern) if (os.path.isdir(h) if want_dir else os.path.isfile(h)))
        if hits:
            return hits[0]
    return None


def resolve_image_dir(root, name):
    cand = os.path.join(root, name)
    if not os.path.isdir(cand):
        cand = glob_by_depth(name, want_dir=True)
    if cand is None:
        return None
    nested = os.path.join(cand, name)          # some uploads have train2017/train2017
    return nested if os.path.isdir(nested) else cand


def discover_coco():
    train_ann = glob_by_depth(os.path.join("annotations", "captions_train2017.json"))
    val_ann = glob_by_depth(os.path.join("annotations", "captions_val2017.json"))
    if train_ann is None or val_ann is None:
        raise FileNotFoundError("COCO caption files not found. Add the 'COCO 2017 Dataset' (awsaf49) with Add Input.")
    root = os.path.dirname(os.path.dirname(train_ann))
    paths = {
        "TRAIN_IMG_DIR": resolve_image_dir(root, "train2017"),
        "VAL_IMG_DIR": resolve_image_dir(root, "val2017"),
        "TRAIN_ANN_FILE": train_ann,
        "VAL_ANN_FILE": val_ann,
    }
    for k, v in paths.items():
        if v is None:
            raise FileNotFoundError(f"Could not find {k}")
    return paths
''')

code(r'''
PATHS = discover_coco()
for k, v in PATHS.items():
    print(f"{k:15s}: {v}")
print(f"\nTrain images found: {len(os.listdir(PATHS['TRAIN_IMG_DIR'])):,}")
print(f"Val images found  : {len(os.listdir(PATHS['VAL_IMG_DIR'])):,}")
''')

md(r'''
Step 6: All settings of the project are kept in one class so they are easy to find and change.

About the number of epochs: the image encoder is already pretrained and stays frozen, so only the decoder has to learn. We use 12 epochs together with a learning rate that slowly goes down to zero. This is long enough for the model to converge well and squeeze out maximum performance. Note that 12 epochs might take around 4-6 hours on a Kaggle T4 GPU (roughly 20 to 30 minutes per epoch), so make sure to use "Save Version -> Save & Run All (Commit)" so it runs in the background.
''')
code(r'''
class Config:
    # paths
    TRAIN_IMG_DIR = PATHS["TRAIN_IMG_DIR"]
    VAL_IMG_DIR = PATHS["VAL_IMG_DIR"]
    TRAIN_ANN_FILE = PATHS["TRAIN_ANN_FILE"]
    VAL_ANN_FILE = PATHS["VAL_ANN_FILE"]
    OUTPUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./outputs"

    # pretrained parts
    ENCODER_NAME = "google/vit-base-patch16-224-in21k"   # ViT-B/16, 768 dim, 14x14 patches + CLS token
    TOKENIZER_NAME = "bert-base-uncased"                 # WordPiece vocabulary of about 30k tokens
    IMAGE_SIZE = 224
    IMAGE_MEAN = [0.5, 0.5, 0.5]                         # normalisation used when the ViT was pretrained
    IMAGE_STD = [0.5, 0.5, 0.5]

    # decoder
    EMBED_DIM = 768
    DECODER_LAYERS = 6
    DECODER_HEADS = 8
    DECODER_FF_DIM = 2048
    DROPOUT = 0.1
    MAX_CAPTION_LEN = 40                                 # tokens including start and end token
    FINE_TUNE_ENCODER_LAYERS = 0                         # 0 means the ViT is completely frozen

    # training
    BATCH_SIZE = 64
    EPOCHS = 12
    LEARNING_RATE = 3e-4
    ENCODER_LR = 1e-5                                    # only used if FINE_TUNE_ENCODER_LAYERS > 0
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.05
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP = 1.0
    USE_AMP = True
    NUM_WORKERS = min(4, os.cpu_count() or 2)
    LOG_EVERY = 10                                       # store the training loss every N steps

    # subsets for quick tests (None means use everything)
    TRAIN_SUBSET = None
    VAL_SUBSET = None

    # evaluation
    EVAL_NUM_IMAGES = 5000
    BEAM_EVAL_IMAGES = 500
    BEAM_SIZE = 3
    LENGTH_PENALTY = 0.7
    BEAM_STUDY_SIZES = [1, 2, 3, 5]
    BEAM_STUDY_IMAGES = 200
    NUM_MONITOR_IMAGES = 4
    RESUME = True

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
''')

code(r'''
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
seed_everything(Config.SEED)
DEVICE = Config.DEVICE
AMP_ENABLED = Config.USE_AMP and DEVICE.type == "cuda"

for k, v in vars(Config).items():
    if not k.startswith("_"):
        print(f"{k:26s}: {v}")
''')

# PART 3. DATA EXPLORATION
md(r'''
Step 7: Load the caption annotation files. Each annotation links one caption to one image.
''')
code(r'''
coco_train = COCO(Config.TRAIN_ANN_FILE)
coco_val = COCO(Config.VAL_ANN_FILE)

print(f"\nTrain: {len(coco_train.imgs):,} images, {len(coco_train.anns):,} captions")
print(f"Val  : {len(coco_val.imgs):,} images, {len(coco_val.anns):,} captions")
''')

md(r'''
Step 8: Count how many captions each image has. Most images have exactly five.
''')
code(r'''
caps_per_img = Counter(len(coco_train.imgToAnns[i]) for i in coco_train.imgs)
for n, count in sorted(caps_per_img.items()):
    print(f"{n} captions: {count:,} images")

example_ann = next(iter(coco_train.anns.values()))
print("\nOne raw annotation:", example_ann)
''')

md(r'''
Step 9: Look at how long the captions are, measured in words.
''')
code(r'''
def simple_words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


train_captions = [a["caption"] for a in coco_train.anns.values()]
word_lengths = np.array([len(simple_words(c)) for c in train_captions])

print(f"Mean length  : {word_lengths.mean():.2f} words")
print(f"Median length: {np.median(word_lengths):.0f} words")
print(f"Min / Max    : {word_lengths.min()} / {word_lengths.max()} words")

plt.figure(figsize=(10, 4))
plt.hist(word_lengths, bins=range(0, 35), color="steelblue", edgecolor="white")
plt.axvline(word_lengths.mean(), color="red", linestyle="--", label=f"mean = {word_lengths.mean():.1f}")
plt.xlabel("caption length (words)")
plt.ylabel("number of captions")
plt.title("Caption length distribution (train set)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, "eda_caption_lengths.png"), dpi=120)
plt.show()
''')

md(r'''
Step 10: Find the most common words. The first chart includes all words, the second one leaves out very common small words like "a" and "the" so the content words become visible.
''')
code(r'''
word_counter = Counter(w for c in train_captions for w in simple_words(c))
STOPWORDS = {"a", "an", "the", "on", "of", "in", "with", "and", "is", "to", "at", "are", "it", "its",
             "his", "her", "their", "some", "that", "while", "for", "by", "from", "near", "next", "up", "down", "there"}

print(f"Unique words in training captions: {len(word_counter):,}")
print(f"Words that appear at least 5 times: {sum(1 for c in word_counter.values() if c >= 5):,}")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
top_all = word_counter.most_common(25)
top_content = [(w, c) for w, c in word_counter.most_common(300) if w not in STOPWORDS][:25]
for ax, data, title in [(axes[0], top_all, "Top 25 words"), (axes[1], top_content, "Top 25 content words")]:
    words, counts = zip(*data)
    ax.bar(words, counts, color="teal")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=70)
    ax.set_ylabel("count")
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, "eda_word_frequency.png"), dpi=120)
plt.show()
''')

md(r'''
Step 11: Show a few training images together with all of their human written captions.
''')
code(r'''
def show_images_with_captions(coco, img_dir, n=3):
    ids = random.sample(list(coco.imgs.keys()), n)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    for ax, img_id in zip(axes, ids):
        info = coco.imgs[img_id]
        img = Image.open(os.path.join(img_dir, info["file_name"])).convert("RGB")
        caps = [a["caption"].strip() for a in coco.imgToAnns[img_id]]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"id {img_id}  ({img.width}x{img.height})", fontsize=10)
        ax.text(0, img.height + 10, "\n".join(f"{i+1}. {c}" for i, c in enumerate(caps)),
                fontsize=9, va="top", wrap=True)
    plt.tight_layout()
    plt.show()


show_images_with_captions(coco_train, Config.TRAIN_IMG_DIR)
''')

md(r'''
Step 12: Check the image sizes. COCO images have different sizes, which is why we resize all of them to 224 by 224 pixels before giving them to the ViT.
''')
code(r'''
sample_ids = random.sample(list(coco_train.imgs.keys()), 2000)
widths = np.array([coco_train.imgs[i]["width"] for i in sample_ids])
heights = np.array([coco_train.imgs[i]["height"] for i in sample_ids])

plt.figure(figsize=(6, 5))
plt.scatter(widths, heights, s=5, alpha=0.4)
plt.xlabel("width (pixels)")
plt.ylabel("height (pixels)")
plt.title("Image sizes (2000 random train images)")
plt.tight_layout()
plt.show()
print(f"Mean size: {widths.mean():.0f} x {heights.mean():.0f}")
''')

# PART 4. TOKENIZER
md(r'''
Step 13: Load the tokenizer. A tokenizer splits a sentence into small pieces (tokens) and maps each piece to a number. We use the BERT tokenizer, so we do not need to build our own vocabulary. Its CLS token is used as the start of a caption, SEP as the end, and PAD fills empty positions.
''')
code(r'''
tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_NAME)
BOS_ID = tokenizer.cls_token_id
EOS_ID = tokenizer.sep_token_id
PAD_ID = tokenizer.pad_token_id
VOCAB_SIZE = len(tokenizer)

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Start token: {tokenizer.cls_token} -> {BOS_ID}")
print(f"End token  : {tokenizer.sep_token} -> {EOS_ID}")
print(f"Pad token  : {tokenizer.pad_token} -> {PAD_ID}")
''')

md(r'''
Step 14: See the tokenizer in action on a real caption. Rare words are split into smaller pieces that start with ##.
''')
code(r'''
example_caption = train_captions[0].lower()
enc = tokenizer(example_caption, padding="max_length", truncation=True, max_length=20)

print("Caption :", example_caption)
print("Tokens  :", tokenizer.convert_ids_to_tokens(enc.input_ids))
print("Ids     :", enc.input_ids)
print("Decoded :", tokenizer.decode(enc.input_ids, skip_special_tokens=True))
print("\nSubword example:", tokenizer.tokenize("a skateboarder performing a kickflip"))
''')

md(r'''
Step 15: Measure caption length in tokens to check that our maximum length of 40 tokens is enough. Captions longer than this are cut.
''')
code(r'''
sample_caps = random.sample(train_captions, 20000)
token_lengths = np.array([len(tokenizer.tokenize(c.lower())) + 2 for c in sample_caps])
coverage = (token_lengths <= Config.MAX_CAPTION_LEN).mean() * 100

plt.figure(figsize=(10, 4))
plt.hist(token_lengths, bins=range(0, 50), color="slateblue", edgecolor="white")
plt.axvline(Config.MAX_CAPTION_LEN, color="red", linestyle="--", label=f"max length = {Config.MAX_CAPTION_LEN}")
plt.xlabel("caption length (tokens, including start and end)")
plt.ylabel("count")
plt.title("Token length distribution (20k random captions)")
plt.legend()
plt.tight_layout()
plt.show()
print(f"{coverage:.2f}% of captions fit completely within {Config.MAX_CAPTION_LEN} tokens")
''')

# PART 5. DATASET AND DATALOADER
md(r'''
Step 16: Define the image transforms. For training we add small random crops and small colour changes so the model sees slightly different versions of each image (data augmentation). We do not flip images, because flipping would make captions with "left" or "right" wrong. For validation we only resize and normalise.
''')
code(r'''
train_transform = T.Compose([
    T.RandomResizedCrop(Config.IMAGE_SIZE, scale=(0.85, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.ColorJitter(0.1, 0.1, 0.1),
    T.ToTensor(),
    T.Normalize(Config.IMAGE_MEAN, Config.IMAGE_STD),
])
eval_transform = T.Compose([
    T.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(Config.IMAGE_MEAN, Config.IMAGE_STD),
])


def denormalize(t):
    mean = torch.tensor(Config.IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(Config.IMAGE_STD).view(3, 1, 1)
    return (t.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
''')

md(r'''
Step 17: Show what the transforms do to one image: the original, the validation version, and three random training versions.
''')
code(r'''
demo_id = random.choice(list(coco_train.imgs.keys()))
demo_img = Image.open(os.path.join(Config.TRAIN_IMG_DIR, coco_train.imgs[demo_id]["file_name"])).convert("RGB")

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
axes[0].imshow(demo_img); axes[0].set_title(f"original {demo_img.width}x{demo_img.height}")
axes[1].imshow(denormalize(eval_transform(demo_img))); axes[1].set_title("validation transform 224x224")
for i in range(3):
    axes[2 + i].imshow(denormalize(train_transform(demo_img))); axes[2 + i].set_title(f"train augmentation {i+1}")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
''')

md(r'''
Step 18: Define the dataset classes. In training, one sample is one image together with one of its captions chosen at random. This means one epoch goes over every image once, and over several epochs the model sees different captions of the same image. The second class returns only images and is used later when we generate captions for evaluation.
''')
code(r'''
class CocoCaptionDataset(Dataset):
    def __init__(self, coco, img_dir, tokenizer, transform, max_len, train=True, subset=None):
        self.coco = coco
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.train = train
        ids = sorted(i for i in coco.imgs.keys() if len(coco.imgToAnns[i]) > 0)
        self.img_ids = ids[:subset] if subset else ids
        self.captions = {i: [a["caption"].strip() for a in coco.imgToAnns[i]] for i in self.img_ids}

    def __len__(self):
        return len(self.img_ids)

    def image_path(self, img_id):
        return os.path.join(self.img_dir, self.coco.imgs[img_id]["file_name"])

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = Image.open(self.image_path(img_id)).convert("RGB")
        pixel_values = self.transform(image)
        caps = self.captions[img_id]
        caption = random.choice(caps) if self.train else caps[0]
        ids = self.tokenizer(caption.lower(), padding="max_length", truncation=True,
                             max_length=self.max_len, return_tensors="pt").input_ids.squeeze(0)
        return pixel_values, ids, img_id


class CocoImageDataset(Dataset):
    def __init__(self, caption_ds, transform, img_ids=None):
        self.ds = caption_ds
        self.transform = transform
        self.img_ids = img_ids if img_ids is not None else caption_ds.img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = Image.open(self.ds.image_path(img_id)).convert("RGB")
        return self.transform(image), img_id
''')

md(r'''
Step 19: Create the datasets and the data loaders. The data loader groups samples into batches and loads images in parallel on the CPU while the GPU is training.
''')
code(r'''
def make_loader(ds, shuffle, drop_last=False, batch_size=Config.BATCH_SIZE):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                      num_workers=Config.NUM_WORKERS, pin_memory=True,
                      persistent_workers=Config.NUM_WORKERS > 0)


train_ds = CocoCaptionDataset(coco_train, Config.TRAIN_IMG_DIR, tokenizer, train_transform,
                              Config.MAX_CAPTION_LEN, train=True, subset=Config.TRAIN_SUBSET)
val_ds = CocoCaptionDataset(coco_val, Config.VAL_IMG_DIR, tokenizer, eval_transform,
                            Config.MAX_CAPTION_LEN, train=False, subset=Config.VAL_SUBSET)
train_loader = make_loader(train_ds, shuffle=True, drop_last=True)
val_loader = make_loader(val_ds, shuffle=False)

print(f"Train samples per epoch: {len(train_ds):,}  ->  {len(train_loader):,} batches")
print(f"Val samples            : {len(val_ds):,}  ->  {len(val_loader):,} batches")
''')

md(r'''
Step 20: Take one batch from the training loader and inspect it, to make sure the shapes and values are what we expect.
''')
code(r'''
t0 = time.time()
batch_pixels, batch_ids, batch_img_ids = next(iter(train_loader))
print(f"Time to load first batch: {time.time() - t0:.1f} s")
print(f"Image tensor shape : {tuple(batch_pixels.shape)}  (batch, channels, height, width)")
print(f"Image value range  : {batch_pixels.min():.2f} to {batch_pixels.max():.2f}")
print(f"Token tensor shape : {tuple(batch_ids.shape)}  (batch, max_len)")
print(f"Image ids (first 5): {batch_img_ids[:5].tolist()}")
print("\nFirst sample token ids:", batch_ids[0].tolist())
print("First sample tokens   :", tokenizer.convert_ids_to_tokens(batch_ids[0][batch_ids[0] != PAD_ID]))
''')

code(r'''
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(denormalize(batch_pixels[i]))
    ax.set_title(tokenizer.decode(batch_ids[i], skip_special_tokens=True), fontsize=9, wrap=True)
    ax.axis("off")
plt.suptitle("One training batch after augmentation (with the randomly chosen caption)")
plt.tight_layout()
plt.show()
''')

# PART 6. MODEL - ENCODER
md(r'''
Step 21: Positional encoding. A Transformer has no built in sense of word order, so we add a fixed pattern of sine and cosine waves to each word embedding. Every position gets a different pattern.
''')
code(r'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])
''')

md(r'''
Step 22: Visualise the positional encoding for the first 40 positions. Each row is one position and each column is one dimension of the embedding.
''')
code(r'''
pe_demo = PositionalEncoding(Config.EMBED_DIM, 64, 0.0).pe[0, :Config.MAX_CAPTION_LEN].numpy()
fig, axes = plt.subplots(1, 2, figsize=(18, 4))
im = axes[0].imshow(pe_demo[:, :128], aspect="auto", cmap="RdBu")
axes[0].set_xlabel("embedding dimension (first 128)"); axes[0].set_ylabel("position")
axes[0].set_title("Sinusoidal positional encoding")
plt.colorbar(im, ax=axes[0])
for d in [0, 2, 10, 40]:
    axes[1].plot(pe_demo[:, d], label=f"dim {d}")
axes[1].set_xlabel("position"); axes[1].set_title("A few dimensions as functions of position"); axes[1].legend()
plt.tight_layout()
plt.show()
''')

md(r'''
Step 23: The image encoder. We load a ViT that was pretrained on ImageNet 21k. It splits the 224 by 224 image into 196 patches of 16 by 16 pixels, adds one extra CLS token, and returns a 768 dimensional vector for each of these 197 positions. By default we freeze it, which saves memory and time and keeps the good pretrained features.
''')
code(r'''
class ViTEncoder(nn.Module):
    def __init__(self, name, fine_tune_layers=0):
        super().__init__()
        self.vit = ViTModel.from_pretrained(name, add_pooling_layer=False)
        self.hidden_size = self.vit.config.hidden_size
        self.fine_tune = fine_tune_layers > 0
        for p in self.vit.parameters():
            p.requires_grad = False
        if self.fine_tune:
            for block in self.vit.encoder.layer[-fine_tune_layers:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.vit.layernorm.parameters():
                p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if not self.fine_tune:
            self.vit.eval()        # a frozen encoder always stays in eval mode
        return self

    def forward(self, pixel_values):
        if self.fine_tune:
            return self.vit(pixel_values=pixel_values).last_hidden_state
        with torch.no_grad():
            return self.vit(pixel_values=pixel_values).last_hidden_state
''')

md(r'''
Step 24: Show how the ViT cuts an image into a 14 by 14 grid of patches.
''')
code(r'''
vis_ids = random.sample(val_ds.img_ids, 3)
vis_imgs = [Image.open(val_ds.image_path(i)).convert("RGB").resize((224, 224)) for i in vis_ids]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, img in zip(axes, vis_imgs):
    ax.imshow(img)
    for k in range(0, 225, 16):
        ax.axhline(k, color="white", linewidth=0.6)
        ax.axvline(k, color="white", linewidth=0.6)
    ax.set_title("14 x 14 grid of 16 x 16 pixel patches")
    ax.axis("off")
plt.tight_layout()
plt.show()
''')

md(r'''
Step 25: Run the pretrained ViT on these images and look at its output. We print the output shape, and then show two views of what the encoder has learned. The first view reduces the 768 numbers of each patch to 3 numbers with PCA and shows them as colours, so patches with similar features get similar colours. The second view shows where the CLS token of the last ViT layer looks, which often highlights the main object.
''')
code(r'''
try:
    vit_viz = ViTModel.from_pretrained(Config.ENCODER_NAME, add_pooling_layer=False,
                                       attn_implementation="eager").to(DEVICE).eval()
    px = torch.stack([eval_transform(img) for img in vis_imgs]).to(DEVICE)
    with torch.no_grad():
        out = vit_viz(pixel_values=px, output_attentions=True)
    feats = out.last_hidden_state                       # (3, 197, 768)
    cls_attn = out.attentions[-1][:, :, 0, 1:].mean(1)  # last layer, CLS -> patches, mean over heads
    print(f"Encoder output shape: {tuple(feats.shape)}  (images, 1 CLS + 196 patches, feature size)")
    print(f"Number of attention layers: {len(out.attentions)}, heads per layer: {out.attentions[-1].shape[1]}")

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for r, img in enumerate(vis_imgs):
        f = feats[r, 1:].float().cpu().numpy()
        f = f - f.mean(0)
        _, _, vt = np.linalg.svd(f, full_matrices=False)
        proj = f @ vt[:3].T
        proj = (proj - proj.min(0)) / (proj.max(0) - proj.min(0) + 1e-8)
        a = cls_attn[r].reshape(14, 14).cpu().numpy()
        axes[r, 0].imshow(img); axes[r, 0].set_title("input")
        axes[r, 1].imshow(proj.reshape(14, 14, 3), interpolation="nearest"); axes[r, 1].set_title("PCA of patch features")
        axes[r, 2].imshow(img); axes[r, 2].imshow(np.array(Image.fromarray(a).resize((224, 224), Image.BICUBIC)),
                                                  cmap="jet", alpha=0.5)
        axes[r, 2].set_title("CLS attention, last ViT layer")
        for ax in axes[r]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "encoder_features.png"), dpi=120)
    plt.show()
    del vit_viz, out, feats
    torch.cuda.empty_cache()
except Exception as e:
    print(f"Encoder visualisation skipped: {e}")
''')

# PART 7. MODEL - DECODER
md(r'''
Step 26: One decoder layer, written from scratch. It has three parts. First, masked self attention, where each word looks at the words before it. Second, cross attention, where each word looks at the image patches. Third, a small feed forward network. We use layer normalisation before each part (pre-LN), which makes training more stable. The important detail is that this layer can return the cross attention weights, which we use later to draw heatmaps of where the model looks for each word.
''')
code(r'''
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask=None, key_padding_mask=None, need_weights=False):
        # 1) masked self attention between words
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=causal_mask, key_padding_mask=key_padding_mask,
                              need_weights=False)
        x = x + self.drop1(h)
        # 2) cross attention from words to image patches
        h = self.norm2(x)
        h, attn = self.cross_attn(h, memory, memory, need_weights=need_weights, average_attn_weights=False)
        x = x + self.drop2(h)
        # 3) feed forward network
        x = x + self.drop3(self.ffn(self.norm3(x)))
        return x, attn   # attn has shape (batch, heads, words, patches) or is None
''')

md(r'''
Step 27: The causal mask. While training, the decoder sees the whole caption at once, so we must stop each word from looking at the words that come after it. The mask below shows this: a yellow cell means the word in that row is not allowed to look at the word in that column.
''')
code(r'''
def causal_mask(sz, device):
    # True means blocked (a future position)
    return torch.triu(torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1)


demo_tokens = tokenizer.convert_ids_to_tokens(tokenizer("a dog runs on the beach").input_ids)
m = causal_mask(len(demo_tokens), "cpu").numpy()
plt.figure(figsize=(6, 5))
plt.imshow(m, cmap="viridis")
plt.xticks(range(len(demo_tokens)), demo_tokens, rotation=60)
plt.yticks(range(len(demo_tokens)), demo_tokens)
plt.title("Causal mask (yellow = blocked)")
plt.tight_layout()
plt.show()
''')

md(r'''
Step 28: The full decoder. Each token id is turned into a vector (embedding), the positional encoding is added, the result goes through 6 decoder layers, and a final linear layer gives a score for every word in the vocabulary. The output layer shares its weights with the input embedding (weight tying), which reduces the number of parameters and usually helps generalisation.
''')
code(r'''
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, ff_dim, dropout, max_len, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()
        self.pos_encoder = PositionalEncoding(d_model, max_len + 8, dropout)
        self.layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.fc_out.weight = self.embedding.weight          # weight tying

    def forward(self, tgt_ids, memory, key_padding_mask=None, return_attn=False):
        x = self.pos_encoder(self.embedding(tgt_ids) * math.sqrt(self.d_model))
        mask = causal_mask(tgt_ids.size(1), tgt_ids.device)
        attns = []
        for layer in self.layers:
            x, attn = layer(x, memory, causal_mask=mask, key_padding_mask=key_padding_mask,
                            need_weights=return_attn)
            if return_attn:
                attns.append(attn)
        # scale by 1/sqrt(d) before the tied output layer (as in T5) so the first predictions are close to uniform
        logits = self.fc_out(self.final_norm(x) * self.d_model ** -0.5)
        return logits, (attns if return_attn else None)
''')

md(r'''
Step 29: Put encoder and decoder together. A small linear layer with layer normalisation sits between them and adapts the ViT features for the decoder.
''')
code(r'''
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, pad_id, cfg=Config):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = ViTEncoder(cfg.ENCODER_NAME, cfg.FINE_TUNE_ENCODER_LAYERS)
        self.enc_proj = nn.Sequential(nn.Linear(self.encoder.hidden_size, cfg.EMBED_DIM),
                                      nn.LayerNorm(cfg.EMBED_DIM))
        self.decoder = TransformerDecoder(vocab_size, cfg.EMBED_DIM, cfg.DECODER_LAYERS, cfg.DECODER_HEADS,
                                          cfg.DECODER_FF_DIM, cfg.DROPOUT, cfg.MAX_CAPTION_LEN, pad_id)

    def encode(self, pixel_values):
        return self.enc_proj(self.encoder(pixel_values))

    def forward(self, pixel_values, tgt_ids):
        memory = self.encode(pixel_values)
        logits, _ = self.decoder(tgt_ids, memory, key_padding_mask=(tgt_ids == self.pad_id))
        return logits

    @torch.no_grad()
    def forward_with_attention(self, pixel_values, tgt_ids):
        memory = self.encode(pixel_values)
        return self.decoder(tgt_ids, memory, return_attn=True)
''')

md(r'''
Step 30: Build the model and count its parameters. Only the decoder part is trained because the ViT is frozen.
''')
code(r'''
model = ImageCaptioningModel(VOCAB_SIZE, PAD_ID).to(DEVICE)


def count(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


print(f"{'part':28s} {'parameters':>14s} {'trainable':>14s}")
for name, module in [("ViT encoder", model.encoder), ("projection", model.enc_proj),
                     ("decoder embedding (tied)", model.decoder.embedding),
                     ("decoder layers", model.decoder.layers), ("whole model", model)]:
    print(f"{name:28s} {count(module) / 1e6:12.2f} M {count(module, True) / 1e6:12.2f} M")
''')

md(r'''
Step 31: A quick sanity check with one batch. Before training, the model should guess randomly, so the loss should be close to the natural log of the vocabulary size (about 10.3). If it is very different, something is wired wrong.
''')
code(r'''
def trim_batch(ids):
    # cut the padded tensor to the longest real caption in the batch to save compute
    max_len = int((ids != PAD_ID).sum(1).max().item())
    return ids[:, :max_len]


pixels, ids, _ = next(iter(val_loader))
pixels, ids = pixels.to(DEVICE), trim_batch(ids.to(DEVICE))
with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP_ENABLED):
    logits = model(pixels, ids[:, :-1])
init_loss = F.cross_entropy(logits.float().reshape(-1, VOCAB_SIZE), ids[:, 1:].reshape(-1), ignore_index=PAD_ID)
print(f"Input shape  : {tuple(ids[:, :-1].shape)}")
print(f"Output shape : {tuple(logits.shape)}  (batch, words, vocabulary)")
print(f"Initial loss : {init_loss.item():.3f}")
print(f"ln(vocab)    : {math.log(VOCAB_SIZE):.3f}")
''')

# PART 8. DECODING
md(r'''
Step 32: Greedy decoding. To write a caption we start with the start token, ask the model for the most likely next word, add it, and repeat until the end token appears. This version handles a whole batch of images at once, which makes evaluation fast.
''')
code(r'''
def ids_to_text(ids):
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if EOS_ID in ids:
        ids = ids[: ids.index(EOS_ID)]
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


@torch.no_grad()
def greedy_decode(model, pixel_values, max_len=Config.MAX_CAPTION_LEN):
    model.eval()
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP_ENABLED):
        memory = model.encode(pixel_values)
        B = pixel_values.size(0)
        seqs = torch.full((B, 1), BOS_ID, dtype=torch.long, device=pixel_values.device)
        finished = torch.zeros(B, dtype=torch.bool, device=pixel_values.device)
        for _ in range(max_len - 1):
            logits, _ = model.decoder(seqs, memory)
            nxt = logits[:, -1].argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, PAD_ID), nxt)
            seqs = torch.cat([seqs, nxt[:, None]], dim=1)
            finished |= nxt == EOS_ID
            if finished.all():
                break
    return seqs
''')

md(r'''
Step 33: Beam search. Instead of keeping only the single best word at each step, we keep the k best partial captions and extend all of them. At the end we choose the caption with the best score. The score is divided by the caption length raised to a power (length penalty), so that the search does not simply prefer very short captions.
''')
code(r'''
@torch.no_grad()
def beam_search(model, pixel_values, beam_size=Config.BEAM_SIZE, max_len=Config.MAX_CAPTION_LEN,
                length_penalty=Config.LENGTH_PENALTY):
    # pixel_values must contain exactly one image: shape (1, 3, H, W)
    model.eval()
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP_ENABLED):
        memory = model.encode(pixel_values)
    seqs = torch.full((1, 1), BOS_ID, dtype=torch.long, device=pixel_values.device)
    scores = torch.zeros(1, device=pixel_values.device)
    completed = []
    for _ in range(max_len - 1):
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP_ENABLED):
            logits, _ = model.decoder(seqs, memory.expand(seqs.size(0), -1, -1))
        logp = F.log_softmax(logits[:, -1].float(), dim=-1)
        V = logp.size(-1)
        cand = (scores[:, None] + logp).view(-1)
        top_scores, top_idx = cand.topk(min(2 * beam_size, cand.numel()))
        new_seqs, new_scores = [], []
        for s, idx in zip(top_scores.tolist(), top_idx.tolist()):
            b, tok = divmod(idx, V)
            seq = torch.cat([seqs[b], torch.tensor([tok], device=seqs.device)])
            if tok == EOS_ID:
                completed.append((seq, s / ((len(seq) - 1) ** length_penalty)))
            else:
                new_seqs.append(seq)
                new_scores.append(s)
            if len(new_seqs) == beam_size:
                break
        if len(completed) >= beam_size or not new_seqs:
            break
        seqs = torch.stack(new_seqs)
        scores = torch.tensor(new_scores, device=seqs.device)
    if not completed:
        completed = [(seqs[i], scores[i].item() / ((seqs.size(1) - 1) ** length_penalty))
                     for i in range(seqs.size(0))]
    return max(completed, key=lambda x: x[1])[0].tolist()


def caption_from_path(path, beam_size=Config.BEAM_SIZE):
    px = eval_transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
    if beam_size <= 1:
        return ids_to_text(greedy_decode(model, px)[0, 1:])
    return ids_to_text(beam_search(model, px, beam_size=beam_size))
''')

md(r'''
Step 34: Choose a few fixed validation images that we will caption after every epoch, so we can watch the model improve. First we caption them with the untrained model. As expected, the output is random words, which gives us a baseline to compare against.
''')
code(r'''
rng = random.Random(Config.SEED)
monitor_ids = rng.sample(val_ds.img_ids, Config.NUM_MONITOR_IMAGES)

fig, axes = plt.subplots(1, len(monitor_ids), figsize=(5 * len(monitor_ids), 5.5))
for ax, img_id in zip(axes, monitor_ids):
    pred = caption_from_path(val_ds.image_path(img_id), beam_size=1)
    ax.imshow(Image.open(val_ds.image_path(img_id)).convert("RGB"))
    ax.set_title(f"untrained: {pred[:60]}\nhuman: {val_ds.captions[img_id][0]}", fontsize=9, loc="left", wrap=True)
    ax.axis("off")
plt.tight_layout()
plt.show()
''')

# PART 9. TRAINING SETUP
md(r'''
Step 35: The optimiser and the learning rate schedule. We use AdamW. Bias and normalisation parameters get no weight decay. The learning rate rises linearly during the first 5 percent of the steps (warm up) and then follows a cosine curve down to zero.
''')
code(r'''
def build_optimizer(model):
    dec_decay, dec_no_decay, enc_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            enc_params.append(p)
        elif p.ndim < 2:
            dec_no_decay.append(p)
        else:
            dec_decay.append(p)
    groups = [
        {"params": dec_decay, "lr": Config.LEARNING_RATE, "weight_decay": Config.WEIGHT_DECAY},
        {"params": dec_no_decay, "lr": Config.LEARNING_RATE, "weight_decay": 0.0},
    ]
    if enc_params:
        groups.append({"params": enc_params, "lr": Config.ENCODER_LR, "weight_decay": Config.WEIGHT_DECAY})
    return torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)


def make_lr_lambda(total_steps):
    warmup = max(1, int(Config.WARMUP_RATIO * total_steps))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return lr_lambda


TOTAL_STEPS = Config.EPOCHS * len(train_loader)
print(f"Steps per epoch: {len(train_loader):,} | total steps: {TOTAL_STEPS:,}")
''')

md(r'''
Step 36: Plot the planned learning rate over the whole training run.
''')
code(r'''
lam = make_lr_lambda(TOTAL_STEPS)
steps = np.arange(TOTAL_STEPS)
plt.figure(figsize=(10, 3.5))
plt.plot(steps, [Config.LEARNING_RATE * lam(s) for s in steps])
for e in range(1, Config.EPOCHS):
    plt.axvline(e * len(train_loader), color="grey", linestyle=":", linewidth=0.8)
plt.xlabel("training step"); plt.ylabel("learning rate")
plt.title("Learning rate schedule (dotted lines = epoch boundaries)")
plt.tight_layout()
plt.show()
''')

md(r'''
Step 37: The training function for one epoch. We use teacher forcing: the input is the caption without its last token and the target is the caption without its first token, so at every position the model learns to predict the next word. The loss is cross entropy with label smoothing of 0.1, and padding positions are ignored. Mixed precision (float16) makes training faster on the GPU, and gradient clipping protects against very large updates.
''')
code(r'''
def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, epoch):
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    total, n, window, step_log = 0.0, 0, [], []
    pbar = tqdm(loader, desc=f"Epoch {epoch} train", leave=False)
    for step, (pixels, ids, _) in enumerate(pbar):
        pixels = pixels.to(DEVICE, non_blocking=True)
        ids = trim_batch(ids.to(DEVICE, non_blocking=True))
        tgt_in, tgt_out = ids[:, :-1], ids[:, 1:]

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP_ENABLED):
            logits = model(pixels, tgt_in)
        loss = criterion(logits.float().reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, Config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total += loss.item()
        n += 1
        window.append(loss.item())
        if len(window) == Config.LOG_EVERY:
            step_log.append(float(np.mean(window)))
            window = []
        if step % 50 == 0:
            pbar.set_postfix(loss=f"{total / n:.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total / max(1, n), step_log
''')

md(r'''
Step 38: The validation function. Here we use plain cross entropy without label smoothing, so we can also report the perplexity, which is the exponential of this loss. Lower perplexity means the model is less surprised by the real captions.
''')
code(r'''
@torch.no_grad()
def validate(model, loader):
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")
    total, tokens = 0.0, 0
    for pixels, ids, _ in tqdm(loader, desc="validation", leave=False):
        pixels = pixels.to(DEVICE, non_blocking=True)
        ids = trim_batch(ids.to(DEVICE, non_blocking=True))
        tgt_in, tgt_out = ids[:, :-1], ids[:, 1:]
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=AMP_ENABLED):
            logits = model(pixels, tgt_in)
        total += ce(logits.float().reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
        tokens += (tgt_out != PAD_ID).sum().item()
    return total / max(1, tokens)
''')

md(r'''
Step 39: Helpers to save and load checkpoints. The last checkpoint contains everything needed to continue training if the session stops. The best model file is updated only when the validation loss improves.
''')
code(r'''
BEST_PATH = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
LAST_PATH = os.path.join(Config.OUTPUT_DIR, "last_checkpoint.pth")
HISTORY_PATH = os.path.join(Config.OUTPUT_DIR, "history.json")


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val, history):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "best_val": best_val, "history": history}, path)


def new_history():
    return {"epoch": [], "train_loss": [], "val_loss": [], "val_ppl": [], "lr": [], "time_min": [],
            "gpu_mem_gb": [], "step_loss": [], "samples": {str(i): [] for i in monitor_ids}}
''')

md(r'''
Step 40: Create the loss function, optimiser, scheduler and gradient scaler. If a previous checkpoint exists, training continues from it.
''')
code(r'''
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=Config.LABEL_SMOOTHING)
optimizer = build_optimizer(model)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(TOTAL_STEPS))
scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

start_epoch, best_val, history = 1, float("inf"), new_history()
if Config.RESUME and os.path.exists(LAST_PATH):
    ckpt = torch.load(LAST_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch, best_val, history = ckpt["epoch"] + 1, ckpt["best_val"], ckpt["history"]
    print(f"Resumed after epoch {ckpt['epoch']} (best validation loss {best_val:.4f})")
else:
    print("Starting a new training run")
''')

# PART 10. TRAINING
md(r'''
Step 41: Train the model. After every epoch we print the training loss, validation loss, perplexity, time and GPU memory, save the checkpoints, and print the captions the model now writes for our fixed monitor images.
''')
code(r'''
for epoch in range(start_epoch, Config.EPOCHS + 1):
    t0 = time.time()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_loss, step_log = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, criterion, epoch)
    val_loss = validate(model, val_loader)
    minutes = (time.time() - t0) / 60
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_ppl"].append(math.exp(val_loss))
    history["lr"].append(scheduler.get_last_lr()[0])
    history["time_min"].append(minutes)
    history["gpu_mem_gb"].append(gpu_mem)
    history["step_loss"].extend(step_log)

    improved = val_loss < best_val
    if improved:
        best_val = val_loss
        torch.save(model.state_dict(), BEST_PATH)

    print(f"Epoch {epoch}/{Config.EPOCHS} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
          f"perplexity {math.exp(val_loss):.2f} | {minutes:.1f} min | GPU {gpu_mem:.1f} GB"
          f"{' | new best model saved' if improved else ''}")
    for img_id in monitor_ids:
        cap = caption_from_path(val_ds.image_path(img_id))
        history["samples"][str(img_id)].append(cap)
        print(f"    image {img_id}: {cap}")

    save_checkpoint(LAST_PATH, model, optimizer, scheduler, scaler, epoch, best_val, history)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

print(f"\nTraining finished. Best validation loss: {best_val:.4f} (perplexity {math.exp(best_val):.2f})")
''')

# PART 11. TRAINING ANALYSIS
md(r'''
Step 42: Plot the training loss recorded every few steps. The light line is the raw value and the dark line is a moving average. Note that the training loss includes label smoothing, so it can not go down to zero.
''')
code(r'''
with open(HISTORY_PATH) as f:
    history = json.load(f)

sl = np.array(history["step_loss"])
x = np.arange(len(sl)) * Config.LOG_EVERY
k = max(1, len(sl) // 100)
smooth = np.convolve(sl, np.ones(k) / k, mode="valid")
steps_per_epoch = len(train_loader)

plt.figure(figsize=(12, 4))
plt.plot(x, sl, alpha=0.3, label="raw")
plt.plot(x[k - 1:], smooth, color="darkblue", label=f"moving average ({k * Config.LOG_EVERY} steps)")
for e in range(1, len(history["epoch"])):
    plt.axvline(e * steps_per_epoch, color="grey", linestyle=":", linewidth=0.8)
plt.xlabel("training step"); plt.ylabel("loss (label smoothed)")
plt.title("Training loss during training")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, "train_step_loss.png"), dpi=120)
plt.show()
''')

md(r'''
Step 43: Plot the loss per epoch for training and validation, together with the validation perplexity and the learning rate at the end of each epoch.
''')
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
axes[0].plot(history["epoch"], history["train_loss"], "o-", label="train (smoothed CE)")
axes[0].plot(history["epoch"], history["val_loss"], "s-", label="validation (CE)")
axes[0].set_title("Loss per epoch"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history["epoch"], history["val_ppl"], "s-", color="green")
for e, p in zip(history["epoch"], history["val_ppl"]):
    axes[1].annotate(f"{p:.1f}", (e, p), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
axes[1].set_title("Validation perplexity"); axes[1].set_xlabel("epoch"); axes[1].grid(alpha=0.3)
axes[2].plot(history["epoch"], history["lr"], "o-", color="red")
axes[2].set_title("Learning rate at end of epoch"); axes[2].set_xlabel("epoch"); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, "learning_curves.png"), dpi=150)
plt.show()
''')

md(r'''
Step 44: A table with the numbers for every epoch.
''')
code(r'''
epoch_df = pd.DataFrame({
    "epoch": history["epoch"],
    "train loss": np.round(history["train_loss"], 4),
    "val loss": np.round(history["val_loss"], 4),
    "val perplexity": np.round(history["val_ppl"], 2),
    "learning rate": [f"{v:.2e}" for v in history["lr"]],
    "time (min)": np.round(history["time_min"], 1),
    "peak GPU memory (GB)": np.round(history["gpu_mem_gb"], 2),
})
epoch_df.to_csv(os.path.join(Config.OUTPUT_DIR, "epoch_summary.csv"), index=False)
display(epoch_df)
''')

md(r'''
Step 45: Show how the captions for the monitor images changed from epoch to epoch.
''')
code(r'''
n = len(monitor_ids)
fig, axes = plt.subplots(n, 2, figsize=(16, 3.6 * n), gridspec_kw={"width_ratios": [1, 2.2]})
for r, img_id in enumerate(monitor_ids):
    axes[r, 0].imshow(Image.open(val_ds.image_path(img_id)).convert("RGB"))
    axes[r, 0].axis("off")
    lines = [f"epoch {e}: {c}" for e, c in zip(history["epoch"], history["samples"][str(img_id)])]
    lines.append(f"human  : {val_ds.captions[img_id][0]}")
    axes[r, 1].text(0, 0.5, "\n".join(lines), fontsize=10, va="center", family="monospace")
    axes[r, 1].axis("off")
plt.suptitle("Caption evolution across epochs")
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, "caption_evolution.png"), dpi=110)
plt.show()
''')

# PART 12. EVALUATION
md(r'''
Step 46: Load the weights of the best epoch (the one with the lowest validation loss) for all the evaluation that follows.
''')
code(r'''
if os.path.exists(BEST_PATH):
    model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE, weights_only=False))
    print(f"Loaded best model from {BEST_PATH}")
model.eval()
refs = {i: val_ds.captions[i] for i in val_ds.img_ids}
''')

md(r'''
Step 47: Define the evaluation scores. Every generated caption is compared with all human captions of the same image. BLEU counts how many word sequences of length 1 to 4 match. METEOR also accepts word stems and synonyms. ROUGE-L looks at the longest common sequence of words. CIDEr gives more weight to words that are typical for this image and rare in others, and is the main score used in image captioning research.
''')
code(r'''
try:
    from pycocoevalcap.cider.cider import Cider
    HAS_CIDER = True
except Exception as e:
    HAS_CIDER = False
    print(f"CIDEr not available: {e}")

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
_smooth = SmoothingFunction().method1


def compute_metrics(preds, refs, per_image=False):
    ids = list(preds.keys())
    hyp = [simple_words(preds[i]) for i in ids]
    ref = [[simple_words(r) for r in refs[i]] for i in ids]
    m = {}
    for n in range(1, 5):
        m[f"BLEU-{n}"] = corpus_bleu(ref, hyp, weights=tuple([1.0 / n] * n), smoothing_function=_smooth)
    try:
        m["METEOR"] = float(np.mean([meteor_score(r, h) for r, h in zip(ref, hyp)]))
    except Exception as e:
        print(f"METEOR skipped: {e}")
    rouge_each = [max(_rouge.score(" ".join(r), " ".join(h))["rougeL"].fmeasure for r in rs)
                  for rs, h in zip(ref, hyp)]
    m["ROUGE-L"] = float(np.mean(rouge_each))
    cider_each = None
    if HAS_CIDER:
        gts = {i: [" ".join(r) for r in rs] for i, rs in zip(ids, ref)}
        res = {i: [" ".join(h)] for i, h in zip(ids, hyp)}
        m["CIDEr"], cider_each = Cider().compute_score(gts, res)
    if not per_image:
        return m
    each = pd.DataFrame({"image_id": ids, "prediction": [preds[i] for i in ids],
                         "BLEU-4": [sentence_bleu(r, h, smoothing_function=_smooth) for r, h in zip(ref, hyp)],
                         "ROUGE-L": rouge_each})
    if cider_each is not None:
        each["CIDEr"] = np.asarray(cider_each)
    return m, each
''')

md(r'''
Step 48: Generate captions with greedy decoding for the whole validation set (5000 images).
''')
code(r'''
eval_ids = val_ds.img_ids[: Config.EVAL_NUM_IMAGES]
eval_loader = make_loader(CocoImageDataset(val_ds, eval_transform, eval_ids), shuffle=False)

t0 = time.time()
greedy_preds = {}
for pixels, img_ids in tqdm(eval_loader, desc="greedy decoding"):
    seqs = greedy_decode(model, pixels.to(DEVICE))
    for img_id, seq in zip(img_ids.tolist(), seqs[:, 1:]):
        greedy_preds[img_id] = ids_to_text(seq)
greedy_time = time.time() - t0
print(f"Captioned {len(greedy_preds)} images in {greedy_time:.1f} s ({1000 * greedy_time / len(greedy_preds):.1f} ms per image)")
for i in eval_ids[:5]:
    print(f"  {i}: {greedy_preds[i]}")
''')

code(r'''
greedy_metrics, greedy_each = compute_metrics(greedy_preds, refs, per_image=True)
for k, v in greedy_metrics.items():
    print(f"{k:8s}: {v:.4f}")
''')

md(r'''
Step 49: Generate captions with beam search (beam size 3) for the first 500 validation images. Beam search is slower, so we use a subset and compare it with greedy decoding on exactly the same images.
''')
code(r'''
beam_ids = val_ds.img_ids[: Config.BEAM_EVAL_IMAGES]
t0 = time.time()
beam_preds = {i: caption_from_path(val_ds.image_path(i)) for i in tqdm(beam_ids, desc=f"beam search k={Config.BEAM_SIZE}")}
beam_time = time.time() - t0
print(f"{1000 * beam_time / len(beam_ids):.1f} ms per image")

beam_metrics = compute_metrics(beam_preds, refs)
greedy_subset_metrics = compute_metrics({i: greedy_preds[i] for i in beam_ids}, refs)
''')

md(r'''
Step 50: Results table for the report.
''')
code(r'''
results_df = pd.DataFrame({
    f"Greedy ({len(greedy_preds)} images)": greedy_metrics,
    f"Greedy ({len(beam_ids)} images)": greedy_subset_metrics,
    f"Beam k={Config.BEAM_SIZE} ({len(beam_ids)} images)": beam_metrics,
}).round(4)
results_df.to_csv(os.path.join(Config.OUTPUT_DIR, "results_table.csv"))
with open(os.path.join(Config.OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump({"greedy_full": greedy_metrics, "greedy_subset": greedy_subset_metrics,
               "beam_subset": beam_metrics, "beam_size": Config.BEAM_SIZE}, f, indent=2)
display(results_df)
''')

code(r'''
names = list(beam_metrics.keys())
xpos = np.arange(len(names))
plt.figure(figsize=(12, 4.5))
b1 = plt.bar(xpos - 0.2, [greedy_subset_metrics[k] for k in names], 0.4, label="greedy")
b2 = plt.bar(xpos + 0.2, [beam_metrics[k] for k in names], 0.4, label=f"beam k={Config.BEAM_SIZE}")
for bars in (b1, b2):
    for b in bars:
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.3f}",
                 ha="center", fontsize=8)
plt.xticks(xpos, names)
plt.ylabel("score")
plt.title(f"Greedy vs beam search on the same {len(beam_ids)} validation images")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, "greedy_vs_beam.png"), dpi=120)
plt.show()
''')

md(r'''
Step 51: Study the effect of the beam size. We caption the same 200 images with beam sizes 1, 2, 3 and 5 and record the scores and the time per image.
''')
code(r'''
study_ids = val_ds.img_ids[: Config.BEAM_STUDY_IMAGES]
study_rows = []
for k in Config.BEAM_STUDY_SIZES:
    t0 = time.time()
    preds = {i: caption_from_path(val_ds.image_path(i), beam_size=k) for i in tqdm(study_ids, desc=f"k={k}", leave=False)}
    ms = 1000 * (time.time() - t0) / len(study_ids)
    m = compute_metrics(preds, refs)
    m.update({"beam size": k, "ms per image": ms,
              "avg length": np.mean([len(simple_words(p)) for p in preds.values()])})
    study_rows.append(m)
study_df = pd.DataFrame(study_rows).set_index("beam size").round(4)
display(study_df)
''')

code(r'''
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
main_metric = "CIDEr" if "CIDEr" in study_df else "BLEU-4"
axes[0].plot(study_df.index, study_df[main_metric], "o-"); axes[0].set_title(f"{main_metric} vs beam size")
axes[1].plot(study_df.index, study_df["BLEU-4"], "o-", color="orange"); axes[1].set_title("BLEU-4 vs beam size")
axes[2].plot(study_df.index, study_df["ms per image"], "o-", color="red"); axes[2].set_title("Time per image (ms)")
for ax in axes:
    ax.set_xlabel("beam size"); ax.set_xticks(study_df.index); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, "beam_size_study.png"), dpi=120)
plt.show()
''')

# PART 13. ERROR ANALYSIS
md(r'''
Step 52: Look at the score of each image separately. The histograms show that some images get very good captions and some get poor ones.
''')
code(r'''
score_cols = [c for c in ["CIDEr", "BLEU-4", "ROUGE-L"] if c in greedy_each]
fig, axes = plt.subplots(1, len(score_cols), figsize=(6 * len(score_cols), 4))
axes = np.atleast_1d(axes)
for ax, c in zip(axes, score_cols):
    ax.hist(greedy_each[c], bins=40, color="slateblue", edgecolor="white")
    ax.axvline(greedy_each[c].mean(), color="red", linestyle="--", label=f"mean {greedy_each[c].mean():.3f}")
    ax.set_title(f"Per image {c}"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(Config.OUTPUT_DIR, "per_image_scores.png"), dpi=120)
plt.show()
''')

md(r'''
Step 53: Show the four best and the four worst captioned images according to the per image score, together with the human captions. This helps to understand what kind of images the model handles well and where it fails.
''')
code(r'''
rank_col = score_cols[0]
ranked = greedy_each.sort_values(rank_col, ascending=False)


def show_examples(rows, title):
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 6))
    for ax, (_, r) in zip(axes, rows.iterrows()):
        img_id = int(r["image_id"])
        ax.imshow(Image.open(val_ds.image_path(img_id)).convert("RGB"))
        ax.set_title(f"{rank_col} = {r[rank_col]:.2f}\nmodel: {r['prediction']}\nhuman: {refs[img_id][0]}",
                     fontsize=9, loc="left", wrap=True)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


show_examples(ranked.head(4), f"Best captions by {rank_col}")
show_examples(ranked.tail(4), f"Worst captions by {rank_col}")
''')

md(r'''
Step 54: Compare the length of the generated captions with the length of the human captions.
''')
code(r'''
pred_len = [len(simple_words(p)) for p in greedy_preds.values()]
ref_len = [len(simple_words(r)) for i in greedy_preds for r in refs[i]]
plt.figure(figsize=(10, 4))
plt.hist(ref_len, bins=range(0, 30), alpha=0.6, density=True, label=f"human (mean {np.mean(ref_len):.1f})")
plt.hist(pred_len, bins=range(0, 30), alpha=0.6, density=True, label=f"model (mean {np.mean(pred_len):.1f})")
plt.xlabel("caption length (words)"); plt.ylabel("fraction")
plt.title("Caption length: model vs human")
plt.legend()
plt.tight_layout()
plt.show()
''')

md(r'''
Step 55: Compare the vocabulary. Captioning models often use a smaller set of safe, common words than people do. Here we count the distinct words and show the words the model uses most often.
''')
code(r'''
pred_counter = Counter(w for p in greedy_preds.values() for w in simple_words(p))
ref_counter = Counter(w for i in greedy_preds for r in refs[i] for w in simple_words(r))
print(f"Distinct words used by the model : {len(pred_counter):,}")
print(f"Distinct words used by humans    : {len(ref_counter):,}")
print(f"Unique captions generated        : {len(set(greedy_preds.values())):,} out of {len(greedy_preds):,}")

top_pred = [(w, c) for w, c in pred_counter.most_common(300) if w not in STOPWORDS][:20]
words, counts = zip(*top_pred)
plt.figure(figsize=(12, 4))
plt.bar(words, counts, color="darkorange")
plt.xticks(rotation=60)
plt.title("Most frequent content words in the generated captions")
plt.tight_layout()
plt.show()
''')

md(r'''
Step 56: A grid of random validation images with the beam search caption and one human caption.
''')
code(r'''
def show_predictions(n=8, cols=4):
    ids = random.sample(val_ds.img_ids, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.2 * rows))
    for ax, img_id in zip(np.array(axes).ravel(), ids):
        ax.imshow(Image.open(val_ds.image_path(img_id)).convert("RGB"))
        ax.set_title(f"model: {caption_from_path(val_ds.image_path(img_id))}\nhuman: {refs[img_id][0]}",
                     fontsize=9, loc="left", wrap=True)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "qualitative_results.png"), dpi=120)
    plt.show()


show_predictions()
''')

# PART 14. ATTENTION
md(r'''
Step 57: Helper to get the cross attention for a generated caption. We run the decoder once more on the generated caption and collect the attention weights of every layer. Row t of the attention belongs to the step that produced word t+1, so each heatmap is matched with the word it produced.
''')
code(r'''
@torch.no_grad()
def get_caption_attention(image_path):
    img = Image.open(image_path).convert("RGB")
    px = eval_transform(img).unsqueeze(0).to(DEVICE)
    seq = beam_search(model, px)                        # start, w1, ..., (end)
    inp = torch.tensor([seq[:-1]], device=DEVICE)
    _, attns = model.forward_with_attention(px, inp)
    att = torch.stack(attns).float()[:, 0]              # (layers, heads, words, 197)
    words = [(t, tokenizer.convert_ids_to_tokens(tok)) for t, tok in enumerate(seq[1:])
             if tok not in (EOS_ID, PAD_ID)]
    return img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)), seq, att, words


def to_heatmap(att_row):
    grid = int(math.sqrt(att_row.numel() - 1))          # 14 for ViT-B/16
    a = att_row[1:].reshape(1, 1, grid, grid)           # drop the CLS patch
    a = F.interpolate(a, size=(Config.IMAGE_SIZE, Config.IMAGE_SIZE), mode="bicubic", align_corners=False)[0, 0]
    return ((a - a.min()) / (a.max() - a.min() + 1e-8)).cpu().numpy()
''')

md(r'''
Step 58: Draw one heatmap per generated word, averaged over all heads and all layers. Red areas are where the model paid most attention when it wrote that word.
''')
code(r'''
def visualize_attention(image_path, cols=5, save_as=None):
    img, seq, att, words = get_caption_attention(image_path)
    mean_att = att.mean(dim=(0, 1))                     # average over layers and heads
    rows = math.ceil((len(words) + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    axes = np.array(axes).ravel()
    axes[0].imshow(img); axes[0].set_title("input")
    for ax, (t, word) in zip(axes[1:], words):
        ax.imshow(img)
        ax.imshow(to_heatmap(mean_att[t]), cmap="jet", alpha=0.5)
        ax.set_title(word, fontsize=12)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Caption: {ids_to_text(seq)}", fontsize=14)
    plt.tight_layout()
    if save_as:
        plt.savefig(os.path.join(Config.OUTPUT_DIR, save_as), dpi=110)
    plt.show()


for n, img_id in enumerate(random.sample(val_ds.img_ids, 3)):
    visualize_attention(val_ds.image_path(img_id), save_as=f"attention_example_{n + 1}.png")
''')

md(r'''
Step 59: Compare the attention of the different decoder layers for one word. Lower layers often spread their attention widely, while higher layers tend to focus on more specific regions.
''')
code(r'''
img_id = monitor_ids[0]
img, seq, att, words = get_caption_attention(val_ds.image_path(img_id))
t, word = max(words, key=lambda tw: len(tw[1].replace("##", "")))   # pick the longest word as an example

L = att.size(0)
fig, axes = plt.subplots(1, L + 1, figsize=(3.2 * (L + 1), 3.6))
for l in range(L):
    axes[l].imshow(img); axes[l].imshow(to_heatmap(att[l].mean(0)[t]), cmap="jet", alpha=0.5)
    axes[l].set_title(f"layer {l + 1}")
axes[L].imshow(img); axes[L].imshow(to_heatmap(att.mean(dim=(0, 1))[t]), cmap="jet", alpha=0.5)
axes[L].set_title("mean of layers")
for ax in axes:
    ax.axis("off")
fig.suptitle(f"Attention for the word '{word}' in: {ids_to_text(seq)}")
plt.tight_layout()
plt.show()
''')

md(r'''
Step 60: Compare the attention of the individual heads in the last layer for the same word. Different heads often look at different parts of the image.
''')
code(r'''
H = att.size(1)
fig, axes = plt.subplots(1, H, figsize=(3 * H, 3.4))
for h in range(H):
    axes[h].imshow(img); axes[h].imshow(to_heatmap(att[-1, h, t]), cmap="jet", alpha=0.5)
    axes[h].set_title(f"head {h + 1}"); axes[h].axis("off")
fig.suptitle(f"Last layer heads for the word '{word}'")
plt.tight_layout()
plt.show()
''')

# PART 15. CUSTOM IMAGE + WRAP UP
md(r'''
Step 61: Caption any image. Upload your own picture as a Kaggle dataset or into /kaggle/working and pass its path to the function. Here we use a random validation image as an example and show the captions for greedy search and beam search.
''')
code(r'''
def caption_image(image_path):
    img = Image.open(image_path).convert("RGB")
    greedy = caption_from_path(image_path, beam_size=1)
    beam = caption_from_path(image_path, beam_size=Config.BEAM_SIZE)
    plt.figure(figsize=(6, 6)); plt.imshow(img); plt.axis("off")
    plt.title(f"greedy: {greedy}\nbeam {Config.BEAM_SIZE}: {beam}", fontsize=10, loc="left")
    plt.show()
    return beam


example_path = val_ds.image_path(random.choice(val_ds.img_ids))   # replace with your own image path
print(caption_image(example_path))
''')

md(r'''
Step 62: Final summary table for the report, built only from the values measured in this run.
''')
code(r'''
summary = {
    "training images": len(train_ds),
    "validation images": len(val_ds),
    "epochs trained": len(history["epoch"]),
    "total training time (min)": round(float(np.sum(history["time_min"])), 1),
    "best validation loss": round(min(history["val_loss"]), 4),
    "best validation perplexity": round(min(history["val_ppl"]), 2),
    "trainable parameters (M)": round(count(model, True) / 1e6, 2),
}
summary.update({f"greedy {k}": round(v, 4) for k, v in greedy_metrics.items()})
summary.update({f"beam{Config.BEAM_SIZE} {k}": round(v, 4) for k, v in beam_metrics.items()})
summary_df = pd.DataFrame(list(summary.items()), columns=["item", "value"])
summary_df.to_csv(os.path.join(Config.OUTPUT_DIR, "final_summary.csv"), index=False)
display(summary_df)
''')

md(r'''
Step 63: Save the configuration and list all files produced by this notebook. These files are kept in the output of the Kaggle version and can be downloaded.
''')
code(r'''
export_cfg = {k: (str(v) if isinstance(v, torch.device) else v) for k, v in vars(Config).items()
              if not k.startswith("_") and not callable(v)}
export_cfg.update({"VOCAB_SIZE": VOCAB_SIZE, "BOS_ID": BOS_ID, "EOS_ID": EOS_ID, "PAD_ID": PAD_ID})
with open(os.path.join(Config.OUTPUT_DIR, "config.json"), "w") as f:
    json.dump(export_cfg, f, indent=2, default=str)

for fn in sorted(os.listdir(Config.OUTPUT_DIR)):
    p = os.path.join(Config.OUTPUT_DIR, fn)
    if os.path.isfile(p):
        print(f"{fn:32s} {os.path.getsize(p) / 1e6:9.2f} MB")
''')

md(r'''
Comparison with the reference project (zarzouram/image_captioning_with_transformers)

| Aspect | Reference project | This notebook |
|---|---|---|
| Image encoder | CNN (ResNet) grid features | Pretrained ViT-B/16 patch features |
| Attention maps | Obtained by changing PyTorch source code | Returned directly by our own decoder layer |
| Decoding | Greedy search | Greedy and beam search with length penalty, plus a beam size study |
| Evaluation | BLEU | BLEU-1 to 4, METEOR, ROUGE-L, CIDEr with all human captions |
| Training | Adam | AdamW, warm up and cosine schedule, label smoothing, mixed precision, weight tying |
| Analysis | Loss curves | Data exploration, encoder feature views, per image scores, best and worst cases, layer and head attention |
''')

md(r'''
Conclusion

We trained an image captioning model that combines a pretrained Vision Transformer with a Transformer decoder written from scratch. The training curves, the caption evolution across epochs and the evaluation scores above show how well the model learned to describe COCO images. Beam search was compared with greedy decoding on the same images, and the attention heatmaps show which image regions the model used for each word.

Possible next steps are to fine tune the last two ViT blocks by setting FINE_TUNE_ENCODER_LAYERS to 2 and continuing from the best model, to train with self critical sequence training that directly optimises the CIDEr score, or to use a larger image encoder such as CLIP.
''')


# BUILD
def _check_cell(kind, src, idx):
    bad = sorted({c for c in src if ord(c) > 127})
    if bad:
        raise SystemExit(f"Non-ASCII characters in cell #{idx}: {bad}")
    if kind == "markdown":
        if "**" in src or "__" in src:
            raise SystemExit(f"Bold marker found in markdown cell #{idx}")
        if any(line.lstrip().startswith("#") for line in src.splitlines()):
            raise SystemExit(f"Heading found in markdown cell #{idx}")
        if "|" in src and "Comparison with the reference" not in src:
            raise SystemExit(f"Table found outside the report section in markdown cell #{idx}")
    else:
        py = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("!", "%")))
        try:
            ast.parse(py)
        except SyntaxError as e:
            raise SystemExit(f"Syntax error in code cell #{idx}: {e}")


def build(out_path):
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        _check_cell(kind, src, i)
        lines = src.splitlines(keepends=True)
        if kind == "code":
            cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                          "outputs": [], "source": lines})
        else:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    n_code = sum(1 for k, _ in CELLS if k == "code")
    print(f"Wrote {out_path}  ({len(cells)} cells: {n_code} code, {len(cells) - n_code} markdown)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    build(os.path.join(here, "image_captioning_kaggle.ipynb"))

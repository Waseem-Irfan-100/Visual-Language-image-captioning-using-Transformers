import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from transformers import AutoTokenizer, AutoImageProcessor
from config import Config

class CocoCaptionDataset(Dataset):
    def __init__(self, img_dir, ann_file, image_processor_name, tokenizer_name, max_length=Config.MAX_CAPTION_LEN):
        """
        Custom PyTorch Dataset for MS COCO Image Captioning.
        """
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.anns.keys())
        
        # Hugging Face Processors
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
        # Using BERT tokenizer for high-quality SubWord/WordPiece tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else '[PAD]'

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Get annotation id
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']
        
        # Load Image
        image = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
        
        # Process Image for Vision Transformer
        # Returns a dictionary, we extract pixel_values (tensor of shape 3x224x224)
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Tokenize Caption
        # Includes [CLS] at start and [SEP] at end
        tokens = self.tokenizer(
            caption, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        input_ids = tokens.input_ids.squeeze(0)
        attention_mask = tokens.attention_mask.squeeze(0)
        
        return pixel_values, input_ids, attention_mask

def get_data_loaders():
    """
    Returns train and validation DataLoaders.
    """
    train_dataset = CocoCaptionDataset(
        img_dir=Config.TRAIN_IMG_DIR,
        ann_file=Config.TRAIN_ANN_FILE,
        image_processor_name=Config.ENCODER_NAME,
        tokenizer_name='bert-base-uncased' 
    )
    
    val_dataset = CocoCaptionDataset(
        img_dir=Config.VAL_IMG_DIR,
        ann_file=Config.VAL_ANN_FILE,
        image_processor_name=Config.ENCODER_NAME,
        tokenizer_name='bert-base-uncased'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Update vocab size dynamically based on tokenizer
    Config.VOCAB_SIZE = train_dataset.tokenizer.vocab_size
    
    return train_loader, val_loader, train_dataset.tokenizer

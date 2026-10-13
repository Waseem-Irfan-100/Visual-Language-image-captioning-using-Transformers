import torch

class Config:
    # ---------------------------------------------------------
    # Paths (Configured for Kaggle's COCO 2017 Dataset)
    # ---------------------------------------------------------
    TRAIN_IMG_DIR = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
    TRAIN_ANN_FILE = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_train2017.json'
    VAL_IMG_DIR = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
    VAL_ANN_FILE = '/kaggle/input/coco-2017-dataset/coco2017/annotations/captions_val2017.json'
    
    # ---------------------------------------------------------
    # Model Hyperparameters
    # ---------------------------------------------------------
    # We use a pre-trained Vision Transformer for high-quality feature extraction
    ENCODER_NAME = 'google/vit-base-patch16-224-in21k'
    
    # Transformer Decoder params (Built from scratch for academic rigor)
    EMBED_DIM = 768        # Matches ViT base output dimension
    DECODER_LAYERS = 6
    DECODER_HEADS = 8
    DECODER_FF_DIM = 2048
    DROPOUT = 0.1
    
    MAX_CAPTION_LEN = 50   # Maximum words in a caption
    VOCAB_SIZE = 30000     # Will be set by our tokenizer
    
    # ---------------------------------------------------------
    # Training Hyperparameters
    # ---------------------------------------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    EPOCHS = 10
    NUM_WORKERS = 2
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

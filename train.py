import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import get_data_loaders
from model import ImageCaptioningModel
from config import Config

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    # Progress bar for Kaggle Notebook output
    loop = tqdm(dataloader, total=len(dataloader), desc="Training")
    
    for pixel_values, input_ids, attention_mask in loop:
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        
        # We need to create inputs (excluding the last token) and targets (excluding the first token)
        # for standard causal language modeling.
        tgt_input = input_ids[:, :-1]
        tgt_expected = input_ids[:, 1:]
        
        # Create tgt mask for self-attention in decoder
        seq_len = tgt_input.size(1)
        tgt_mask = model.decoder.generate_square_subsequent_mask(seq_len, device)
        
        # Padding mask
        tgt_padding_mask = (tgt_input == model.pad_token_id)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values, 
            input_ids=tgt_input, 
            tgt_mask=tgt_mask, 
            tgt_padding_mask=tgt_padding_mask
        )
        
        # Reshape for loss calculation: (batch_size * seq_len, vocab_size)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt_expected.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def main():
    print("Loading datasets...")
    train_loader, val_loader, tokenizer = get_data_loaders()
    
    print(f"Vocabulary Size: {Config.VOCAB_SIZE}")
    
    print("Initializing model...")
    model = ImageCaptioningModel(vocab_size=Config.VOCAB_SIZE).to(Config.DEVICE)
    
    # Add pad token id to model so train loop can use it for masking
    model.pad_token_id = tokenizer.pad_token_id
    
    # Optimizer & Loss Function
    # We ignore the loss calculated on padding tokens!
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    
    # Simple Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"Starting Training on {Config.DEVICE}...")
    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{Config.EPOCHS} ---")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        
        scheduler.step(train_loss)
        
        # Save checkpoint
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
        print(f"Model saved to model_epoch_{epoch}.pth")

if __name__ == "__main__":
    # Note: On Kaggle, this will be executed by running !python train.py 
    # or by copying the logic into a notebook cell.
    pass

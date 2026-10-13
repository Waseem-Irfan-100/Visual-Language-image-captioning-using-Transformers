import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from model import ImageCaptioningModel
from config import Config
from transformers import AutoTokenizer, AutoImageProcessor

def generate_caption_beam_search(model, image_path, tokenizer, image_processor, device, beam_size=3, max_len=Config.MAX_CAPTION_LEN):
    """
    Generates a caption using Beam Search, an algorithm that explores multiple paths 
    simultaneously, yielding far superior captions to standard greedy search.
    """
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    with torch.no_grad():
        memory = model.encoder(pixel_values)
        
    # Beams store tuples of: (sequence_of_tokens, cumulative_log_prob)
    beams = [([tokenizer.cls_token_id], 0.0)]
    completed_beams = []
    
    for step in range(max_len):
        new_beams = []
        for seq, score in beams:
            # If the sentence is already finished, save it
            if seq[-1] == tokenizer.sep_token_id or seq[-1] == tokenizer.eos_token_id:
                completed_beams.append((seq, score))
                continue
                
            tgt = torch.tensor([seq]).to(device)
            tgt_mask = model.decoder.generate_square_subsequent_mask(tgt.size(1), device)
            
            with torch.no_grad():
                out, _ = model.decoder(tgt, memory, tgt_mask=tgt_mask)
                
            # Calculate probabilities for the next token
            next_token_logits = out[0, -1, :]
            probs = torch.log_softmax(next_token_logits, dim=0)
            
            # Find the top `beam_size` words
            top_probs, top_indices = torch.topk(probs, beam_size)
            
            # Create new branched beams
            for i in range(beam_size):
                new_seq = seq + [top_indices[i].item()]
                new_score = score + top_probs[i].item()
                new_beams.append((new_seq, new_score))
                
        if not new_beams:
            break
            
        # Sort and keep only the top `beam_size` paths overall
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        beams = new_beams
        
    completed_beams.extend(beams)
    # Pick the absolute best sequence
    best_seq = max(completed_beams, key=lambda x: x[1])[0]
    
    caption = tokenizer.decode(best_seq, skip_special_tokens=True)
    return caption, best_seq

def visualize_attention(model, image_path, tokenizer, image_processor, device):
    """
    Generates a heatmap over the image showing exactly where the model was looking
    when it predicted each individual word.
    """
    # 1. Get the generated tokens
    caption, token_ids = generate_caption_beam_search(model, image_path, tokenizer, image_processor, device, beam_size=1)
    
    image = Image.open(image_path).convert('RGB')
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    tgt = torch.tensor([token_ids]).to(device)
    tgt_mask = model.decoder.generate_square_subsequent_mask(tgt.size(1), device)
    
    # 2. Extract attention maps from our custom decoder
    with torch.no_grad():
        _, attn_weights = model.forward_with_attention(pixel_values, tgt, tgt_mask=tgt_mask)
        
    # Average across all attention heads
    attn_weights = attn_weights.mean(dim=1).squeeze(0) # shape: (seq_len, num_patches)
    
    # The ViT divides the image into a 14x14 grid (196 patches + 1 CLS token)
    num_patches = attn_weights.shape[1]
    grid_size = int(np.sqrt(num_patches - 1)) 
    
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # 3. Plotting
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Caption: {caption}", fontsize=16)
    
    plot_idx = 1
    for i, token in enumerate(tokens):
        if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
            continue
            
        # Get attention weights for this specific word
        # We skip index 0 because that corresponds to the Vision CLS token, not an image patch
        attn = attn_weights[i, 1:].cpu().numpy() 
        attn = attn.reshape(grid_size, grid_size)
        
        # Scale heatmap to image size
        attn = cv2.resize(attn, (image.width, image.height))
        
        ax = fig.add_subplot(int(np.ceil(len(tokens)/4.0)), 4, plot_idx)
        ax.set_title(token.replace('##', '')) # Clean up subword tokens
        
        ax.imshow(image)
        ax.imshow(attn, cmap='jet', alpha=0.5)
        ax.axis('off')
        plot_idx += 1
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pass

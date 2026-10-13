import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
from config import Config

# Make sure to run this once to download the required NLTK resources for METEOR
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def evaluate_metrics(model, dataloader, tokenizer, device, num_batches_to_eval=10):
    """
    Evaluates the model using a comprehensive set of NLP metrics:
    - BLEU-4: Standard n-gram overlap
    - METEOR: Accounts for synonyms and stemming (much better human correlation)
    - ROUGE-L: Longest Common Subsequence (used heavily in summarization)
    """
    model.eval()
    
    # For BLEU (expects tokenized lists)
    references_bleu = []
    hypotheses_bleu = []
    
    # For METEOR and ROUGE (expects strings)
    references_str = []
    hypotheses_str = []
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    print(f"Evaluating advanced metrics on {num_batches_to_eval} batches...")
    
    with torch.no_grad():
        for batch_idx, (pixel_values, input_ids, _) in enumerate(dataloader):
            if batch_idx >= num_batches_to_eval:
                break
                
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            
            for i in range(pixel_values.size(0)):
                # 1. Ground Truth (Reference)
                ref_tokens = input_ids[i].tolist()
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                
                references_bleu.append([ref_text.split()]) 
                references_str.append([ref_text])
                
                # 2. Hypothesis (Prediction)
                img_features = model.encoder(pixel_values[i].unsqueeze(0))
                
                curr_input = [tokenizer.cls_token_id]
                for _ in range(Config.MAX_CAPTION_LEN):
                    tgt = torch.tensor([curr_input]).to(device)
                    tgt_mask = model.decoder.generate_square_subsequent_mask(tgt.size(1), device)
                    
                    # We only care about output here, ignore attention weights
                    out, _ = model.decoder(tgt, img_features, tgt_mask=tgt_mask)
                    next_token = out[0, -1, :].argmax().item()
                    
                    curr_input.append(next_token)
                    if next_token == tokenizer.sep_token_id or next_token == tokenizer.eos_token_id:
                        break
                        
                hyp_text = tokenizer.decode(curr_input, skip_special_tokens=True)
                hypotheses_bleu.append(hyp_text.split())
                hypotheses_str.append(hyp_text)
                
    # 1. BLEU-4
    bleu4 = corpus_bleu(references_bleu, hypotheses_bleu, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"Validation BLEU-4: {bleu4:.4f}")
    
    # 2. METEOR
    # METEOR compares a list of references (tokenized) to a single hypothesis (tokenized)
    meteor_scores = [meteor_score(ref, hyp) for ref, hyp in zip(references_bleu, hypotheses_bleu)]
    meteor_avg = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    print(f"Validation METEOR: {meteor_avg:.4f}")
    
    # 3. ROUGE-L
    rouge_scores = [scorer.score(ref[0], hyp)['rougeL'].fmeasure for ref, hyp in zip(references_str, hypotheses_str)]
    rouge_avg = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    print(f"Validation ROUGE-L: {rouge_avg:.4f}")
    
    return {'bleu4': bleu4, 'meteor': meteor_avg, 'rougeL': rouge_avg}

if __name__ == "__main__":
    pass

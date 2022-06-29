import torch 
import argparse
from torch.utils.data import DataLoader  
import os 
import numpy as np 
import torch.nn.functional as F 

from perceiver_ar_pytorch import PerceiverAR 
from dataset import create_epiano_datasets 
from preprocess import decode_midi 
from utils import *
from tqdm import tqdm 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def greedy_decode(condi, model, args): 
    ys = [] 
    for i in range(args.max_sequence): 
        if i > 0: 
            predict = torch.tensor(ys).long().unsqueeze(0)
            input = torch.cat((condi, predict), dim=-1) 
        else: 
            input = condi
        out = model(input)
        logits = out[0][-1, :].cpu().data.numpy() 
        next_token = np.argsort(logits)[-1] 
        ys.append(next_token) 
    return ys 



def sample_sequence(condi, model, args, temperature=0.7, top_k=0, top_p=0.9): 
    ys = [] 
    for i in range(args.max_sequence): 
        if i > 0: 
            predict = torch.tensor(ys).long().unsqueeze(0) 
            input = torch.cat((condi, predict), dim=-1) 
        else: 
            input = condi 
        out = model(input)
        logits = out[0][-1, :] / temperature 
        logits = top_filtering(logits, top_k=top_k, top_p=top_p) 
        probs = F.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, 1).item() 
        max_iter = 0 
        while next_token == TOKEN_END or next_token == TOKEN_PAD: 
            next_token = torch.multinomial(probs, 1).item() 
            max_iter += 1 
            if max_iter > 100: 
                next_token = TOKEN_END - 1 
        ys.append(next_token) 
        if len(ys) > 100:
            break
        
    return ys 



def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits





def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="./data", help="Folder of preprocessed and pickled midi files") 
    parser.add_argument("-ckpt_dir", type=str, default="./ckpt", help="Folder to save model weights. Saves one every epoch")
    parser.add_argument("-output_dir", type=str, default="./result", help="Folder to save the generated music with midi form")
    parser.add_argument("-n_workers", type=int, default=1, help="Number of threads for the dataloader")
    parser.add_argument("-batch_size", type=int, default=1, help="Batch size to use") 
    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("-num_prime", type=int, default=1024, help="Amount of messages to prime the generator with")
    parser.add_argument("-decode_method", type=str, default="sample", help="greedy, sample, beam search")
    args = parser.parse_args() 

    train_dataset, val_dataset, test_dataset = create_epiano_datasets(args.data_dir, args.max_sequence) 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers) 

    model = PerceiverAR(
        num_tokens = VOCAB_SIZE, 
        dim = args.d_model, 
        depth = 8, 
        dim_head = 64, 
        heads = 8, 
        max_seq_len = args.max_sequence, 
        cross_attn_seq_len = 1024, 
        cross_attn_dropout = 0.5,
    )
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'latest.pth'))['state_dict'])
    model = model.to(device) 

    model.eval() 
    with torch.no_grad(): 
        with tqdm(enumerate(test_loader), total=len(test_loader)) as t: 
            for batch_num, batch in t: 
                x = batch[0].to(device) 
                tgt = batch[1].to(device) 
                condi = x[:, :1025] 
                if args.decode_method == 'greedy': 
                    predict = greedy_decode(condi, model, args) 
                elif args.decode_method == 'sample':
                    predict = sample_sequence(condi, model, args) 
                
                predict = np.array(predict) 
                midi_name = 'sample_' + str(batch_num) + '.mid' 
                midi_path = os.path.join(args.output_dir, midi_name)
                decode_midi(predict, file_path=midi_path) 
                break


if __name__ == "__main__":
    main()
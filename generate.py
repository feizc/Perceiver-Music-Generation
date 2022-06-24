import torch 
import argparse
from torch.utils.data import DataLoader  
import os 

from perceiver_ar_pytorch import PerceiverAR 
from dataset import create_epiano_datasets 
from preprocess import decode_midi 
from utils import *
from tqdm import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="./data", help="Folder of preprocessed and pickled midi files") 
    parser.add_argument("-ckpt_dir", type=str, default="./ckpt", help="Folder to save model weights. Saves one every epoch")
    parser.add_argument("-n_workers", type=int, default=1, help="Number of threads for the dataloader")
    parser.add_argument("-batch_size", type=int, default=1, help="Batch size to use") 
    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    parser.add_argument("-num_prime", type=int, default=1024, help="Amount of messages to prime the generator with")
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
                decode_midi(tgt[0].cpu().numpy(), file_path='test.mid') 



if __name__ == "__main__":
    main()
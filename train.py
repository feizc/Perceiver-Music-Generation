import os 
import torch 
import torch.nn as nn 
from torch.optim.lr_scheduler import LambdaLR 
from torch.utils.data import DataLoader
from torch.optim import Adam 
import argparse
from tqdm import tqdm 

from preprocess import decode_midi 
from dataset import create_epiano_datasets 
from perceiver_ar_pytorch import PerceiverAR 
from utils import * 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def train(cur_epoch, model, data_loader, opt, lr_scheduler=None): 
    model.train() 
    sum_loss = .0 
    sum_acc = .0 
    
    with tqdm(enumerate(data_loader), total=len(data_loader)) as t: 
        for batch_num, batch in t: 
            opt.zero_grad()
            x   = batch[0].to(device)
            tgt = batch[1].to(device) 
            out = model(x, labels=tgt) 
            loss = out[0] 
            acc = out[1]
            loss.backward()
            opt.step() 
            if(lr_scheduler is not None):
                lr_scheduler.step()
            sum_loss += loss.item()
            sum_acc += acc.item()
            t.set_description('Epoch %i' % cur_epoch)
            t.set_postfix(loss=sum_loss / (batch_num+1), acc=sum_acc/(batch_num+1))
            break 


def eval(model, data_loader): 
    model.eval() 
    sum_loss = .0 
    sum_acc = .0 
    with torch.no_grad(): 
        with tqdm(enumerate(data_loader), total=len(data_loader)) as t: 
            for batch_num, batch in t: 
                x = batch[0].to(device) 
                tgt = batch[1].to(device) 
                out = model(x, labels=tgt) 
                loss = out[0].item() 
                acc = out[1].item() 
                sum_loss += loss 
                sum_acc += acc 
                t.set_description('Evaluation')
                t.set_postfix(loss=sum_loss / (batch_num+1), acc=sum_acc/(batch_num+1))
                break 
    return sum_acc/(batch_num+1) 



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", type=str, default="./data", help="Folder of preprocessed and pickled midi files") 
    parser.add_argument("-ckpt_dir", type=str, default="./ckpt", help="Folder to save model weights. Saves one every epoch")
    parser.add_argument("-lr", type=float, default=None, help="Constant learn rate. Leave as None for a custom scheduler.")
    parser.add_argument("-n_workers", type=int, default=1, help="Number of threads for the dataloader")
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size to use")
    parser.add_argument("-epochs", type=int, default=100, help="Number of epochs to use")
    parser.add_argument("-max_sequence", type=int, default=2048, help="Maximum midi sequence to consider")
    parser.add_argument("-d_model", type=int, default=512, help="Dimension of the model (output dim of embedding layers, etc.)")
    args = parser.parse_args() 

    os.makedirs(args.ckpt_dir, exist_ok=True) 

    ##### Datasets #####
    train_dataset, val_dataset, test_dataset = create_epiano_datasets(args.data_dir, args.max_sequence) 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers) 

    ##### Model #####
    model = PerceiverAR(
        num_tokens = VOCAB_SIZE, 
        dim = args.d_model, 
        depth = 8, 
        dim_head = 64, 
        heads = 8, 
        max_seq_len = args.max_sequence, 
        cross_attn_seq_len = 1024, 
        cross_attn_dropout = 0.5,
    ).to(device) 

    ##### Learning rate scheduler and optimizer #####
    if args.lr is None: 
        init_step = 0 
        lr = LR_DEFAULT_START 
        lr_stepper = LrStepTracker(args.d_model, SCHEDULER_WARMUP_STEPS, init_step)
    else: 
        lr = args.lr

    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

    if args.lr is None: 
        lr_scheduler = LambdaLR(opt, lr_stepper.step)
    else:
        lr_scheduler = None

    ##### Loss #####
    # loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    # print(TOKEN_PAD)
    
    best_acc = .0 
    for epoch in range(args.epochs): 
        train(epoch, model, train_loader, opt, lr_scheduler)
        acc = eval(model, val_loader)
        if acc > best_acc: 
            best_acc = acc 
            torch.save({
                'state_dict': model.state_dict(),
                'optimizer': opt.state_dict(),
            }, os.path.join(args.ckpt_dir, 'latest.pth'))
        break 



if __name__ == "__main__":
    main() 
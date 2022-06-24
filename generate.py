from preprocess import decode_midi 
import torch 

tgt = torch.randint(0, 390, (1, 2048))
decode_midi(tgt[0].cpu().numpy(), file_path='test.mid')
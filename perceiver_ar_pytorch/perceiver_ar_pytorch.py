import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim, bias = False)
    )

# rotary positional embedding
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device = device, dtype = self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim = -1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(pos, t):
    seq_len, rotate_dim = t.shape[-2], pos.shape[-1]
    pos = pos[..., -seq_len:, :]
    t, t_pass = t[..., :rotate_dim], t[..., rotate_dim:]
    t = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return torch.cat((t, t_pass), dim = -1)

# attention

class CausalAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, rotary_pos_emb = None):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(rotary_pos_emb, q)
            k = apply_rotary_pos_emb(rotary_pos_emb, k)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CausalPrefixAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        max_heads_process = 2,
        dropout = 0.,
        cross_attn_dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.max_heads_process = max_heads_process

        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.cross_attn_dropout = cross_attn_dropout # they drop out a percentage of the prefix during training, shown to help prevent overfitting

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, context_mask = None, rotary_pos_emb = None):
        batch, context_len, device = x.shape[0], context.shape[-2], x.device

        q_rotary_pos_emb = rotary_pos_emb
        k_rotary_pos_emb = rotary_pos_emb

        # take care of cross attention dropout

        if self.training and self.cross_attn_dropout > 0.:
            rand = torch.zeros((batch, context_len), device = device).uniform_()
            keep_context_len = context_len - int(context_len * self.cross_attn_dropout)
            keep_indices = rand.topk(keep_context_len, dim = -1).indices
            keep_mask = torch.zeros_like(rand).scatter_(1, keep_indices, 1).bool()

            context = rearrange(context[keep_mask], '(b n) d -> b n d', b = batch)

            if exists(context_mask):
                context_mask = rearrange(context_mask[keep_mask], '(b n) -> b n', b = batch)

            # operate on rotary position embeddings for keys

            k_rotary_pos_emb = repeat(k_rotary_pos_emb, '... -> b ...', b = batch)
            k_rotary_pos_emb_context, k_rotary_pos_emb_seq = k_rotary_pos_emb[:, :context_len], k_rotary_pos_emb[:, context_len:]
            k_rotary_pos_emb_context = rearrange(k_rotary_pos_emb_context[keep_mask], '(b n) d -> b n d', b = batch)

            k_rotary_pos_emb = torch.cat((k_rotary_pos_emb_context, k_rotary_pos_emb_seq), dim = 1)
            k_rotary_pos_emb = rearrange(k_rotary_pos_emb, 'b n d -> b 1 n d')

        # normalization

        x = self.norm(x)
        context = self.context_norm(context)

        # derive queries, keys, values

        q = self.to_q(x)

        k_input, v_input = self.to_kv(x).chunk(2, dim = -1)
        k_context, v_context = self.to_kv(context).chunk(2, dim = -1)

        k = torch.cat((k_context, k_input), dim = 1)
        v = torch.cat((v_context, v_input), dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        # rotate queries and keys with rotary embeddings

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q_rotary_pos_emb, q)
            k = apply_rotary_pos_emb(k_rotary_pos_emb, k)

        # take care of masking

        i, j = q.shape[-2], k.shape[-2]
        mask_value = -torch.finfo(q.dtype).max

        if exists(context_mask):
            mask_len = context_mask.shape[-1]
            context_mask = F.pad(context_mask, (0, max(j - mask_len, 0)), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')

        causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)

        # process in chunks of heads

        out = []

        max_heads = self.max_heads_process

        for q_chunk, k_chunk, v_chunk in zip(q.split(max_heads, dim = 1), k.split(max_heads, dim = 1), v.split(max_heads, dim = 1)):
            sim = einsum('b h i d, b h j d -> b h i j', q_chunk, k_chunk)

            if exists(context_mask):
                sim = sim.masked_fill(~context_mask, mask_value)

            sim = sim.masked_fill(causal_mask, mask_value)

            attn = sim.softmax(dim = -1)
            attn = self.dropout(attn)

            out_chunk = einsum('b h i j, b h j d -> b h i d', attn, v_chunk)
            out.append(out_chunk)

        # concat all the heads together

        out = torch.cat(out, dim = 1)

        # merge heads and then combine with linear

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class PerceiverAR(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        cross_attn_seq_len,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        cross_attn_dropout = 0.,
        ff_mult = 4,
        perceive_depth = 1,
        perceive_max_heads_process = 2, # processes the heads in the perceiver layer in chunks to lower peak memory, in the case the prefix is really long
        return_encoder_hidden_states = False,
    ):
        super().__init__()
        assert max_seq_len > cross_attn_seq_len, 'max_seq_len must be greater than cross_attn_seq_len, the length of the sequence for which to cross attend to "perceiver" style'
        self.max_seq_len = max_seq_len
        self.cross_attn_seq_len = cross_attn_seq_len

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.rotary_pos_emb = RotaryEmbedding(dim = max(32, dim_head // 2))

        self.perceive_layers  = nn.ModuleList([])
        
        self.token_pad = 389

        for _ in range(perceive_depth):
            self.perceive_layers.append(nn.ModuleList([
                CausalPrefixAttention(dim = dim, dim_head = dim_head, heads = heads, max_heads_process = perceive_max_heads_process, dropout = dropout, cross_attn_dropout = cross_attn_dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim, mult = ff_mult, dropout = dropout),
            ]))

        self.to_logits = nn.Linear(dim, num_tokens, bias = False) 
        self.return_encoder_hidden_states = return_encoder_hidden_states 
    

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.token_pad) 
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc


    def forward(
        self,
        x,
        prefix_mask = None,
        labels = None
    ):
        seq_len, device = x.shape[1], x.device
        assert self.cross_attn_seq_len < seq_len <= self.max_seq_len

        x = self.token_emb(x)
        x = x + self.pos_emb(torch.arange(seq_len, device = device))

        # rotary positional embedding

        rotary_pos_emb = self.rotary_pos_emb(seq_len, device = device)

        # divide into prefix to cross attend to and sequence to self attend to

        prefix, x = x[:, :self.cross_attn_seq_len], x[:, self.cross_attn_seq_len:]

        # initial perceiver attention and feedforward (one cross attention)

        for cross_attn, ff in self.perceive_layers:
            x = cross_attn(x, prefix, context_mask = prefix_mask, rotary_pos_emb = rotary_pos_emb) + x
            x = ff(x) + x

        # layers

        for attn, ff in self.layers:
            x = attn(x, rotary_pos_emb = rotary_pos_emb) + x
            x = ff(x) + x

        # to logits

        logits = self.to_logits(x)

        # take care of cross entropy loss if labels are provided

        if not exists(labels): 
            if self.return_encoder_hidden_states == True: 
                return (logits, prefix) 
            else: 
                return logits 

        labels = labels[:, self.cross_attn_seq_len:]
        loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), labels, ignore_index = self.token_pad) 
        acc = self.compute_accuracy(logits, labels) 
        return (loss, acc,)



class CopyPerceiverAR(nn.Module): 
    """
    PerceiverAR with copy mechanism 
    """

    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        cross_attn_seq_len,
    ):
        super().__init__() 
        self.model = PerceiverAR(
            num_tokens = num_tokens, 
            dim = dim, 
            depth = depth, 
            max_seq_len = max_seq_len, 
            cross_attn_seq_len = cross_attn_seq_len, 
            return_encoder_hidden_states = True, 
        ) 

        self.attn_layer = nn.Linear(dim, 1, bias=True) 

        self.p_gen_context_layer = nn.Linear(dim, 1, bias=True) 
        self.p_gen_decoder_output_layer = nn.Linear(dim, 1, bias=True) 
        self.p_gen_decoder_prev_output_layer = nn.Linear(dim, 1, bias=True) 

    def _compute_output_dist(
        self, 
        prefix_outputs,
        logits,
        prefix_ids,
    ):
        """
        Compute the output distribution with the copy mechanism 
        Args:
            prefix_outputs: (bsz, prefix_len, d_model) 
            logits: (bsz, seq_len, d_model) 
            prefix_ids: (bsz, prefix_len)
        """ 

        prefix_len = prefix_outputs.shape(1)
        bsz = prefix_outputs.shape(0) 
        seq_len = logits.shape(1) 
        d_model = prefix_outputs.shape(2)

        proj_prefix = prefix_outputs
        proj_seq = logits 

        sum_projs = torch.nn.GELU()(
            (proj_seq[:, :, None, :] + proj_prefix[:, None, :, :]).view(
                (bsz, seq_len, prefix_len, d_model)
            )
        )

        e = self.attn_layer(sum_projs).squeeze(-1) 
        attns = self._compute_cross_attn_prob(e) 
        context_vectors = torch.einsum("ijk, ikf -> ijf", attns, prefix_outputs) 

        # Compute p_vocab 
        p_vocab_context = self.model.to_logits(context_vectors) 
        p_vocab_seq = self.model.to_logits(logits) 
        p_vocab = p_vocab_context + p_vocab_seq 
        p_vocab = nn.Softmax(dim=-1)(p_vocab) 

        # Compute p_gen 
        p_gen_context = self.p_gen_context_layer(context_vectors) 
        p_gen_seq = self.p_gen_decoder_output_layer(logits) 
        p_gen_prev_seq = self.p_gen_decoder_prev_output_layer(self._shift_right_one_pad(logits)) 

        p_gen = nn.Sigmoid()(
            p_gen_context + p_gen_seq + p_gen_prev_seq
        ) 

        p_copy = torch.zeros_like(p_vocab)
        p_copy = p_copy.scatter_add(
            -1,
            prefix_ids.repeat_interleave(attns.shape[1], dim=0).view(
                bsz, seq_len, -1
            ),
            attns,
        )

        output_dist = torch.log((1.0 - p_gen) * p_copy + p_gen * p_vocab) 
        return output_dist 



    def _shift_right_one_pad(x): 
        shifted = x.roll(1) 
        shifted[0] = 0 
        return shifted


    def _compute_cross_attn_prob(self, e): 
        return nn.Softmax(dim=-1)(e)


    def forward(
        self,
        x,
        prefix_mask = None,
        labels = None,
    ):
        logits, prefix = self.model(x, prefix_mask, labels) 
        output_dist = self._compute_output_dist(prefix, logits, x[:, self.model.cross_attn_seq_len:])
        return output_dist 




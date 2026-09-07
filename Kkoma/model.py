import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        assert d_model % n_heads == 0, "d_model은 n_heads로 나누어 떨어져야 합니다."
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V 프로젝션 레이어
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Causal Mask (미래 토큰 가리기용 하삼각행렬) - 버퍼로 등록하여 GPU 이동 시 자동 포함
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time(Seq_len), Channel(d_model)
        
        # Q, K, V 계산 및 Multi-head 형태로 Reshape: (B, H, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, H, T, T)
        
        # Causal Mask 적용 (0인 부분을 -inf로 만들어 softmax 시 0%가 되도록)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Attention Weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Context Vector 계산 및 원래 모양으로 복구
        out = attn_weights @ v # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, max_seq_len):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model)
        )

    def forward(self, x):
        # Residual + LayerNorm 구조
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

class KkomaModel(nn.Module):
    def __init__(self, vocab_size, d_model=32, n_heads=4, n_layers=1, ffn_dim=128, max_seq_len=64):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, max_seq_len)
            for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        
        # 위치 인덱스 생성
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # 임베딩 합산
        x = self.tok_emb(idx) + self.pos_emb(pos)
        
        # 트랜스포머 블록 통과
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits

def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
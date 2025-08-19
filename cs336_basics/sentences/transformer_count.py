from cs336_basics.transformer.transformer_lm import Transformer
from cs336_basics.transformer.transformer import TransformerBlock
from cs336_basics.transformer.attention import Attention
from cs336_basics.transformer.swiglu import SwiGLU
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.linear import Linear

import torch
from fvcore.nn import flop_count
# uv run ./cs336_basics/sentences/transformer_count.py
""" 
vocab_size : 50,257
context_length : 1,024 -> 16,384
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400
"""
t = Transformer(vocab_size = 50257, context_length = 1024, d_model = 1600, num_layers = 48, num_heads = 25, d_ff = 6400, rope_theta = 10000.0, device = torch.device('cuda'))
# t = Transformer(vocab_size = 50257, context_length = 1024, d_model = 1600, num_layers = 1, num_heads = 25, d_ff = 6400, rope_theta = 10000.0)
print(t)
""" 
Transformer(
  (embedding): Embedding()
  (attn_layers): ModuleList(
    (0): TransformerBlock(
      (attn): Attention()
      (norm_attn): RMSNorm()
      (ffn): SwiGLU()
      (norm_ffn): RMSNorm()
    )
  )
  (rms_norm): RMSNorm()
  (lm_head): Linear()
)
"""
"""
                trainable_paras             
Embedding()     vocab_size * d_model        
attn_layer
- attn          num_heads * (4 * d_model * d_k) 
                =  4 * d_model * d_model       
- norm_attn     d_model
- ffn           3 * d_model * d_ff
- norm_ffn      d_model
rms_norm        d_model
lm_head         d_model * vocab_size
"""
"""
                FLOPs
Embedding()     0(only choose the correct line)
attn_layer
- attn          num_heads * (
                    x -> QKVO: 4 * 2 * seq_len * d_model * d_k
                    QK: 2 * seq_len * seq_len * d_k
                    div: seq_len * seq_len
                    softmax: seq_len * (3 * seq_len) # exp, add, div
                    V: seq_len * seq_len * d_k
                    O: seq_len * d_k * d_model # outside actually
                )
- norm_attn     seq_len * ( 
                    square: d_model
                    add, div, add(eps), sqrt(coff): 4 * d_model
                    div, mul: 2 * d_model
                )
- ffn           seq_len * ( 
                    weight: 2 * d_ff * d_model * 1
                    value: 2 * d_ff * d_model * 1
                    dot_mul: d_ff
                    up: 2 * d_model * d_ff * 1
                )
- norn_ffn      seq_len * 7 * d_model
rms_norm        seq_len * 7 * d_model
lm_head         2 * seq_len * d_model * vocab_size
"""

def count_parameters(model: Transformer):
    return [p.numel() for p in model.parameters() if p.requires_grad]
print(count_parameters(t))
print(sum(count_parameters(t)) * 2)

"""
                trainable_paras             
Embedding       vocab_size * d_model = 50257 * 1600 = 80411200

attn_layer      total = 48 * (4 * 10240000 + 2 * 1600) 
                      = 48 * 40963200 = 1,966,233,600
- attn          num_heads * (4 * d_model * d_k) 
                = 4 * d_model * d_model
                = 4 * 1600 * 1600
                = 4 * 2560000
                = 10240000
- norm_attn     d_model = 1600
- ffn           3 * d_model * d_ff
                = 3 * 1600 * 6400
                = 3 * 10240000
- norm_ffn      d_model = 1600

rms_norm        d_model = 1600
lm_head         d_model * vocab_size = 1600 * 50257 = 80411200
"""

# Int[Tensor, " batch_size sequence_length"]
# print(flop_count(t, torch.ones((1, 1024), dtype=torch.int)))

"""
                FLOPs
Embedding()     0(only choose the correct line)
attn_layer      tot = 48 * (25 * 1,107,296,256 + 62,914,560,000)
                    = 48 * 95,839,846,400
                    = 1,580,413,747,200
- attn          num_heads * (
                    x -> QKV: 3 * 2 * seq_len * d_model * d_k
                            = 3 * 2 * 1024 * 1600 * 64 # d_k = 1600 / 25
                            = 629,145,600
                    QK: 2 * seq_len * d_k * seq_len
                      = 2 * 1024 * 64 * 1024
                      = 134,217,728
                    div: seq_len * seq_len
                    softmax: seq_len * (3 * seq_len) # exp, add, div
                    V: 2 * seq_len * seq_len * d_k
                     = 2 * 1024 * 1024 * 64 = 134,217,728
                    O: 2 * seq_len * d_k * d_model # outside actually
                     = 2 * 1024 * 64 * 1600 = 209,715,200
                )
- norm_attn     seq_len * ( 
                    square: d_model
                    sum, div, add(eps), sqrt(coff): 4 * d_model
                    div, mul: 2 * d_model
                )
- ffn           seq_len * ( 
                    weight: 2 * d_ff * d_model * 1
                          = 2 * 6400 * 1600
                          = 20,480,000
                    silu(mul, sigmoid): 
                    value: 2 * d_ff * d_model * 1 = 20,480,000
                    dot_mul: d_ff
                    up: 2 * d_model * d_ff * 1 = 20,480,000
                )
- norn_ffn      seq_len * 7 * d_model
rms_norm        seq_len * 7 * d_model
lm_head         2 * seq_len * d_model * vocab_size
              = 2 * 1024 * 1600 * 50257 = 164,682,137,600
"""
# 27,682,406,400 + 20,480,000 * 3 * 1024 + 164,682,137,600 = 255,279,104,000
# 127.6465 Gflops (127,639,552,000, mul and add together is one)
"""
'aten::div': 8,         norm(3) * 2(coff, res) + attn(1) * 2(QK, softmax)
'aten::mul': 8,         norm(3) #? attn(3) + ffn::dot_mul ffn::silu(1)
'aten::add': 5,         norm(3) #? ffn(2)
'aten::sum': 4,         norm(3) + attn::softmax(1)
'aten::square': 3,      norm
'aten::sqrt': 3,        norm
'aten::triu': 1,        attn::softmax(mask)
'aten::where': 1,       attn::mask
'aten::sub': 1,         attn::softmax
'aten::exp': 1,         attn::softmax
'aten::mul_': 1,        attn #?
'aten::sigmoid': 1      ffn::silu
"""

# Float[Tensor, " batch sequence_length d_model"]
t = TransformerBlock(d_model = 1600, num_heads = 25, d_ff = 6400, max_seq_len = 1024, theta = 10000.0)
print(flop_count(t, torch.ones((1, 1024, 1600))))
# (27,682,406,400 + 20,480,000 * 3 * 1,024) = 90,596,966,400
# 90,596,966,400 -> 45,298,483,200  45.2965 Gflops

t = Attention(d_model = 1600, num_heads = 25, d_k = 64, d_v = 64)
print(flop_count(t, torch.ones((1, 1024, 1600))))
# 27,682,406,400 -> 13,841,203,200  13.8415 Gflops

t = SwiGLU(d_model = 1600, d_ff = 6400)
print(flop_count(t, torch.ones((1, 1024, 1600))))
# 62,914,560,000 -> 31,457,280,000  31.455 Gflops

t = RMSNorm(d_model = 1600)
print(flop_count(t, torch.ones((1, 1024, 1600))))
# 0 Gflops

t = Embedding(num_embeddings = 50257, embedding_dim = 1600)
print(flop_count(t, torch.ones((1, 1024), dtype=torch.int)))
# 0 Gflops

t = Linear(in_features = 1600, out_features = 50257)
print(flop_count(t, torch.ones((1, 1024, 1600))))
# 164,682,137,600 -> 82,341,068,800 82.35 Gflops
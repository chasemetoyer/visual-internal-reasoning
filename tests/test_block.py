"""Regression test for the residual wiring of Block in src/train_imaginer.py.

Run with plain Python (or pytest, if installed):
    python tests/test_block.py
"""
import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_imaginer import Block, GPTConfig

def test_attention_output_reaches_residual_stream():
    # With the MLP's output zeroed, a pre-LN block reduces to x + attn(ln1(x)). The old wiring,
    # x + mlp(ln2(x + attn(ln1(x)))), returned x unchanged: attention never reached the stream.
    torch.manual_seed(0)
    block = Block(GPTConfig(vocab_size=16, block_size=8, n_layer=1, n_head=2, n_embd=16)).eval()
    with torch.no_grad():
        block.mlp[-1].weight.zero_()
        x = torch.randn(2, 8, 16)
        out = block(x)
        assert not torch.allclose(out, x), "Block returned its input: the attention output is lost"
        assert torch.allclose(out, x + block.attn(block.ln1(x)), atol=1e-6), "Block is not pre-LN"

if __name__ == "__main__":
    test_attention_output_reaches_residual_stream()
    print("ok: attention output reaches the residual stream")

import pytest
import torch
from protmyth.modules.row_attention import RowSelfAttention, ColumnSelfAttention

# Fixtures to initialize the RowSelfAttention and ColumnSelfAttention modules
@pytest.fixture
def row_attention():
    return RowSelfAttention(
        q_dim=64,
        c=32,
        n_head=4,
        out_dim=128,
        max_tokens_per_msa=1024,
        dropout=0.1,
        use_bias=False,
        gating=True
    )

@pytest.fixture
def column_attention():
    return ColumnSelfAttention(
        q_dim=64,
        c=32,
        n_head=4,
        out_dim=128,
        max_tokens_per_msa=1024,
        dropout=0.1,
        use_bias=False,
        gating=True
    )

# Test the forward pass for RowSelfAttention
def test_row_attention_forward(row_attention):
    x = torch.randn(1, 512, 64)  # Input tensor of shape (batch_size, num_rows, q_dim)
    output = row_attention(x)
    assert output.shape == (1, 512, 128), f"Expected shape (1, 512, 128), got {output.shape}"

# Test the batched forward pass for large inputs in RowSelfAttention
def test_row_attention_batched_forward(row_attention):
    x = torch.randn(1, 2048, 64)  # Input tensor exceeding max_tokens_per_msa
    output = row_attention(x)
    assert output.shape == (1, 2048, 128), f"Expected shape (1, 2048, 128), got {output.shape}"

# Test the forward pass for ColumnSelfAttention
def test_column_attention_forward(column_attention):
    x = torch.randn(1, 512, 64)  # Input tensor of shape (batch_size, num_columns, q_dim)
    output = column_attention(x)
    assert output.shape == (1, 512, 128), f"Expected shape (1, 512, 128), got {output.shape}"

# Test the batched forward pass for large inputs in ColumnSelfAttention
def test_column_attention_batched_forward(column_attention):
    x = torch.randn(1, 2048, 64)  # Input tensor exceeding max_tokens_per_msa
    output = column_attention(x)
    assert output.shape == (1, 2048, 128), f"Expected shape (1, 2048, 128), got {output.shape}"

# Test the make_graph function for RowSelfAttention
def test_row_attention_make_graph(row_attention):
    graph = row_attention.make_graph(batch_dims=[1], q_len=512, device=torch.device('cpu'))
    assert isinstance(graph, torchviz.Dot), "Expected a torchviz.Dot object"

# Test the make_graph function for ColumnSelfAttention
def test_column_attention_make_graph(column_attention):
    graph = column_attention.make_graph(batch_dims=[1], c_len=512, device=torch.device('cpu'))
    assert isinstance(graph, torchviz.Dot), "Expected a torchviz.Dot object"

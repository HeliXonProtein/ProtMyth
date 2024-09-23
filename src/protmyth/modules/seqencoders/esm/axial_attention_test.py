import pytest
import torch
import einops
import torchviz
from graphviz import Digraph
from protmyth.modules.seqencoders.esm.axial_attention import RowSelfAttention, ColumnSelfAttention


"""Module for testing RowSelfAttention and ColumnSelfAttention.

This module contains PyTest fixtures for initializing and testing
the RowSelfAttention and ColumnSelfAttention modules from the 
protmyth library.
"""

@pytest.fixture
def row_attention():
    """Fixture to initialize the RowSelfAttention module.

    Returns:
        RowSelfAttention: An instance of the RowSelfAttention class 
        with predefined parameters for testing.
    """
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
    """Fixture to initialize the ColumnSelfAttention module.

    Returns:
        ColumnSelfAttention: An instance of the ColumnSelfAttention class 
        with predefined parameters for testing.
    """
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

def test_row_attention_forward(row_attention):
    """Test the forward pass of the RowSelfAttention module.

    This test checks whether the output of RowSelfAttention has the expected shape
    when provided with a small input tensor.

    Args:
        row_attention (RowSelfAttention): The RowSelfAttention module initialized via the fixture.

    Raises:
        AssertionError: If the output shape is not as expected.
    """
    x = torch.randn(1, 512, 64)  # Input tensor of shape (batch_size, num_rows, q_dim)
    output = row_attention(x)
    assert output.shape == (1, 512, 128), f"Expected shape (1, 512, 128), got {output.shape}"


def test_row_attention_batched_forward(row_attention):
    """Test the batched forward pass of RowSelfAttention with large inputs.

    This test verifies that RowSelfAttention can handle inputs exceeding
    max_tokens_per_msa, and checks whether the output shape is as expected.

    Args:
        row_attention (RowSelfAttention): The RowSelfAttention module initialized via the fixture.

    Raises:
        AssertionError: If the output shape is not as expected.
    """
    x = torch.randn(1, 2048, 64)  # Input tensor exceeding max_tokens_per_msa
    output = row_attention(x)
    assert output.shape == (1, 2048, 128), f"Expected shape (1, 2048, 128), got {output.shape}"


def test_column_attention_forward(column_attention):
    """Test the forward pass of the ColumnSelfAttention module.

    This test checks whether the output of ColumnSelfAttention has the expected shape
    when provided with a small input tensor.

    Args:
        column_attention (ColumnSelfAttention): The ColumnSelfAttention module initialized via the fixture.

    Raises:
        AssertionError: If the output shape is not as expected.
    """
    x = torch.randn(1, 512, 64)  # Input tensor of shape (batch_size, num_columns, q_dim)
    output = column_attention(x)
    assert output.shape == (1, 512, 128), f"Expected shape (1, 512, 128), got {output.shape}"


def test_column_attention_batched_forward(column_attention):
    """Test the batched forward pass of ColumnSelfAttention with large inputs.

    This test verifies that ColumnSelfAttention can handle inputs exceeding
    max_tokens_per_msa, and checks whether the output shape is as expected.

    Args:
        column_attention (ColumnSelfAttention): The ColumnSelfAttention module initialized via the fixture.

    Raises:
        AssertionError: If the output shape is not as expected.
    """
    x = torch.randn(1, 2048, 64)  # Input tensor exceeding max_tokens_per_msa
    output = column_attention(x)
    assert output.shape == (1, 2048, 128), f"Expected shape (1, 2048, 128), got {output.shape}"


def test_row_attention_make_graph(row_attention):
    """Test the make_graph function for RowSelfAttention.

    This test checks whether the make_graph method produces a valid graph object using torchviz.

    Args:
        row_attention (RowSelfAttention): The RowSelfAttention module initialized via the fixture.

    Raises:
        AssertionError: If the generated graph is not an instance of torchviz.Dot.
    """
    graph = row_attention.make_graph(batch_dims=[1], q_len=512, device=torch.device('cpu'))
    assert isinstance(graph, torchviz.Dot), "Expected a torchviz.Dot object"


def test_column_attention_make_graph(column_attention):
    """Test the make_graph function for ColumnSelfAttention.

    This test checks whether the make_graph method produces a valid graph object using torchviz.

    Args:
        column_attention (ColumnSelfAttention): The ColumnSelfAttention module initialized via the fixture.

    Raises:
        AssertionError: If the generated graph is not an instance of torchviz.Dot.
    """
    graph = column_attention.make_graph(batch_dims=[1], c_len=512, device=torch.device('cpu'))
    assert isinstance(graph, torchviz.Dot), "Expected a torchviz.Dot object"


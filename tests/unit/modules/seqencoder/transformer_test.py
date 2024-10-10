import torch
import pytest
from esm_layer_norm import ESM1LayerNorm, ESM1bLayerNorm  # assuming esm_layer_norm is the filename
from torchviz import Digraph


# Helper function to check if tensor is normalized
def check_normalization(tensor: torch.Tensor, eps: float = 1e-12):
    mean = tensor.mean()
    var = tensor.var(unbiased=False)
    assert torch.isclose(mean, torch.tensor(0.0), atol=1e-5), f"Mean is not close to 0, got {mean.item()}"
    assert torch.isclose(var, torch.tensor(1.0), atol=1e-5), f"Variance is not close to 1, got {var.item()}"


@pytest.mark.parametrize("LayerNormClass", [ESM1LayerNorm, ESM1bLayerNorm])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("hidden_size", [128, (64, 128)])
def test_forward(LayerNormClass, affine, hidden_size):
    # Initialize layer norm
    layer_norm = LayerNormClass(hidden_size=hidden_size, affine=affine)

    # Create a sample input tensor
    batch_size = 4
    seq_len = 10
    if isinstance(hidden_size, tuple):
        x = torch.randn(batch_size, seq_len, *hidden_size)
    else:
        x = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    output = layer_norm(x)

    # Check output shape
    assert output.shape == x.shape, f"Expected shape {x.shape}, but got {output.shape}"

    # Check normalization (mean ~ 0, variance ~ 1)
    check_normalization(output)

    # If affine, check that weights and biases are applied correctly
    if affine:
        assert layer_norm.weight is not None, "Expected weight to be initialized"
        assert layer_norm.bias is not None, "Expected bias to be initialized"


@pytest.mark.parametrize("LayerNormClass", [ESM1LayerNorm, ESM1bLayerNorm])
@pytest.mark.parametrize("batch_dims", [(2, 10), (4, 8)])
@pytest.mark.parametrize("hidden_size", [128, (64, 128)])
def test_make_graph(LayerNormClass, batch_dims, hidden_size):
    # Initialize layer norm
    layer_norm = LayerNormClass(hidden_size=hidden_size)

    # Create a graph using make_graph
    device = torch.device("cpu")
    graph = layer_norm.make_graph(batch_dims=batch_dims, device=device)

    # Check if the returned object is a Digraph
    assert isinstance(graph, Digraph), f"Expected output to be a Digraph, but got {type(graph)}"


if __name__ == "__main__":
    pytest.main()


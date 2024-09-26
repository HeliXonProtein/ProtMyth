    def __init__(self, q_dim: int, c: int, n_head: int, out_dim: int,
                 max_tokens_per_msa: int = 1024, dropout: float = 0.1,
                 use_bias: bool = False, gating: bool = True) -> None:
        """Initializes the RowSelfAttention module, reusing most of the logic from the parent class.

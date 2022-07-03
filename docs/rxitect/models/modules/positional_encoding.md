Module rxitect.models.modules.positional_encoding
=================================================

Classes
-------

`PositionalEncoding(d_model: int, dropout: float = 0.1, max_len: int = 5000)`
:   Positional Encoding impl.
    
    from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
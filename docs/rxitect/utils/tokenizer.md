Module rxitect.utils.tokenizer
==============================

Classes
-------

`Tokenizer(pad_token: str = ' ', start_token: str = '[SOS]', end_token: str = '[EOS]', missing_token: str = '[UNK]', vocabulary: Optional[List[str]] = None)`
:   Abstract base class for tokenizers, defining the interface that can subsequently be used in
    dataset/datamodules/models.

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * rxitect.utils.smiles.SmilesTokenizer

    ### Static methods

    `detokenize(tokenized_mol_string: List[str]) ‑> str`
    :

    `tokenize(mol_string: str) ‑> List[str]`
    :

    ### Methods

    `decode(self, mol_tensor: torch.Tensor) ‑> str`
    :

    `encode(self, mol_string: str) ‑> torch.Tensor`
    :

    `fit(self, mol_strings: List[str]) ‑> None`
    :

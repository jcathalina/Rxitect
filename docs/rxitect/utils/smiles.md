Module rxitect.utils.smiles
===========================

Functions
---------


`clean_and_canonalize(smiles: str) ‑> str`
:   Removes charges and canonalizes the SMILES representation of a molecule.

    Args:
        smiles (str): SMILES string representation of a molecule.
    Returns:
        Cleaned (uncharged version of largest fragment) & Canonicalized SMILES,
        or empty string on invalid Mol.


`randomize_smiles(smiles: str) ‑> str`
:

Classes
-------

`SmilesTokenizer()`
:   Abstract base class for tokenizers, defining the interface that can subsequently be used in
    dataset/datamodules/models.

    ### Ancestors (in MRO)

    * rxitect.utils.tokenizer.Tokenizer
    * abc.ABC

    ### Static methods

    `detokenize(tokenized_mol_string: List[str]) ‑> str`
    :   Takes an array of indices and returns the corresponding SMILES.

    `tokenize(mol_string: str) ‑> List[str]`
    :   Method that takes the string representation of a molecule (in this case, SMILES) and
        returns a list of tokens that the string is made up of.

        Args:
            mol_string (str): The SMILES representation of a molecule
        Returns:
            A list of tokens that make up the passed SMILES.

    ### Methods

    `batch_encode(self, mol_strings: List[str]) ‑> torch.Tensor`
    :

    `decode(self, mol_tensor: torch.Tensor) ‑> str`
    :   Takes an array of indices and returns the corresponding SMILES.

    `encode(self, mol_string: str) ‑> torch.Tensor`
    :   Takes a list containing tokens from the passed SMILES (eg '[NH]') and encodes to array
        of indices.

    `fit(self, mol_strings: List[str]) ‑> None`
    :

    `fit_from_file(self, vocabulary_filepath: pathlib.Path) ‑> None`
    :

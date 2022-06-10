Module rxitect.data.datasets
============================

Functions
---------

    
`smiles_to_fingerprint(smiles: str) ‑> numpy.ndarray`
:   Helper function that transforms SMILES strings into
    the enhanced 2067D-Fingerprint representation used for training in Rxitect.
    If only a single SMILES was passed, will return a single array containing
    its fingerprint.
    
    Args:
        smiles: A list of SMILES representations of molecules.

Classes
-------

`PyTorchQSARDataset(ligand_file: str, target_chembl_id: str, transform: Callable = None)`
:   An abstract class representing a :class:`Dataset`.
    
    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.
    
    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic
Module rxitect.datasets.smiles_dataset
======================================

Classes
-------

`SmilesDataset(data: pandas.core.frame.DataFrame, tokenizer: rxitect.utils.smiles.SmilesTokenizer)`
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

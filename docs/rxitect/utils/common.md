Module rxitect.utils.common
===========================

Functions
---------

    
`flatten_iterable(iterable: Iterable[+T_co]) ‑> Iterable[+T_co]`
:   

Classes
-------

`BlockDataLoader(dataset: torch.utils.data.dataset.Dataset, batch_size: int = 100, block_size: int = 10000, shuffle: bool = True, n_workers: int = 0, pin_memory: bool = True)`
:   Main `DataLoader` class which has been modified so as to read training data from disk in
    blocks, as opposed to a single line at a time (as is done in the original `DataLoader` class).
    
    From: https://github.com/MolecularAI/GraphINVENT/

    ### Ancestors (in MRO)

    * torch.utils.data.dataloader.DataLoader
    * typing.Generic

    ### Class variables

    `batch_size: Optional[int]`
    :

    `dataset: torch.utils.data.dataset.Dataset[+T_co]`
    :

    `drop_last: bool`
    :

    `num_workers: int`
    :

    `pin_memory: bool`
    :

    `prefetch_factor: int`
    :

    `sampler: Union[torch.utils.data.sampler.Sampler, Iterable[+T_co]]`
    :

    `timeout: float`
    :

`BlockDataset(dataset: torch.utils.data.dataset.Dataset, batch_size: int = 100, block_size: int = 10000)`
:   Modified `Dataset` class which returns BLOCKS of data when `__getitem__()` is called.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

`ShuffleBlockWrapper(data: torch.Tensor)`
:   Wrapper class used to wrap a block of data, enabling data to get shuffled.
    
    *within* a block.
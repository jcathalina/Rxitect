Module rxitect.datamodules.smiles_datamodule
============================================

Classes
-------

`SmilesDataModule(data_filepath: pathlib.Path, train_val_test_split: Tuple[int, int, int], augment: bool = False, batch_size: int = 128, num_workers: int = 0, npartitions: Optional[int] = None, pin_memory: bool = False, random_state: int = 42)`
:   A DataModule standardizes the training, val, test splits, data preparation and transforms. The main
    advantage is consistent data splits, data preparation and transforms across models.
    
    Example::
    
        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self, stage):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)
            def teardown(self):
                # clean up after fit or test
                # called on every process in DDP

    ### Ancestors (in MRO)

    * pytorch_lightning.core.datamodule.LightningDataModule
    * pytorch_lightning.core.hooks.CheckpointHooks
    * pytorch_lightning.core.hooks.DataHooks
    * pytorch_lightning.core.mixins.hparams_mixin.HyperparametersMixin

    ### Class variables

    `name: str`
    :

    ### Methods

    `custom_collate_and_pad(self, batch: List[torch.Tensor]) ‑> List[torch.Tensor]`
    :   Args:
            batch (List[str]): A list of vectorized smiles.
        
        Returns:
            A list containing the padded versions of the tensors that were passed in.

    `prepare_data(self) ‑> None`
    :   Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
        settings) will result in corrupted data. Lightning ensures this method is called only within a single
        process, so you can safely add your downloading logic within.
        
        .. warning:: DO NOT set state to the model (use ``setup`` instead)
            since this is NOT called on every device
        
        Example::
        
            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()
        
                # bad
                self.split = data_split
                self.some_state = some_other_state()
        
        In DDP ``prepare_data`` can be called in two ways (using Trainer(prepare_data_per_node)):
        
        1. Once per node. This is the default and is only called on LOCAL_RANK=0.
        2. Once in total. Only called on GLOBAL_RANK=0.
        
        See :ref:`prepare_data_per_node<common/lightning_module:prepare_data_per_node>`.
        
        Example::
        
            # DEFAULT
            # called once per node on LOCAL_RANK=0 of that node
            Trainer(prepare_data_per_node=True)
        
            # call on GLOBAL_RANK=0 (great for shared file systems)
            Trainer(prepare_data_per_node=False)
        
        This is called before requesting the dataloaders:
        
        .. code-block:: python
        
            model.prepare_data()
            initialize_distributed()
            model.setup(stage)
            model.train_dataloader()
            model.val_dataloader()
            model.test_dataloader()

    `setup(self, stage: Optional[str] = None) ‑> None`
    :   Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when
        you need to build models dynamically or adjust something about them. This hook is called on every process
        when using DDP.
        
        Args:
            stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``
        
        Example::
        
            class LitModel(...):
                def __init__(self):
                    self.l1 = None
        
                def prepare_data(self):
                    download_data()
                    tokenize()
        
                    # don't do this
                    self.something = else
        
                def setup(self, stage):
                    data = load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)

    `test_dataloader(self) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader]]`
    :   Implement one or multiple PyTorch DataLoaders for testing.
        
        For data processing use the following pattern:
        
            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`
        
        However, the above are only necessary for distributed processing.
        
        .. warning:: do not assign state in prepare_data
        
        
        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`
        - :meth:`prepare_data`
        - :meth:`setup`
        
        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.
        
        Return:
            A :class:`torch.utils.data.DataLoader` or a sequence of them specifying testing samples.
        
        Example::
        
            def test_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False, transform=transform,
                                download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=False
                )
        
                return loader
        
            # can also return multiple dataloaders
            def test_dataloader(self):
                return [loader_a, loader_b, ..., loader_n]
        
        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.
        
        Note:
            In the case where you return multiple test dataloaders, the :meth:`test_step`
            will have an argument ``dataloader_idx`` which matches the order here.

    `train_dataloader(self) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader], Sequence[Sequence[torch.utils.data.dataloader.DataLoader]], Sequence[Dict[str, torch.utils.data.dataloader.DataLoader]], Dict[str, torch.utils.data.dataloader.DataLoader], Dict[str, Dict[str, torch.utils.data.dataloader.DataLoader]], Dict[str, Sequence[torch.utils.data.dataloader.DataLoader]]]`
    :   Implement one or more PyTorch DataLoaders for training.
        
        Return:
            A collection of :class:`torch.utils.data.DataLoader` specifying training samples.
            In the case of multiple dataloaders, please see this :ref:`section <multiple-dataloaders>`.
        
        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.
        
        For data processing use the following pattern:
        
            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`
        
        However, the above are only necessary for distributed processing.
        
        .. warning:: do not assign state in prepare_data
        
        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`prepare_data`
        - :meth:`setup`
        
        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.
        
        Example::
        
            # single dataloader
            def train_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=True, transform=transform,
                                download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=True
                )
                return loader
        
            # multiple dataloaders, return as list
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a list of tensors: [batch_mnist, batch_cifar]
                return [mnist_loader, cifar_loader]
        
            # multiple dataloader, return as dict
            def train_dataloader(self):
                mnist = MNIST(...)
                cifar = CIFAR(...)
                mnist_loader = torch.utils.data.DataLoader(
                    dataset=mnist, batch_size=self.batch_size, shuffle=True
                )
                cifar_loader = torch.utils.data.DataLoader(
                    dataset=cifar, batch_size=self.batch_size, shuffle=True
                )
                # each batch will be a dict of tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
                return {'mnist': mnist_loader, 'cifar': cifar_loader}

    `val_dataloader(self) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader]]`
    :   Implement one or multiple PyTorch DataLoaders for validation.
        
        The dataloader you return will not be reloaded unless you set
        :paramref:`~pytorch_lightning.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.
        
        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.
        
        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
        - :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`
        - :meth:`prepare_data`
        - :meth:`setup`
        
        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.
        
        Return:
            A :class:`torch.utils.data.DataLoader` or a sequence of them specifying validation samples.
        
        Examples::
        
            def val_dataloader(self):
                transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (1.0,))])
                dataset = MNIST(root='/path/to/mnist/', train=False,
                                transform=transform, download=True)
                loader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    shuffle=False
                )
        
                return loader
        
            # can also return multiple dataloaders
            def val_dataloader(self):
                return [loader_a, loader_b, ..., loader_n]
        
        Note:
            If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
            implement this method.
        
        Note:
            In the case where you return multiple validation dataloaders, the :meth:`validation_step`
            will have an argument ``dataloader_idx`` which matches the order here.
Module rxitect.utils
====================

Sub-modules
-----------
* rxitect.utils.common
* rxitect.utils.smiles
* rxitect.utils.tokenizer

Functions
---------

    
`extras(config: omegaconf.dictconfig.DictConfig) ‑> None`
:   Applies optional utilities, controlled by config flags.
    Utilities:
    - Ignoring python warnings
    - Rich config printing

    
`finish(config: omegaconf.dictconfig.DictConfig, model: pytorch_lightning.core.lightning.LightningModule, datamodule: pytorch_lightning.core.datamodule.LightningDataModule, trainer: pytorch_lightning.trainer.trainer.Trainer, callbacks: List[pytorch_lightning.callbacks.base.Callback], logger: List[pytorch_lightning.loggers.base.LightningLoggerBase]) ‑> None`
:   Makes sure everything closed properly.

    
`get_logger(name='rxitect.utils') ‑> logging.Logger`
:   Initializes multi-GPU-friendly python command line logger.

    
`log_hyperparameters(config: omegaconf.dictconfig.DictConfig, model: pytorch_lightning.core.lightning.LightningModule, datamodule: pytorch_lightning.core.datamodule.LightningDataModule, trainer: pytorch_lightning.trainer.trainer.Trainer, callbacks: List[pytorch_lightning.callbacks.base.Callback], logger: List[pytorch_lightning.loggers.base.LightningLoggerBase]) ‑> None`
:   Controls which config parts are saved by Lightning loggers. Additionaly saves:
    
    - number of model parameters

    
`print_config(config: omegaconf.dictconfig.DictConfig, print_order: Sequence[str] = ('datamodule', 'model', 'callbacks', 'logger', 'trainer'), resolve: bool = True) ‑> None`
:   Prints content of DictConfig using Rich library and its tree structure.
    
    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order
                                               config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
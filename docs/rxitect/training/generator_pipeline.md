Module rxitect.training.generator_pipeline
==========================================

Functions
---------

    
`train(config: omegaconf.dictconfig.DictConfig) ‑> Optional[float]`
:   Contains the training pipeline.
    
    Can additionally evaluate model on a testset, using best
    weights achieved during training.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
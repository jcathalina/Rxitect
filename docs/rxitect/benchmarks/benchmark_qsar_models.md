Module rxitect.benchmarks.benchmark_qsar_models
===============================================
ABSOLUTELY NOT DONE YET.

Functions
---------

    
`cross_validate_svr(dataset: pandas.core.frame.DataFrame, n_splits: int, random_state: int, out_dir: str, target: str)`
:   

    
`date_label() ‑> str`
:   Helper function that returns the formatted datetime as a convenient
    label for document name creation.
    
    Returns:
        The formatted datetime string

    
`kfold_cv_benchmark(model: Union[sklearn.ensemble._forest.RandomForestRegressor, xgboost.sklearn.XGBRegressor], dataset: rxitect.data.utils.QSARDataset, k: int, out_dir: str, target: str, scoring: List[str] = ['r2', 'neg_root_mean_squared_error'], n_jobs: int = -1) ‑> None`
:   

    
`main(cfg: omegaconf.dictconfig.DictConfig) ‑> None`
:   Function to benchmark QSAR model(s)

    
`safe_mkdir(dir_path: str) ‑> None`
:   A helper function that allows you to cleanly create a directory if
    it does not exist yet.
    
    Args:
        dir_path: The path to the directory that would have to be created
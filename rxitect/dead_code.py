"""File containing dead code snippets that may be useful as reference later"""

# def smiles_to_fingerprint(smiles: str) -> np.ndarray:
#     """
#     Helper function that transforms SMILES strings into
#     the enhanced 2067D-Fingerprint representation used for training in Rxitect.
#     If only a single SMILES was passed, will return a single array containing
#     its fingerprint.

#     Args:
#         smiles: A list of SMILES representations of molecules.
#     """
#     fingerprint: np.ndarray = calc_single_fp(smiles, accept_smiles=True)
#     fingerprint = torch.from_numpy(fingerprint.astype(np.float32))
#     return fingerprint


# class MultiTargetTorchQSARDataset(Dataset):
#     def __init__(
#         self,
#         ligand_file: str,
#         target_chembl_id: str,
#         transform: Callable = None,
#     ) -> None:
#         self.pchembl_values: pd.DataFrame = (
#             pd.read_csv(ligand_file, usecols=["smiles", target_chembl_id])
#             .dropna()
#             .reset_index(drop=True)
#         )
#         self.transform: Callable = transform

#     def __len__(self) -> int:
#         return len(self.pchembl_values)

#     def __getitem__(self, index) -> Tuple[ArrayLike, float]:
#         smiles = self.pchembl_values.iloc[index, 0]
#         pchembl_value = self.pchembl_values.iloc[index, 1].astype(np.float32)
#         if self.transform:
#             smiles = self.transform(smiles)
#         return smiles, pchembl_value


# if __name__ == "__main__":
#     test_data = MultiTargetTorchQSARDataset(
#         ligand_file=abspath("data/processed/ligand_test_splityear=2015.csv"),
#         target_chembl_id="CHEMBL226",
#         transform=smiles_to_fingerprint,
#     )

#     train_data = MultiTargetTorchQSARDataset(
#         ligand_file=abspath("data/processed/ligand_train_splityear=2015.csv"),
#         target_chembl_id="CHEMBL226",
#         transform=smiles_to_fingerprint,
#     )

#     X_train, y_train = train_data[:]
#     print(X_train.shape)


# @dataclass
# class MultiTargetQSARDataset:
#     """Class representing the dataset used to train QSAR models"""

#     df_train: pd.DataFrame
#     df_test: pd.DataFrame
#     targets: List[str]
#     _X_train: ArrayDict = field(init=False)
#     _X_test: ArrayDict = field(init=False)
#     _y_train: ArrayDict = field(init=False)
#     _y_test: ArrayDict = field(init=False)

#     def __post_init__(self) -> None:
#         """Initializes the train and test data based on a temporal split in the data to be used for QSAR model fitting."""
#         self._X_train = {k: np.array([]) for k in self.targets}
#         self._X_test = {k: np.array([]) for k in self.targets}
#         self._y_train = {k: np.array([]) for k in self.targets}
#         self._y_test = {k: np.array([]) for k in self.targets}

#     def get_train_test_data(
#         self, target_chembl_id: str
#     ) -> Tuple[np.ndarray, ...]:
#         """ """
#         return (
#             self.X_train(target_chembl_id),
#             self.y_train(target_chembl_id),
#             self.X_test(target_chembl_id),
#             self.y_test(target_chembl_id),
#         )

#     def X_train(self, target_chembl_id: str) -> np.ndarray:
#         """Lazily evaluates the train data points for a given target ChEMBL ID

#         Args:
#             target_chembl_id:

#         Returns:
#             An array containing the fingerprints of all train data points for the given target ChEMBL ID
#         """
#         if not self._X_train[target_chembl_id].size:
#             data = self.df_train.dropna(subset=[target_chembl_id])["smiles"]
#             self._X_train[target_chembl_id] = calc_fp(data, accept_smiles=True)
#         return self._X_train[target_chembl_id]

#     def X_test(self, target_chembl_id: str) -> np.ndarray:
#         """Lazily evaluates the test data points for a given target ChEMBL ID

#         Args:
#             target_chembl_id:

#         Returns:
#             An array containing the fingerprints of all test data points for the given target ChEMBL ID
#         """
#         if not self._X_test[target_chembl_id].size:
#             data = self.df_test.dropna(subset=[target_chembl_id])["smiles"]
#             self._X_test[target_chembl_id] = calc_fp(data, accept_smiles=True)
#         return self._X_test[target_chembl_id]

#     def y_train(self, target_chembl_id: str) -> np.ndarray:
#         """Lazily evaluates the train labels for a given target ChEMBL ID

#         Args:
#             target_chembl_id:

#         Returns:
#             An array containing the pChEMBL value of all train data points for the given target ChEMBL ID
#         """
#         if not self._y_train[target_chembl_id].size:
#             data = self.df_train[target_chembl_id].dropna().values
#             self._y_train[target_chembl_id] = data
#         return self._y_train[target_chembl_id]

#     def y_test(self, target_chembl_id: str) -> np.ndarray:
#         """Lazily evaluates the test labels for a given target ChEMBL ID

#         Args:
#             target_chembl_id:

#         Returns:
#             An array containing the pChEMBL value of all test data points for the given target ChEMBL ID
#         """
#         if not self._y_test[target_chembl_id].size:
#             data = self.df_test[target_chembl_id].dropna().values
#             self._y_test[target_chembl_id] = data
#         return self._y_test[target_chembl_id]

#     def get_classifier_labels(
#         self, target_chembl_id: str
#     ) -> Tuple[ArrayLike, ArrayLike]:
#         """ """
#         y_train_clf = np.where(
#             self.y_train(target_chembl_id) > 6.5, 1, 0
#         )  # TODO: Make 6.5 thresh a const
#         y_test_clf = np.where(self.y_test(target_chembl_id) > 6.5, 1, 0)

#         return y_train_clf, y_test_clf

#     @classmethod
#     def load_from_file(cls, train_file: str, test_file: str) -> MultiTargetQSARDataset:
#         """ """
#         df_train = pd.read_csv(train_file)
#         df_test = pd.read_csv(test_file)
#         targets = [
#             target for target in df_train.columns[1:].values
#         ]  # TODO: Assert that targets are equal for train and test

#         return MultiTargetQSARDataset(df_train, df_test, targets)


# def construct_qsar_dataset(
#     raw_data_path: str,
#     targets: List[str],
#     cols: List[str],
#     px_placeholder: float = 3.99,
#     random_state: int = 42,
#     tsplit_year: Optional[int] = None,
#     negative_samples: bool = True,
#     out_dir: Optional[str] = None,
# ) -> QSARDataset:
#     """Method that constructs a dataset from ChEMBL data to train QSAR regression models on,
#     using a temporal split to create a hold-out test dataset for evaluation.

#     Args:
#         raw_data_path: filepath of the raw data
#         targets: ChEMBL IDs that are relevant for the dataset creation
#         cols: relevant columns for current dataset creation
#         px_placeholder: pChEMBL value to use for negative examples
#         random_state: seed integer to ensure reproducibility when randomness is involved
#         tsplit_year (optional): year at which the temporal split should happen to create the held-out test set
#         negative_samples: boolean flag that determines if negative samples should be included, default is True
#         out_dur (optional): filepath where the processed data should be saved to, default is None

#     Returns:
#         A QSARDataset object - a convenient abstraction.
#     """
#     # Load and standardize raw data
#     df = pd.read_csv(raw_data_path, sep="\t")
#     df.columns = df.columns.str.lower()
#     df.dropna(subset=["smiles"], inplace=True)

#     # Filter data to only contain relevant targets
#     df = df[df["target_chembl_id"].isin(targets)]

#     # Create temporal split for hold-out test set creation downstream
#     smiles_by_year = df.groupby("smiles")["document_year"].min().dropna()
#     smiles_by_year = smiles_by_year.astype("Int16")
#     tsplit_test_idx = smiles_by_year[smiles_by_year > 2015].index

#     # Re-index data to divide SMILES per target
#     df = df[cols].set_index(["target_chembl_id", "smiles"])

#     # Process positive examples from data, taking the mean of duplicates and removing missing entries
#     pos_samples = (
#         df["pchembl_value"]
#         .groupby(["target_chembl_id", "smiles"])
#         .mean()
#         .dropna()
#     )

#     df_processed = pos_samples
#     if negative_samples:
#         # Process negative examples from data, setting their default pChEMBL values to some threshold (default 3.99)
#         # Looks for where inhibition or no activity are implied
#         comments = df[(df["comment"].str.contains("Not Active") == True)]
#         inhibitions = df[
#             (df["standard_type"] == "Inhibition")
#             & df["standard_relation"].isin(["<", "<="])
#         ]
#         relations = df[
#             df["standard_type"].isin(["EC50", "IC50", "Kd", "Ki"])
#             & df["standard_relation"].isin([">", ">="])
#         ]
#         neg_samples = pd.concat([comments, inhibitions, relations])
#         # Ensure only true negative samples remain in the negative sample set
#         neg_samples = neg_samples[~neg_samples.index.isin(pos_samples.index)]
#         neg_samples["pchembl_value"] = px_placeholder
#         neg_samples = (
#             neg_samples["pchembl_value"]
#             .groupby(["target_chembl_id", "smiles"])
#             .first()
#         )  # Regroup indices
#         df_processed = pd.concat([pos_samples, neg_samples])

#     df_processed = df_processed.unstack("target_chembl_id")

#     if tsplit_year:
#         idx_test = list(set(df_processed.index).intersection(tsplit_test_idx))
#         df_test = df_processed.loc[idx_test]
#         df_train = df_processed.drop(df_test.index)
#         file_suffix = f"splityear={tsplit_year}.csv"
#     else:
#         df_train, df_test = train_test_split(
#             df_processed, test_size=0.2, random_state=random_state
#         )
#         file_suffix = f"seed={random_state}.csv"

#     qsar_dataset = QSARDataset(
#         df_train=df_train.reset_index(drop=False),
#         df_test=df_test.reset_index(drop=False),
#         targets=targets,
#     )

#     if out_dir:
#         df_test.to_csv(
#             os.path.join(out_dir, f"ligand_test_{file_suffix}"), index=True
#         )
#         df_train.to_csv(
#             os.path.join(out_dir, f"ligand_train_{file_suffix}"), index=True
#         )

#     return qsar_dataset

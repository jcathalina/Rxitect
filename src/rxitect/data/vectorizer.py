from typing import Iterable, Optional
import numpy as np
import selfies as sf

from abc import ABC, abstractmethod
from rdkit import Chem

from rxitect.utils.types import RDKitMol
from rxitect.chem.utils import mol_from_selfies, mol_to_selfies, smiles_to_rdkit_mol


class Vectorizer(ABC):
    """Abstract class for (de)vectorization, with support for SMILES/SELFIES enumeration (atom order randomization)
    as data augmentation.
    Originally from Esben Jannik Bjerrum's molvecgen @ https://github.com/EBjerrum/molvecgen/blob/master/molvecgen/vectorizers.py

    :parameter vocab: string containing the characters for the vectorization. Can also be generated via the .fit() method
    :parameter pad: Length of the vectorization
    :parameter left_pad: Add spaces to the left of the SMILES
    :parameter isomeric_smiles: Generate SMILES containing information about stereogenic centers
    :parameter augment: Enumerate the SMILES during transform
    :parameter canonical: use canonical SMILES during transform (overrides enum)
    :parameter binary: Use RDKit binary strings instead of molecule objects
    """

    def __init__(
        self,
        vocab: Optional[str] = None,
        pad=10,
        max_length=128,
        left_pad=True,
        isomeric_smiles=True,
        augment=True,
        canonical=False,
        start_token="^",
        end_token="$",
        unknown_token="?",
    ):
        # Checks
        if augment and canonical:
            raise Exception("Can't have both augmentation and canonalization on at the same time.")
            # TODO: Make custom exception for this scenario, e.g. ParamException

        # Special Characters
        self.start_token = start_token
        self.end_token = end_token
        self.unknown_token = unknown_token

        # Vectorization and SMILES options
        self.left_pad = left_pad
        self.isomeric_smiles = isomeric_smiles
        self.augment = augment
        self.canonical = canonical
        self._pad = pad
        self._max_length = max_length

        # The characterset
        self._vocab = None
        self.vocab = vocab

        # Calculate the dimensions
        self.setdims()

    @property
    def vocab(self):
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        # Ensure start and endchars are in the vocab
        for char in [self.start_token, self.end_token, self.unknown_token]:
            if char not in vocab:
                vocab = vocab + char
        # Set the hidden properties
        self._vocab = vocab
        self._vocab_size = len(vocab)
        self._char_to_int = dict((c, i) for i, c in enumerate(vocab))
        self._int_to_char = dict((i, c) for i, c in enumerate(vocab))
        self.setdims()

    @property
    def max_length(self):
        return self._max_length

    @max_length.setter
    def max_length(self, max_length):
        self._max_length = max_length
        self.setdims()

    @property
    def pad(self):
        return self._pad

    @pad.setter
    def pad(self, pad):
        self._pad = pad
        self.setdims()

    def setdims(self):
        """Calculates and sets the output dimensions of the vectorized molecules from the current settings"""
        self.dims = (self.max_length + self.pad, self._vocab_size)

    @abstractmethod
    def fit(self, mols: Iterable[RDKitMol], extra_chars: Iterable[str] = []) -> None:
        """Performs extraction of the vocab and length of a SMILES datasets and sets self.max_length and self.vocab

        :parameter smiles: Numpy array or Pandas series containing smiles as strings
        :parameter extra_chars: List of extra chars to add to the vocab (e.g. "\\\\" when "/" is present)
        """
        pass

    def randomize_mol(self, mol: RDKitMol):
        """Performs a randomization of the atom order of an RDKit molecule"""
        ans = list(range(mol.GetNumAtoms()))
        np.random.shuffle(ans)
        return Chem.RenumberAtoms(mol, ans)

    @abstractmethod
    def transform(self, mol: RDKitMol, augment: bool = None, canonical: bool = None) -> np.ndarray:
        """Perform an enumeration (atom order randomization) and vectorization of a Numpy array of RDkit molecules

        :parameter mol: The RDKit molecule to transform
        :parameter augment: Override the objects .augment setting
        :parameter canonical: Override the objects .canonical setting

        :output: Numpy array with the vectorized molecules with shape [batch, max_length+pad, vocab]
        """
        pass
    
    @abstractmethod
    def reverse_transform(self, vect: np.ndarray, strip: bool = True) -> str:
        """Performs a conversion of a vectorized string representation of molecule.
        vocab must be the same as used for vectorization.

        :parameter vect: Numpy array of vectorized SMILES.
        :parameter strip: Strip start and end tokens from the SMILES string
        """
        pass


class SelfiesVectorizer(Vectorizer):
    """"""
    def __init__(self,
                 start_token: str = "[SOS]",
                 end_token: str = "[EOS]",
                 unknown_token: str = "[UNK]",
                 ) -> None:
        super().__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unknown_token

    def fit(self, selfies_list: Iterable[str], extra_chars: Iterable[str] = [".", "[nop]"]) -> None:
        vocab = sf.get_alphabet_from_selfies(selfies_list)
        self.vocab = "".join(vocab.union(set(extra_chars)))
        self.max_length = max([len(selfies) for selfies in selfies_list])

    def reverse_transform(self, vect: np.ndarray, strip: bool = True) -> str:
        selfies = sf.encoding_to_selfies(encoding=vect, vocab_itos=self._int_to_char, enc_type="one_hot")
        if strip:
            selfies = selfies.strip(self.start_token, self.end_token)
        return selfies

    def transform(self, mol: RDKitMol, augment: bool = None, canonical: bool = None) -> np.ndarray:
        """Perform an enumeration (atom order randomization) and vectorization of a Numpy array of RDkit molecules

        :parameter mol: The RDKit molecule to transform
        :parameter augment: Override the objects .augment setting
        :parameter canonical: Override the objects .canonical setting

        :output: Numpy array with the vectorized molecules with shape [batch, max_length+pad, vocab]
        """
        augment = augment if augment else self.augment
        canonical = canonical if canonical else self.canonical

        if augment:
            mol = self.randomize_mol(mol)

        selfies = mol_to_selfies(mol=mol, canonical=canonical, isomeric_smiles=self.isomeric_smiles)
        one_hot = sf.selfies_to_encoding(selfie=selfies, vocab_stoi=self._char_to_int, pad_to_len=self.pad, enc_type="one_hot")
        return one_hot

    def randomize_selfies(self, selfies: str):
        """Perform a randomization of a SELFIES string
        must be RDKit sanitizable"""
        
        mol = mol_from_selfies(selfies)
        nmol = self.randomize_mol(mol)
        smiles = Chem.MolToSmiles(nmol, canonical=self.canonical, isomeric_smiles=self.isomeric_smiles)
        selfies = sf.encoder(smiles)
        return selfies


if __name__ == "__main__":
    sv = SelfiesVectorizer()
    print(sv.__dict__)
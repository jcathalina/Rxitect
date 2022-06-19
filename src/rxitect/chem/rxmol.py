import selfies as sf
from rdkit import Chem

from rxitect.chem.utils import calc_single_fp
from rxitect.utils.exceptions import RxMolException
from rxitect.utils.types import Fingerprint, Optional, RDKitMol


class RxMol:
    """A base class for molecules that encapsulates an RDKit Mol object and
    extends it with functions that can be applied to said molecules.

    Args:
        rdkit_mol: the RDKit Mol object that is being encapsulated
        smiles: the SMILES representation of the molecule
        selfies: the SELFIES representation of the molecule
        sanitize: if True, the molecule will be immediately sanitized, defaults to False

    Raises:
        RxMolException: if no valid representation for a molecule is passed to the ctor, or if the molecule could not be sanitized
    """

    def __init__(
        self,
        rdkit_mol: RDKitMol = None,
        smiles: str = None,
        sanitize: bool = False,
    ) -> None:
        if not rdkit_mol and not smiles:
            raise RxMolException(
                "Please pass an RDKit Mol object, SMILES or SELFIES string."
            )

        if rdkit_mol:
            self.rdkit_mol = rdkit_mol
            self.smiles = Chem.MolToSmiles(rdkit_mol)

        else:
            self.smiles = smiles
            self.rdkit_mol = Chem.MolFromSmiles(smiles, sanitize=False)

        self._selfies: Optional[str] = None
        self._fingerprint: Fingerprint = {}
        self._is_sanitized: bool = False

        if sanitize:
            self.sanitize()

    @property
    def selfies(self) -> str:
        """The SELFIES representation of the molecule, created by lazy evaluation.

        Returns:
            the SELFIES representation of the molecule.
        """
        if not self._selfies:
            self._selfies = sf.encoder(self.smiles)
        return self._selfies

    @property
    def fingerprint(self) -> Fingerprint:
        """The 2067 dimensional fingerprint representing a molecule created from
        the ECFP + 19 Physicochemical Properties, created by lazy evaluation.

        Returns:
            the Fingerprint representation of the molecule.
        """
        if not self._fingerprint:
            fp_array = calc_single_fp(mol=self.rdkit_mol, accept_smiles=False)
            self._fingerprint = fp_array
        return self._fingerprint


def rx_mol_from_selfies(selfies: str) -> RxMol:
    """Creates an RxMol object from a SELFIES string.

    Args:
        selfies: the SELFIES representation of the molecule.

    Returns:
        an RxMol object.
    """
    smiles = sf.decoder(selfies)
    rx_mol = RxMol(smiles=smiles)
    return rx_mol

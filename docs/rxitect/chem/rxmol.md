Module rxitect.chem.rxmol
=========================

Functions
---------

    
`rxmol_from_selfies(selfies: str) ‑> rxitect.chem.rxmol.RxMol`
:   Creates an RxMol object from a SELFIES string.
    
    Args:
        selfies: the SELFIES representation of the molecule.
    
    Returns:
        an RxMol object.

Classes
-------

`RxMol(rdkit_mol: rdkit.Chem.rdchem.Mol = None, smiles: str = None, sanitize: bool = False)`
:   A base class for molecules that encapsulates an RDKit Mol object and
    extends it with functions that can be applied to said molecules.
    
    Args:
        rdkit_mol: the RDKit Mol object that is being encapsulated
        smiles: the SMILES representation of the molecule
        selfies: the SELFIES representation of the molecule
        sanitize: if True, the molecule will be immediately sanitized, defaults to False
    
    Raises:
        RxMolException: if no valid representation for a molecule is passed to the ctor, or if the molecule could not be sanitized

    ### Instance variables

    `fingerprint: Dict[Union[Tuple[int, int], Tuple[int]], numpy.ndarray]`
    :   TODO

    `selfies: str`
    :   The SELFIES representation of the molecule, created by lazy evaluation.
        
        Returns:
            the SELFIES representation of the molecule.
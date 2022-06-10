Module rxitect.chem.utils
=========================

Functions
---------

    
`calc_fp(mols: Union[List[rdkit.Chem.rdchem.Mol], List[str]], radius: int = 3, bit_len: int = 2048, accept_smiles: bool = False) ‑> Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]`
:   

    
`calc_single_fp(mol: Union[rdkit.Chem.rdchem.Mol, str], radius: int = 3, bit_len: int = 2048, accept_smiles: bool = True) ‑> Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]`
:   

    
`smiles_to_rdkit_mol(smiles_list: List[str]) ‑> List[rdkit.Chem.rdchem.Mol]`
:   Helper function to convert a list of SMILES to RDKit Mol objects
    
    Args:
        smiles: List of SMILES representations of molecules
    
    Returns:
        A list of RDKit Mol objects created from the given SMILES
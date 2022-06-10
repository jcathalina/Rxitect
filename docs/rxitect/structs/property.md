Module rxitect.structs.property
===============================

Functions
---------

    
`calc_prop(mols: List[rdkit.Chem.rdchem.Mol], prop: str) ‑> numpy.ndarray`
:   Calculates the value of a molecular property for a batch of molecules.
    
    Args:
        mols: A list of RDKit Mol objects.
        prop: The dictionary key for the property to be calculated.
    
    Returns:
        an array of scores for each molecule given.

Classes
-------

`Property(value, names=None, *, module=None, qualname=None, type=None, start=1)`
:   An enumeration.

    ### Ancestors (in MRO)

    * builtins.str
    * enum.Enum

    ### Class variables

    `AliphaticRings`
    :

    `AmideBonds`
    :

    `AromaticRings`
    :

    `BertzComplexity`
    :

    `BridgeheadAtoms`
    :

    `CrippenMolMR`
    :

    `FCSP3`
    :

    `HBA`
    :

    `HBD`
    :

    `HeavyAtoms`
    :

    `HeteroAtoms`
    :

    `Heterocycles`
    :

    `LogP`
    :

    `MolecularWeight`
    :

    `QED`
    :

    `RingCount`
    :

    `RotatableBonds`
    :

    `SaturatedRings`
    :

    `SpiroAtoms`
    :

    `SyntheticAccessibility`
    :

    `TPSA`
    :

    `ValenceElectrons`
    :
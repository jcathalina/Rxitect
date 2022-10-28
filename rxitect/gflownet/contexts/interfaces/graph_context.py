from abc import ABC, abstractmethod
from typing import Tuple, List

import torch
from networkx import Graph
from rdkit.Chem import Mol
from torch_geometric.data import Data, Batch

from rxitect.gflownet.utils.graph import GraphAction


class IGraphContext(ABC):
    """A context class defines what the graphs are, how they map to and from data"""
    device: torch.device

    @abstractmethod
    def aidx_to_GraphAction(self, g: Data, action_idx: Tuple[int, int, int]) -> GraphAction:
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction
        Parameters
        ----------
        g: Data
            The graph to which the action is being applied
        action_idx: Tuple[int, int, int]
            The tensor indices for the corresponding action
        Returns
        -------
        action: GraphAction
            A graph action that could be applied to the original graph corresponding to g.
        """
        pass

    @abstractmethod
    def GraphAction_to_aidx(self, g: Data, action: GraphAction) -> Tuple[int, int, int]:
        """Translate a GraphAction to an action index (e.g. from a GraphActionCategorical)
        Parameters
        ----------
        g: Data
            The graph to which the action is being applied
        action: GraphAction
            A graph action that could be applied to the original graph corresponding to g.
        Returns
        -------
        action_idx: Tuple[int, int, int]
            The tensor indices for the corresponding action
        """
        pass

    @abstractmethod
    def graph_to_Data(self, g: Graph) -> Data:
        """Convert a networkx Graph to a torch geometric Data instance
        Parameters
        ----------
        g: Graph
            A graph instance.
        Returns
        -------
        torch_g: gd.Data
            The corresponding torch_geometric graph.
        """
        pass

    @classmethod
    @abstractmethod
    def collate(cls, graphs: List[Data]) -> Batch:
        """Convert a list of torch geometric Data instances to a Batch
        instance.  This exists so that environment contexts can set
        custom batching attributes, e.g. by using `follow_batch`.
        Parameters
        ----------
        graphs: List[Data]
            Graph instances

        Returns
        -------
        batch: gd.Batch
            The corresponding batch.
        """
        pass

    @abstractmethod
    def is_sane(self, g: Graph) -> bool:
        """Verifies whether a graph is sane according to the context. This can
        catch impossible molecules.
        Parameters
        ----------
        g: Graph
            A graph.

        Returns
        -------
        is_sane: bool:
            True if the environment considers g to be sane.
        """
        pass

    @abstractmethod
    def mol_to_graph(self, mol: Mol) -> Graph:
        """Verifies whether a graph is sane according to the context. This can
        catch, e.g. impossible molecules.
        Parameters
        ----------
        mol: Mol
            An RDKit molecule

        Returns
        -------
        g: Graph
            The corresponding Graph representation of that molecule.
        """
        pass

    @abstractmethod
    def graph_to_mol(self, g: Graph) -> Mol:
        """Verifies whether a graph is sane according to the context. This can
        catch, e.g. impossible molecules.
        Parameters
        ----------
        g:
            A networkx Graph instance

        Returns
        -------
        mol: Mol
            The corresponding Graph representation of that molecule.
        """
        pass

"""
Multiversal Computing Module
Real parallel computation using multiple solution pathways
"""

from .protein_folding_engine import ProteinFoldingEngine, AminoAcid, ProteinStructure
from .multiversal_protein_computer import MultiversalProteinComputer, UniverseResult
from .virtual_cell import VirtualCellEnvironment, VirtualCell, CellState

__all__ = [
    'ProteinFoldingEngine',
    'AminoAcid',
    'ProteinStructure',
    'MultiversalProteinComputer',
    'UniverseResult',
    'VirtualCellEnvironment',
    'VirtualCell',
    'CellState'
]

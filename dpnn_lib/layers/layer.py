import torch.nn as nn
from typing import List
from dpnn_lib.cells.cell import DistributionCell

class DistributionLayer(nn.Module):
    def __init__(self, cells: List[DistributionCell]):
        super().__init__()
        self.cells = nn.ModuleList(cells)

    def forward(self):
        # For each cell, propagate its distribution considering other cells in the same layer as neighbors
        for i, cell in enumerate(self.cells):
            # Create a list of neighbors excluding the current cell
            neighbors = [other_cell for j, other_cell in enumerate(self.cells) if i != j]
            new_distribution = cell.forward(neighbors)
            cell.distribution = new_distribution

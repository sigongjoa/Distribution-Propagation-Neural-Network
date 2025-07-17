import torch.nn as nn
from typing import List
from dpnn_lib.cells.cell import DistributionCell
from dpnn_lib.distributions.base import BaseDistribution

class DistributionLayer(nn.Module):
    def __init__(self, cells: List[DistributionCell]):
        super().__init__()
        self.cells = nn.ModuleList(cells)

    def forward(self, input_distributions: List[BaseDistribution]) -> List[BaseDistribution]:
        if len(input_distributions) != len(self.cells):
            raise ValueError("Number of input distributions must match number of cells in the layer.")

        output_distributions = []
        for i, cell_module in enumerate(self.cells): # Renamed 'cell' to 'cell_module' to avoid confusion
            current_distribution = input_distributions[i]
            
            # Prepare neighbor distributions for the current cell
            neighbor_distributions = []
            for j, other_cell_module in enumerate(self.cells):
                if i != j:
                    neighbor_distributions.append(input_distributions[j]) # Pass neighbor's input distribution

            new_distribution = cell_module.forward(current_distribution, neighbor_distributions)
            output_distributions.append(new_distribution)
        return output_distributions

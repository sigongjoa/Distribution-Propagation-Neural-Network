import torch.nn as nn
from typing import List
from dpnn_lib.cells.cell import DistributionCell
from dpnn_lib.distributions.base import BaseDistribution

class DistributionLayer(nn.Module):
    """
    분포 셀들로 구성된 단일 레이어입니다.

    Args:
        cells (List[DistributionCell]): 레이어를 구성하는 분포 셀의 리스트.
    """
    def __init__(self, cells: List[DistributionCell]):
        super().__init__()
        self.cells = nn.ModuleList(cells)

    def forward(self, input_distributions: List[BaseDistribution]) -> List[BaseDistribution]:
        """
        입력 분포를 각 셀에 전파하고 결과를 집계합니다.

        Args:
            input_distributions (List[BaseDistribution]): 레이어에 입력될 분포의 리스트.

        Returns:
            List[BaseDistribution]: 레이어의 출력 분포 리스트.
        """
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

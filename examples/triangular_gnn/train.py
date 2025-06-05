#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ------------------------------------------------------------------------------
"""Training loop for the GNN edge scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from examples.triangular_gnn.graph_builder import build_graph
from examples.triangular_gnn.ingestion import PriceSnapshot
from examples.triangular_gnn.model import EdgeScorer


def load_snapshots(path: Path) -> Iterable[PriceSnapshot]:
    """Load snapshots from CSV for training."""
    # TODO: implement real loading logic
    raise NotImplementedError


def train(model: EdgeScorer, snapshots: Iterable[PriceSnapshot], epochs: int = 1) -> None:
    """Train the ``EdgeScorer`` on historical snapshots."""
    graphs = [build_graph([snap]) for snap in snapshots]
    loader = DataLoader(graphs, batch_size=32)
    optim = Adam(model.parameters())
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for graph in loader:
            optim.zero_grad()
            pred = model(graph)
            # TODO: replace with real labels
            target = torch.zeros_like(pred)
            loss = loss_fn(pred, target)
            loss.backward()
            optim.step()



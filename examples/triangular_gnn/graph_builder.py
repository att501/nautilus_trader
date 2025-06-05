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
"""Build graphs from price snapshots."""

from __future__ import annotations

from typing import Iterable

import torch
from torch_geometric.data import Data

from examples.triangular_gnn.ingestion import PriceSnapshot


def build_graph(snapshots: Iterable[PriceSnapshot]) -> Data:
    """Convert snapshots to a ``torch_geometric`` graph.

    Parameters
    ----------
    snapshots : Iterable[PriceSnapshot]
        Stream of price snapshots.
    """
    # Placeholder implementation using a fully connected graph
    # Nodes are currencies extracted from pair strings
    currencies = {}
    edges_src = []
    edges_dst = []
    edge_attr = []

    for snap in snapshots:
        base, quote = snap.pair.split("/")
        for cur in (base, quote):
            if cur not in currencies:
                currencies[cur] = len(currencies)
        src = currencies[base]
        dst = currencies[quote]
        edges_src.append(src)
        edges_dst.append(dst)
        # Use mid price as edge weight
        mid = (snap.bid + snap.ask) / 2.0
        edge_attr.append([mid])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    num_nodes = len(currencies)

    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)


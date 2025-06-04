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
"""Strategy for trading triangular arbitrage using a GNN model."""

from __future__ import annotations

from typing import List

import torch

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

from examples.triangular_gnn.graph_builder import build_graph
from examples.triangular_gnn.model import EdgeScorer
from examples.triangular_gnn.ingestion import PriceSnapshot


class GNNArbConfig(StrategyConfig, frozen=True):
    """Configuration for ``GNNArbStrategy`` instances."""

    instrument_ids: List[InstrumentId]


class GNNArbStrategy(Strategy):
    """Triangular arbitrage strategy powered by a GNN edge scorer."""

    def __init__(self, config: GNNArbConfig, model: EdgeScorer) -> None:
        super().__init__(config)
        self.model = model

    def on_start(self) -> None:
        for iid in self.config.instrument_ids:
            self.subscribe_quote_ticks(iid)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        snapshot = PriceSnapshot(
            pair=tick.instrument_id.symbol,
            bid=float(tick.bid_price or 0.0),
            ask=float(tick.ask_price or 0.0),
            timestamp=int(tick.ts_event_ns),
        )
        graph = build_graph([snapshot])
        with torch.no_grad():
            score = self.model(graph)
        self.log.info(f"edge score {score}", LogColor.CYAN)
        # TODO: implement trade execution based on score


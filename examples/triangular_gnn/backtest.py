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
"""Backtesting loop for the GNN arbitrage strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.config import BacktestNodeConfig
from nautilus_trader.model.identifiers import TraderId

from examples.triangular_gnn.model import EdgeScorer
from examples.triangular_gnn.strategy import GNNArbConfig, GNNArbStrategy
from examples.triangular_gnn.train import load_snapshots


def run_backtest(data_path: Path, instrument_ids: list[str]) -> None:
    """Run a backtest using snapshots from ``data_path``."""
    model = EdgeScorer()
    snapshots = load_snapshots(data_path)
    config = BacktestNodeConfig(trader_id=TraderId("BACKTESTER"))
    node = BacktestNode(config)
    strategy = GNNArbStrategy(GNNArbConfig(instrument_ids=[]), model)
    node.trader.add_strategy(strategy)
    node.build()
    node.run()


if __name__ == "__main__":
    run_backtest(Path("data.csv"), [])


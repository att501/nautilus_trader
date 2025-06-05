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
"""Example live execution for the GNN arbitrage strategy."""

from __future__ import annotations

from nautilus_trader.adapters.coinbase_intx import COINBASE_INTX
from nautilus_trader.adapters.coinbase_intx import CoinbaseIntxLiveDataClientFactory
from nautilus_trader.adapters.coinbase_intx import CoinbaseIntxLiveExecClientFactory
from nautilus_trader.cache.config import CacheConfig
from nautilus_trader.config import InstrumentProviderConfig, LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig, TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId

from examples.triangular_gnn.model import EdgeScorer
from examples.triangular_gnn.strategy import GNNArbConfig, GNNArbStrategy


def run_live() -> None:
    """Start a live trading node running the strategy."""
    config_node = TradingNodeConfig(
        trader_id=TraderId("LIVE"),
        logging=LoggingConfig(log_level="INFO", use_pyo3=True),
        exec_engine=LiveExecEngineConfig(),
        cache=CacheConfig(encoding="msgpack"),
        data_clients={
            COINBASE_INTX: InstrumentProviderConfig(load_all=True)
        },
    )
    node = TradingNode(config=config_node)
    model = EdgeScorer()
    strategy = GNNArbStrategy(GNNArbConfig(instrument_ids=[]), model)
    node.trader.add_strategy(strategy)
    node.add_data_client_factory(COINBASE_INTX, CoinbaseIntxLiveDataClientFactory)
    node.add_exec_client_factory(COINBASE_INTX, CoinbaseIntxLiveExecClientFactory)
    node.build()
    try:
        node.run()
    finally:
        node.dispose()


if __name__ == "__main__":
    run_live()


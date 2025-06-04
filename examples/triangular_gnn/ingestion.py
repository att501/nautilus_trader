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
"""Data ingestion utilities for triangular arbitrage graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from nautilus_trader.model.data import QuoteTick


def stream_ticks(pairs: List[str]) -> Iterable[QuoteTick]:
    """Yield tick data for the given trading pairs.

    This example uses a placeholder implementation. Replace with logic to
    connect to your desired venue (e.g. Coinbase WebSocket) and yield
    ``QuoteTick`` objects.
    """
    # TODO: Implement live WebSocket connections
    raise NotImplementedError("Tick streaming not implemented")


@dataclass(slots=True)
class PriceSnapshot:
    """Simple snapshot of prices for building graphs."""

    pair: str
    bid: float
    ask: float
    timestamp: int



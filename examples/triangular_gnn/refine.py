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
"""Simple refinement loop for retraining and evaluating the model."""

from __future__ import annotations

from pathlib import Path

from examples.triangular_gnn.backtest import run_backtest
from examples.triangular_gnn.model import EdgeScorer
from examples.triangular_gnn.train import load_snapshots, train


def refine(data_path: Path, iterations: int = 3) -> None:
    """Retrain the model and backtest repeatedly."""
    model = EdgeScorer()
    snapshots = load_snapshots(data_path)
    for _ in range(iterations):
        train(model, snapshots)
        run_backtest(data_path, [])


if __name__ == "__main__":
    refine(Path("data.csv"))


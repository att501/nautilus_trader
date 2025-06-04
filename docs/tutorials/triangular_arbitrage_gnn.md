# GNN-Driven Triangular Arbitrage

This guide outlines one way to leverage **NautilusTrader** to implement a high frequency
triangular arbitrage strategy powered by graph neural networks (GNNs). The example
focuses on Kraken or Coinbase but the workflow can be adapted to other venues.

## 1. Environment Setup

1. Install the package from source or `pip install nautilus-trader`.
2. Create a Python 3.11+ virtual environment.
3. Install extra dependencies for PyTorch and `torch_geometric` for GNN support.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

## 2. Ingesting Market Data

Use NautilusTrader adapters or your own scripts to stream live prices:

- **Coinbase**: see [`docs/integrations/coinbase_intx.md`](../integrations/coinbase_intx.md) for
  details on configuring API keys and streaming order book data.
- **Kraken**: a Kraken adapter is not included, so you will need to build one or
  connect via WebSockets directly.

Store price snapshots for a selection of trading pairs that form triangular
loops (for example `XBT/USD`, `XBT/EUR`, `EUR/USD`).

## 3. Building Graph Snapshots

Convert each snapshot to a directed weighted graph where nodes represent
currencies and edges represent conversion rates. Persist these graphs as
`torch_geometric.data.Data` objects or CSV files. The process is similar to
building bar datasets as shown in other tutorials.

## 4. Training a GNN Edge Scorer

1. Assemble historical graph snapshots and label edges that produced profitable
   triangular arbitrage cycles.
2. Train a custom PyTorch or `torch_geometric` model to score each edge.
3. Export the trained model so it can be loaded by your trading script.

## 5. Implementing the Trader Agent

Create a Python class that loads your model and interacts with
`nautilus_trader` objects. At a high level:

```python
from nautilus_trader.core.datetime import dt_to_unix_nanos

class GNNTrader:
    def __init__(self, model):
        self.model = model

    def on_market_data(self, snapshot):
        graph = convert_snapshot_to_graph(snapshot)
        scores = self.model(graph)
        cycle = find_best_triangular_cycle(scores)
        if cycle.return_ratio > 1.001:  # 0.1% threshold
            submit_orders_for_cycle(cycle)
```

Integrate this agent within a `BacktestEngine` or a live `TradingNode`
instance to either simulate or execute real trades.

## 6. Backtesting and Deployment

- Use the backtest examples (`examples/backtest`) as templates.
- Feed historical snapshots into your trader agent to verify performance.
- When satisfied, configure the appropriate venue adapter and run in live mode.

This approach combines NautilusTrader's robust trading infrastructure with a
GNN-based model for detecting triangular arbitrage opportunities.

## 7. Example Module

The repository includes a reference implementation under
`examples/triangular_gnn/`. This module contains:

- Data ingestion utilities (`ingestion.py`)
- Graph construction helpers (`graph_builder.py`)
- A simple GNN model (`model.py`)
- Training and refinement loops (`train.py`, `refine.py`)
- Backtesting and live execution scripts (`backtest.py`, `execute.py`)

Use these files as a starting point for building your own workflow.

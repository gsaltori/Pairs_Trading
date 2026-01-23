# TRADING SYSTEM ARCHITECTURE

## EXECUTION FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN LOOP (Every Bar)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. DATA LAYER (MT5)                                 │
│  ┌─────────────────┐    ┌─────────────────┐                                 │
│  │  EURUSD H4 Bar  │    │  GBPUSD H4 Bar  │                                 │
│  │  O, H, L, C     │    │  O, H, L, C     │                                 │
│  └────────┬────────┘    └────────┬────────┘                                 │
└───────────┼──────────────────────┼──────────────────────────────────────────┘
            │                      │
            ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      2. UPDATE EQUITY & CHECK POSITIONS                      │
│                                                                              │
│    Account Equity ──► Risk Engine ──► Update Drawdown                       │
│    Check MT5 Positions ──► Detect SL/TP Hits ──► Update State               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         3. SIGNAL ENGINE                                     │
│                                                                              │
│    EURUSD Bar ──► EMA(50), EMA(200), ATR(14)                                │
│                         │                                                    │
│                         ▼                                                    │
│    ┌─────────────────────────────────────────────────────────┐              │
│    │  IF Close > EMA200 AND pullback to EMA50 ──► LONG      │              │
│    │  IF Close < EMA200 AND pullback to EMA50 ──► SHORT     │              │
│    │  ELSE ──► NO SIGNAL                                     │              │
│    └─────────────────────────────────────────────────────────┘              │
│                         │                                                    │
│                         ▼                                                    │
│    Signal: Direction, Entry, SL (ATR×1.5), TP (RR=2.0)                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          │     SIGNAL EXISTS?    │
                          └───────────┬───────────┘
                                      │ YES
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     4. RISK ENGINE CHECK                                     │
│                                                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │  CHECK 1: System halted? ─────────────────────────► BLOCK        │     │
│    │  CHECK 2: Max concurrent trades? ─────────────────► BLOCK        │     │
│    │  CHECK 3: Max concurrent risk? ───────────────────► BLOCK        │     │
│    │  CHECK 4: Too soon after last trade? ─────────────► BLOCK        │     │
│    │  CHECK 5: Daily loss limit hit? ──────────────────► BLOCK        │     │
│    │  CHECK 6: DD >= 8%? ──────────────────────────────► HALT         │     │
│    │  ALL PASSED ──────────────────────────────────────► ALLOWED      │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ ALLOWED
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     5. GATEKEEPER ENGINE                                     │
│                                                                              │
│    EURUSD + GBPUSD ──► Compute Observables:                                 │
│                        • Z-score (spread)                                    │
│                        • Correlation                                         │
│                        • Correlation Trend                                   │
│                        • Volatility Ratio                                    │
│                                                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │  RULE 1: |Z-score| > 3.0? ────────────────────────► BLOCK        │     │
│    │  RULE 2: Corr Trend < -0.05? ─────────────────────► BLOCK        │     │
│    │  RULE 3: Vol Ratio < 0.7? ────────────────────────► BLOCK        │     │
│    │  ALL PASSED ──────────────────────────────────────► ALLOWED      │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│    ⚠️  EMPIRICALLY VALIDATED: Blocked trades have 6.4% lower win rate       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ ALLOWED
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     6. POSITION SIZING                                       │
│                                                                              │
│    Risk Per Trade = Equity × Risk% (0.5% normal, 0.25% reduced)             │
│    SL Distance = |Entry - SL|                                                │
│    Position Size = Risk Amount / SL Distance                                 │
│                                                                              │
│    Constraints:                                                              │
│    • Min: 0.01 lots                                                          │
│    • Max: 1.0 lots                                                           │
│    • Round to lot step                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     7. EXECUTION ENGINE (MT5)                                │
│                                                                              │
│    ┌──────────────────────────────────────────────────────────────────┐     │
│    │  Build Order Request:                                             │     │
│    │  • Symbol: EURUSD                                                 │     │
│    │  • Volume: Calculated                                             │     │
│    │  • Type: BUY/SELL                                                 │     │
│    │  • SL: REQUIRED ✓                                                 │     │
│    │  • TP: REQUIRED ✓                                                 │     │
│    │  • Magic: 20260117                                                │     │
│    │  • Slippage: 30 points max                                        │     │
│    └──────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│    Execute with retry (max 3 attempts)                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     8. LOGGING & PERSISTENCE                                 │
│                                                                              │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│    │ trades.csv  │  │ blocks.csv  │  │ risk_state  │  │ system.log  │      │
│    │ Every trade │  │ Every gate  │  │  .csv       │  │ All events  │      │
│    │ attempt     │  │ decision    │  │ Periodic    │  │             │      │
│    └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │
│                                                                              │
│    State persisted to: system_state.json                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## DRAWDOWN GOVERNOR STATE MACHINE

```
                    ┌──────────────────────────────────────────────┐
                    │                  NORMAL                       │
                    │  Risk: 0.5% per trade                        │
                    │  Max trades: 2                                │
                    └──────────────────┬───────────────────────────┘
                                       │ DD >= 3%
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │                 REDUCED                       │
                    │  Risk: 0.25% per trade                       │
                    │  Max trades: 2                                │
                    └──────────────────┬───────────────────────────┘
                                       │ DD >= 6%
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │               SINGLE_TRADE                    │
                    │  Risk: 0.25% per trade                       │
                    │  Max trades: 1                                │
                    └──────────────────┬───────────────────────────┘
                                       │ DD >= 8%
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │                 HALTED                        │
                    │  NO NEW TRADES                                │
                    │  Existing positions managed by broker SL/TP  │
                    └──────────────────┬───────────────────────────┘
                                       │ DD >= 10%
                                       ▼
                    ┌──────────────────────────────────────────────┐
                    │             MANUAL_REVIEW                     │
                    │  REQUIRES HUMAN INTERVENTION                  │
                    │  System will not auto-resume                  │
                    └──────────────────────────────────────────────┘

    Recovery: DD must drop below (threshold - 1%) to return to previous level
```

## MODULE DEPENDENCIES

```
trading_system/
├── __init__.py              # Package exports
├── config.py                # All configuration (SINGLE SOURCE OF TRUTH)
│
├── signal_engine.py         # Trend signal generation
│   └── Dependencies: config
│
├── gatekeeper_engine.py     # Structural market filter
│   └── Dependencies: config
│
├── risk_engine.py           # Capital preservation governor
│   └── Dependencies: config
│
├── execution_engine.py      # MT5 order management
│   └── Dependencies: config, signal_engine
│
├── logging_module.py        # Audit trail
│   └── Dependencies: config, signal_engine, gatekeeper_engine, 
│                      execution_engine, risk_engine
│
├── orchestrator.py          # Main loop coordination
│   └── Dependencies: ALL MODULES
│
├── SAFETY_CHECKLIST.md      # Operational procedures
│
└── run_trading_system.py    # Entry point
```

## DATA FLOW SUMMARY

```
MT5 Market Data
      │
      ▼
Signal Engine ──────► Signal (or None)
      │                    │
      │                    ▼
      │              Risk Engine ──────► BLOCKED (if risk limits)
      │                    │
      │                    ▼ ALLOWED
      │              Gatekeeper ──────► BLOCKED (if failure conditions)
      │                    │
      │                    ▼ ALLOWED
      │              Position Sizing
      │                    │
      │                    ▼
      │              Execution Engine ──► MT5 Order
      │                    │
      │                    ▼
      └──────────────► Logging ──────► CSV/JSON Files
```

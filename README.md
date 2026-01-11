# FX Conditional Relative Value (CRV) System

## ⚠️ IMPORTANTE: Esto NO es Statistical Arbitrage

**FX NO exhibe mean reversion permanente.** Este sistema abandona explícitamente el paradigma de cointegración clásica y adopta un enfoque de **Relative Value Condicional**.

## Filosofía del Sistema

```
❌ NO asumimos mean reversion permanente
❌ NO usamos cointegración como condición dura
❌ NO esperamos que spreads vuelvan siempre a la media

✅ Operamos SOLO cuando el régimen lo permite
✅ Usamos z-score CONDICIONAL al régimen
✅ Aceptamos largos períodos de inactividad
✅ Preferimos CERO trades a trades estadísticamente inválidos
```

## Arquitectura en Capas

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRV SYSTEM                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CAPA 1: Selección Estructural (Mensual)                        │
│  └─ Coherencia económica + Estabilidad de correlación           │
│                                                                  │
│  CAPA 2: Filtro de Régimen (Semanal/Diario)                     │
│  └─ Volatilidad + Tendencia + Sentimiento + Eventos Macro       │
│                                                                  │
│  CAPA 3: Spread Condicional (Por barra)                         │
│  └─ Z-score condicional al régimen actual                       │
│                                                                  │
│  CAPA 4: Motor de Señales                                       │
│  └─ Solo señales si TODAS las condiciones se cumplen            │
│                                                                  │
│  CAPA 5: Gestión de Riesgo                                      │
│  └─ Stop por breakdown estructural, no por ticks                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Descargar datos (H4 y D1 recomendados para CRV)
python scripts/download_data.py --forex --timeframes H4,D1 --days 730

# 2. Ejecutar screening CRV
python scripts/crv_screen.py --timeframe H4 --save

# 3. Solo análisis estructural
python scripts/crv_screen.py --timeframe D1 --structural-only
```

## Diferencias con Statistical Arbitrage Clásico

| Aspecto | StatArb Clásico | CRV (Este Sistema) |
|---------|-----------------|-------------------|
| Mean Reversion | Permanente | Condicional al régimen |
| Cointegración | Requerida | NO requerida |
| Z-Score | Universal | Condicional al régimen |
| Filtro principal | Cointegración | Estabilidad estructural |
| Hurst exponent | Filtro duro | NO usado como filtro |
| Inactividad | Error | Comportamiento correcto |

## Capas del Sistema

### Capa 1: Selección Estructural de Pares

**NO requiere cointegración.** Requiere:

1. **Coherencia económica**: 
   - Moneda base compartida (EURUSD/EURJPY)
   - Moneda quote compartida (EURUSD/GBPUSD)
   - Bloque de commodities (AUDUSD/NZDUSD)

2. **Estabilidad de correlación**:
   - Media de correlación > 0.50 (más bajo que StatArb)
   - Desviación estándar de correlación < 0.15 (**más importante que el nivel**)

3. **Estabilidad del hedge ratio**:
   - Coeficiente de variación < 0.30
   - Sin drift estructural

### Capa 2: Filtro de Régimen FX

| Régimen | ADX | Volatilidad | Permite CRV |
|---------|-----|-------------|-------------|
| STABLE_LOW_VOL | < 20 | < P25 | ✅ SÍ |
| STABLE_NORMAL_VOL | < 20 | P25-P50 | ✅ SÍ |
| RANGE_BOUND | < 20 | P50-P75 | ✅ SÍ |
| TRENDING_STRONG | > 25 | Cualquier | ❌ NO |
| HIGH_VOLATILITY | Cualquier | > P75 | ❌ NO |
| MACRO_EVENT | N/A | N/A | ❌ NO |

**Eventos que BLOQUEAN CRV:**
- FOMC, ECB, BOE, BOJ
- NFP, CPI, GDP
- PMI, Employment

### Capa 3: Spread Condicional

```python
# Z-Score CONDICIONAL (lo que usamos)
zscore_conditional = (spread - regime_mean) / regime_std

# Z-Score UNCONDICIONAL (solo referencia)
zscore_unconditional = (spread - global_mean) / global_std
```

**El z-score depende del régimen actual:**
- Un z=2.0 en régimen estable es muy significativo
- Un z=2.0 en régimen volátil puede ser ruido

### Capa 4: Motor de Señales

Una señal es VÁLIDA solo si:

1. ✅ Par estructuralmente válido
2. ✅ Régimen permite CRV
3. ✅ Z-score condicional extremo (|Z| >= 1.5)
4. ✅ Confianza suficiente
5. ✅ Límites de riesgo respetados

**Si CUALQUIER condición falla → NO SIGNAL**

### Capa 5: Gestión de Riesgo

| Parámetro | Valor |
|-----------|-------|
| Max posiciones | 3 |
| Max exposición | 10% |
| Max por posición | 3% |
| Max drawdown | 8% (kill-switch) |
| Time stop | 50 barras |

**Stop-loss por breakdown estructural:**
- NO por ticks/pips
- SÍ por pérdida de estructura del spread

## Reglas de Trading

### Entry
```
1. Régimen = STABLE o RANGE_BOUND
2. Z-score condicional <= -1.5 → LONG spread
3. Z-score condicional >= +1.5 → SHORT spread
4. Tamaño según confianza y régimen
```

### Exit
```
1. Mean reversion: |Z| < 0.3
2. Time stop: > 50 barras
3. Structural stop: Z va demasiado contra
4. Regime change: Cambio a régimen no favorable
```

## Estructura del Proyecto

```
Pairs_Trading/
├── src/
│   ├── crv/                          # Sistema CRV
│   │   ├── pair_selector.py          # Capa 1: Selección estructural
│   │   ├── regime_filter.py          # Capa 2: Filtro de régimen
│   │   ├── conditional_spread.py     # Capa 3: Spread condicional
│   │   ├── signal_engine.py          # Capas 4-5: Señales y riesgo
│   │   └── crv_system.py             # Sistema integrado
│   │
│   ├── analysis/                     # Análisis (legacy StatArb)
│   │   ├── strict_selector.py
│   │   └── institutional_selector.py
│   │
│   └── strategy/                     # Estrategias
│       ├── conditional_statarb.py
│       └── adaptive_params.py
│
├── scripts/
│   ├── crv_screen.py                 # ← USAR ESTE
│   ├── conditional_screen.py
│   ├── strict_screen.py
│   ├── download_data.py
│   └── ...
│
├── data/historical/
├── results/crv/
└── docs/
```

## Output Esperado

```
FX CONDITIONAL RELATIVE VALUE (CRV) SYSTEM
================================================================================

LAYER 2: REGIME ASSESSMENT
────────────────────────────────────────────────────────────────────────────────

  Current Regime: stable_normal_vol
  Permits CRV: YES
  Confidence: 85%

  Volatility:
    ATR Percentile: 42
    Vol Regime: normal
    
  Trend:
    ADX: 18.5
    Strength: weak

LAYER 3-4: CONDITIONAL SPREAD ANALYSIS & SIGNALS
────────────────────────────────────────────────────────────────────────────────

  ⏸️ NO TRADEABLE SIGNALS
    This is expected behavior - CRV requires confluence of conditions.

WATCHLIST (Structurally Valid, Not Tradeable)
────────────────────────────────────────────────────────────────────────────────

  • EURUSD/GBPUSD
    Conditional Z: +0.85 | Unconditional Z: +1.12
    Status: Waiting for Z >= ±1.5

  • AUDUSD/NZDUSD
    Conditional Z: -1.23 | Unconditional Z: -0.98
    Status: Approaching entry threshold
```

## Métricas de Evaluación

| Métrica | Objetivo |
|---------|----------|
| Sharpe (neto) | > 0.5 |
| Max Drawdown | < 8% |
| Hit Rate | > 50% |
| Profit Factor | > 1.3 |
| % Tiempo Activo | < 30% |
| Trades/Año | 10-30 |

## Warnings

⚠️ **Este sistema está diseñado para NO operar la mayoría del tiempo**

⚠️ **Períodos de inactividad de SEMANAS son normales y esperados**

⚠️ **NO relajar filtros para generar más trades**

⚠️ **El z-score CONDICIONAL es lo que importa, no el uncondicional**

## Comparación de Scripts

| Script | Paradigma | Uso Recomendado |
|--------|-----------|-----------------|
| `crv_screen.py` | CRV (Recomendado) | Trading real FX |
| `conditional_screen.py` | StatArb condicional | Transición |
| `strict_screen.py` | StatArb estricto | Backtesting |
| `institutional_screen.py` | StatArb institucional | Análisis |

## License

MIT License

## Disclaimer

Este software es solo para propósitos educativos. El trading de FX conlleva riesgo significativo de pérdida. Los resultados pasados no garantizan resultados futuros. Siempre haga paper trading extensivo antes de operar con dinero real.

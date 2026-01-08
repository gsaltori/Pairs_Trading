# Pairs Trading System - IC Markets Global

Sistema institucional de pairs trading para Forex usando MetaTrader 5.

## ⚠️ Filosofía del Sistema

**Preferimos CERO trades antes que trades estadísticamente inválidos.**

Este sistema implementa filtros extremadamente estrictos. La mayoría de los pares serán rechazados. Esto es por diseño.

## Quick Start

### 1. Descargar Datos

```bash
# Todos los pares forex (recomendado)
python scripts/download_data.py --all --days 730

# O pares específicos
python scripts/download_data.py --symbols EURUSD,GBPUSD,AUDUSD,NZDUSD,EURJPY,GBPJPY --days 730
```

### 2. Screening ESTRICTO (Recomendado)

```bash
# Screening estricto para H1
python scripts/strict_screen.py --backtest --save

# Para H4 (puede mostrar más pares)
python scripts/strict_screen.py --timeframe H4 --backtest --save

# Ver todos los pares analizados
python scripts/strict_screen.py --show-all
```

### 3. Screening Original (Menos Estricto)

```bash
python scripts/institutional_screen.py --backtest --save
```

## Filtros Estrictos por Timeframe

| Timeframe | Max Half-life | Optimal HL | Min Trades/Year |
|-----------|---------------|------------|-----------------|
| M15       | 40 bars       | 5-20       | 20              |
| M30       | 50 bars       | 8-30       | 15              |
| H1        | 60 bars       | 10-40      | 12              |
| H4        | 120 bars      | 15-60      | 8               |
| D1        | 40 bars       | 5-25       | 5               |

## Filtros DUROS (No Negociables)

### 1. Relación Económica
Solo pares con relación económica directa:
- EURUSD / GBPUSD (Europeas vs USD)
- AUDUSD / NZDUSD (Oceanía vs USD)
- EURJPY / GBPJPY (Europeas vs JPY)
- AUDJPY / NZDJPY (Oceanía vs JPY)
- EURJPY / CHFJPY (Safe havens vs JPY)

### 2. Correlación
- Pearson ≥ 0.70
- Spearman ≥ 0.70
- Estabilidad ≥ 60%

### 3. Cointegración
- Engle-Granger p < 0.02 (muy estricto)
- Johansen trace > critical value × 1.10
- Estabilidad rolling (< 20% breakdown)

### 4. Half-life
- **RECHAZO AUTOMÁTICO** si excede máximo del timeframe
- No es un "score", es un filtro duro

### 5. Hurst Exponent
- **RECHAZO AUTOMÁTICO** si Hurst > 0.55
- Objetivo: Hurst < 0.50 (mean-reverting)

### 6. Frecuencia de Trades
- **RECHAZO** si trades estimados/año < mínimo del timeframe

## Output Esperado

Con filtros estrictos, espera:
- **0-2 pares** que pasen TODOS los filtros
- **>95%** de pares rechazados
- Pares seleccionados con:
  - Half-life corto
  - Hurst < 0.50
  - Cointegración fuerte
  - Relación económica clara

## Estructura del Proyecto

```
Pairs_Trading/
├── config/
│   ├── broker_config.py
│   └── settings.py
├── src/
│   ├── analysis/
│   │   ├── correlation.py
│   │   ├── cointegration.py
│   │   ├── spread_builder.py
│   │   ├── pair_screener.py         # Original screener
│   │   ├── institutional_selector.py # Institutional screener
│   │   └── strict_selector.py       # STRICT screener (recommended)
│   ├── strategy/
│   │   ├── signals.py
│   │   ├── pairs_strategy.py
│   │   └── adaptive_params.py
│   ├── backtest/
│   │   └── backtest_engine.py
│   ├── risk/
│   │   └── risk_manager.py
│   └── execution/
│       └── executor.py
├── scripts/
│   ├── download_data.py
│   ├── strict_screen.py             # ← USAR ESTE
│   ├── institutional_screen.py
│   ├── screen_pairs_offline.py
│   ├── backtest_offline.py
│   └── optimize_offline.py
├── data/
│   └── historical/
├── results/
│   ├── screening/
│   ├── backtests/
│   └── optimization/
└── main.py
```

## Checklist de Validación Post-Screening

### ✅ Filtros Estadísticos
- [ ] Half-life ≤ máximo del timeframe
- [ ] Hurst < 0.55
- [ ] EG p-value < 0.02
- [ ] ADF p-value < 0.05
- [ ] Correlación Pearson ≥ 0.70
- [ ] Correlación Spearman ≥ 0.70
- [ ] Estabilidad cointegración > 80%

### ✅ Validación Económica
- [ ] Par tiene relación económica directa
- [ ] Divisa común o economías ligadas
- [ ] No es combinación arbitraria

### ✅ Backtest Validation
- [ ] Sharpe > 0.5 (idealmente > 1.0)
- [ ] Max Drawdown < 20%
- [ ] Win rate > 40%
- [ ] Profit factor > 1.2
- [ ] Trades suficientes (>20 en período)

### ✅ Pre-Production
- [ ] Paper trading 2-4 semanas
- [ ] Verificar ejecución real (slippage, spread)
- [ ] Capital reducido inicial
- [ ] Monitoreo de degradación estadística

## Interpretación de Resultados

### Si 0 pares pasan:
Esto es NORMAL. Significa que el mercado actual no ofrece oportunidades válidas.
- **NO** relajes los filtros
- Prueba otro timeframe (H4, D1)
- Espera cambio de régimen de mercado
- Añade más símbolos

### Si 1-2 pares pasan:
Excelente. Valida con backtest y paper trading antes de operar.

### Si >5 pares pasan:
Revisa que los filtros estén correctamente configurados. Algo puede estar mal.

## Warnings

⚠️ **NO relajar filtros para generar trades**

⚠️ **Half-life largo = NO tradeable** (no es cuestión de "paciencia")

⚠️ **Hurst > 0.55 = trending, NO mean-reverting**

⚠️ **Cointegración débil = relación espuria**

## License

MIT License

## Disclaimer

Este software es solo para propósitos educativos. El trading de Forex conlleva riesgo significativo de pérdida. Siempre haga paper trading antes de operar con dinero real. Los resultados pasados no garantizan resultados futuros.

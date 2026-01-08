# Conditional Statistical Arbitrage System - Architecture

## 📋 Definición

**Conditional StatArb FX**: Un sistema que solo ejecuta pair trading cuando existe cointegración + mean reversion bajo un régimen de mercado favorable, y permanece completamente inactivo fuera de ese régimen.

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONDITIONAL STATARB SYSTEM                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   REGIME     │    │  COINTEGRA-  │    │   SPREAD     │       │
│  │  DETECTOR    │    │    TION      │    │   HEALTH     │       │
│  │              │    │  VALIDATOR   │    │   MONITOR    │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                             ▼                                    │
│                  ┌──────────────────┐                           │
│                  │     PAIR         │                           │
│                  │    MANAGER       │                           │
│                  └────────┬─────────┘                           │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                   │
│         ▼                 ▼                 ▼                   │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐            │
│  │   ACTIVE   │    │  DORMANT   │    │INVALIDATED │            │
│  │   PAIRS    │    │   PAIRS    │    │   PAIRS    │            │
│  └─────┬──────┘    └────────────┘    └────────────┘            │
│        │                                                         │
│        ▼                                                         │
│  ┌──────────────────┐                                           │
│  │     SIGNAL       │                                           │
│  │   GENERATOR      │                                           │
│  └────────┬─────────┘                                           │
│           │                                                      │
│           ▼                                                      │
│  ┌──────────────────┐                                           │
│  │   CONDITIONAL    │   Only if ALL conditions met              │
│  │     SIGNAL       │──────────────────────────────────►        │
│  └──────────────────┘                                    TRADE  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 Flujo de Decisión

```
                          START
                            │
                            ▼
            ┌───────────────────────────────┐
            │   Is Pair Economically Valid?  │
            └───────────────┬───────────────┘
                            │
                   NO ──────┼────── YES
                   │        │        │
                   ▼        │        ▼
              INVALIDATED   │   Check Cointegration
                            │        │
                            │   NO ──┼─── YES
                            │   │    │    │
                            │   ▼    │    ▼
                            │ INVALID│  Check Spread Health
                            │        │    │
                            │   NO ──┼─── YES
                            │   │    │    │
                            │   ▼    │    ▼
                            │ INVALID│  Check Market Regime
                            │        │    │
                            │    UNFAV ──┼── FAVORABLE
                            │      │     │    │
                            │      ▼     │    ▼
                            │   DORMANT  │   ACTIVE
                            │            │    │
                            │            │    ▼
                            │            │  Check Z-Score
                            │            │    │
                            │      NO ENTRY ──┼── ENTRY
                            │         │       │    │
                            │         ▼       │    ▼
                            │      WAIT       │  SIGNAL
                            │                 │
                            └─────────────────┘
```

## 📊 Estados de Pares

| Estado | Significado | Acción |
|--------|-------------|--------|
| **ACTIVE** | Válido + Régimen favorable | Puede generar señales |
| **DORMANT** | Válido pero régimen desfavorable | Esperar cambio de régimen |
| **INVALIDATED** | Falló tests estadísticos | Re-evaluar periódicamente |
| **WARMING_UP** | Datos insuficientes | Acumular más datos |

## 🎯 Regímenes de Mercado

| Régimen | ADX | Volatilidad | Tradeable |
|---------|-----|-------------|-----------|
| **RANGING** | < 20 | Normal | ✅ SÍ |
| **QUIET** | < 20 | Baja | ✅ SÍ |
| **TRENDING_WEAK** | 20-25 | Normal | ⚠️ Cautela |
| **TRENDING_STRONG** | > 25 | Alta | ❌ NO |
| **VOLATILE** | Any | Extrema | ❌ NO |

## 🧪 Validaciones

### 1. Cointegración Dinámica
```
Ventanas: [250, 500, 750] barras
P-value threshold: 0.05
Consistencia mínima: 67% (2/3 ventanas)
Max breakdowns recientes: 10%
```

### 2. Salud del Spread
```
ADF p-value: < 0.05
Half-life: 5-60 barras (H1)
Hurst: < 0.55
Hedge ratio drift: < 2σ
```

### 3. Régimen de Mercado
```
ADX < 25 (no trending fuerte)
ATR percentile < 75 (no volatilidad alta)
Sesión favorable
```

## 📈 Señales Condicionales

Una señal es VÁLIDA solo si:

1. ✅ Spread es estacionario en ventana actual
2. ✅ Half-life en rango óptimo
3. ✅ Volatilidad del spread estable
4. ✅ No hay ruptura estructural
5. ✅ Régimen es favorable
6. ✅ Z-Score >= ±2.0

Si CUALQUIER condición falla → NO HAY SEÑAL

## 🛡️ Risk Management

### Kill Switches
- Pérdida de cointegración → Cerrar posición
- Cambio de régimen → DORMANT (no nuevas entradas)
- Drift del hedge ratio → Re-calcular o cerrar
- Volatilidad explosiva → Cerrar todas las posiciones

### Position Sizing
```python
size_factor = {
    'fast_reversion': 1.0,      # HL < 20
    'moderate_reversion': 0.8,  # HL 20-40
    'slow_reversion': 0.6,      # HL 40-60
}
```

## 📁 Estructura de Archivos

```
src/strategy/
├── conditional_statarb.py    # Componentes core
│   ├── PairState             # Enum de estados
│   ├── MarketRegime          # Enum de regímenes
│   ├── MarketRegimeDetector  # Detección de régimen
│   ├── DynamicCointegrationValidator
│   └── SpreadHealthMonitor
│
├── conditional_manager.py    # Sistema integrado
│   ├── ConditionalSignalGenerator
│   ├── ConditionalPairManager
│   └── ConditionalStatArbSystem
│
└── adaptive_params.py        # Parámetros adaptativos

scripts/
├── conditional_screen.py     # Script principal
├── strict_screen.py          # Screening estricto (sin régimen)
└── institutional_screen.py   # Screening institucional original
```

## 🔧 Uso

```bash
# Screening condicional completo
python scripts/conditional_screen.py --timeframe H1 --save

# Con parámetros personalizados
python scripts/conditional_screen.py --timeframe H4 --half-life-max 120

# Solo símbolos específicos
python scripts/conditional_screen.py --symbols EURUSD,GBPUSD,AUDUSD,NZDUSD
```

## 📊 Output Esperado

```
CONDITIONAL STATARB SYSTEM STATUS
============================================================

Last Update: 2026-01-08 10:30:00
System Active: NO

Pair States:
  ACTIVE:      0
  DORMANT:     3
  INVALIDATED: 12
  WARMING UP:  0

--------------------------------------------------------------
DORMANT PAIRS (waiting for favorable regime)
--------------------------------------------------------------

  EURUSD/GBPUSD
    Reasons: Regime: trending_strong, ADX=32.5
    Dormant since: 2026-01-08 08:00:00

  AUDUSD/NZDUSD
    Reasons: Regime: volatile, ATR percentile=85
    Dormant since: 2026-01-08 09:00:00

--------------------------------------------------------------

⏸️ SYSTEM IS WAITING - No trades

Reasons:
  • No active pairs
  • All valid pairs are DORMANT (regime unfavorable)

============================================================
```

## ⚠️ Principios Fundamentales

1. **Zero trades > Invalid trades**
2. **DORMANT es un estado válido, no un error**
3. **El sistema SABE cuándo NO operar**
4. **La inactividad es una feature, no un bug**
5. **Nunca forzar trades por presión**

## 📝 Checklist de Validación

### Pre-Trade
- [ ] Cointegración estable (67%+ ventanas)
- [ ] Half-life en rango (5-60 para H1)
- [ ] Hurst < 0.55
- [ ] Hedge ratio estable (< 2σ drift)
- [ ] Régimen favorable (RANGING o QUIET)
- [ ] Z-Score >= ±2.0
- [ ] Spread estacionario (ADF p < 0.05)

### En Trade
- [ ] Monitorear cointegración rolling
- [ ] Monitorear régimen
- [ ] Stop por breakdown estadístico

### Post-Trade
- [ ] Registrar razón de salida
- [ ] Actualizar estadísticas de par
- [ ] Re-evaluar clasificación del par

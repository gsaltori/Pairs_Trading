# SAFETY CHECKLIST & OPERATIONAL GUIDE

## WHEN SYSTEM SHUTS DOWN

| Condition | Automatic? | Action | Resume Procedure |
|-----------|-----------|--------|------------------|
| Drawdown ≥ 8% | YES | HALT | Manual review required |
| Drawdown ≥ 10% | YES | HALT + REVIEW FLAG | Manual intervention + root cause analysis |
| SIGINT/SIGTERM | YES | Graceful shutdown | Restart system |
| MT5 disconnection | NO | Continues trying | Automatic reconnection |
| 3 losses in same day | YES | Stop new trades | Wait for next day |
| System exception | NO | Logs + continues | Check logs |

## WHY IT SHUTS DOWN

### Capital Protection (Primary)
- **-8% Drawdown**: Prevents catastrophic loss. At this level, the edge may have failed or market conditions have changed dramatically.
- **-10% Drawdown**: Forces human analysis before any further trading. Something is fundamentally wrong.

### Risk Containment (Secondary)
- **3 Daily Losses**: Prevents revenge trading and emotional decisions. A bad day should stay limited.
- **Max Concurrent Risk Exceeded**: Prevents overexposure even if individual trades pass risk checks.

### System Integrity
- **MT5 Disconnection**: Temporary - system waits and reconnects. Positions have broker-side SL/TP.
- **Graceful Signals (Ctrl+C)**: Allows clean shutdown with state persistence.

## HOW TO RESUME SAFELY

### After Normal Shutdown
```bash
# 1. Check last state
cat trading_system_data/state/system_state.json

# 2. Verify no open positions need attention
# 3. Restart system
python -m trading_system.orchestrator
```

### After Drawdown Halt (-8%)
```python
from trading_system import TradingSystem, SystemConfig

# 1. Analyze what happened
# - Check trade logs
# - Check block logs  
# - Check market conditions

# 2. If confident to resume
config = SystemConfig(dry_run=False)
system = TradingSystem(config)

# 3. Manual resume with optional HWM reset
system.risk_engine.initialize(current_equity)
system.risk_engine.manual_resume(new_hwm=None)  # Or set new HWM
```

### After Manual Review Required (-10%)
**DO NOT AUTO-RESUME**

1. Export all logs for analysis
2. Check if strategy fundamentals still valid
3. Verify gatekeeper was functioning
4. Consider:
   - Was it market regime change?
   - Was it execution issues?
   - Was it a black swan event?
5. Document findings
6. If appropriate, reset HWM to current equity and resume with reduced risk

### After System Exception
```bash
# 1. Check system log
cat trading_system_data/logs/system.log | tail -100

# 2. Check if positions are still managed by broker SL/TP
# 3. Fix issue
# 4. Restart
```

## PRE-DEPLOYMENT CHECKLIST

- [ ] Run in DRY_RUN mode for minimum 1 week
- [ ] Verify all logs are being created correctly
- [ ] Verify state persistence works (stop and restart)
- [ ] Test with small position size first (0.01 lots)
- [ ] Verify SL/TP are set on all orders
- [ ] Confirm broker allows automated trading
- [ ] Set up monitoring/alerts (email, etc.)
- [ ] Document emergency contact procedures

## LIVE DEPLOYMENT PROCEDURE

```python
# 1. Create production config
config = SystemConfig(
    dry_run=False,  # ⚠️ LIVE MODE
)

# 2. Override paths for production
config.paths = PathConfig(base_dir=Path("/path/to/production/data"))

# 3. Start with reduced risk initially
config.risk.risk_per_trade_normal = 0.0025  # Start at 0.25%

# 4. Run
system = TradingSystem(config)
system.run()
```

## EMERGENCY PROCEDURES

### Close All Positions Immediately
```python
system.emergency_close_all()
```

### Force Halt
```python
system.risk_engine._halt_system("Manual emergency halt")
```

### Check Status
```python
print(system.get_status())
print(system.risk_engine.get_status_summary())
```

## MONITORING REQUIREMENTS

### Daily Checks (Minimum)
1. System is running (`get_status()`)
2. No halt conditions
3. Drawdown is within acceptable range
4. Trades are being logged

### Weekly Checks
1. Review all blocked trades - is gatekeeper working correctly?
2. Review win rate vs expectations
3. Compare actual vs expected drawdown
4. Verify state file integrity

### Monthly Checks
1. Full log analysis
2. Performance vs baseline backtest
3. Gatekeeper effectiveness analysis
4. Risk parameter appropriateness

## LOG LOCATIONS

| Log | Location | Purpose |
|-----|----------|---------|
| System | `logs/system.log` | All system events |
| Trades | `logs/trades.csv` | Every trade attempt |
| Blocks | `logs/blocks.csv` | Every gatekeeper decision |
| Risk | `logs/risk_state.csv` | Periodic risk snapshots |
| State | `state/system_state.json` | Persistent state |

## CONTACT ESCALATION

1. **Level 1**: Check logs, restart if needed
2. **Level 2**: Review state, manual resume if appropriate
3. **Level 3**: Full analysis, potential strategy review
4. **Level 4**: Stop all trading, comprehensive audit

---

**REMEMBER**: When in doubt, DO NOT TRADE. The market will be there tomorrow.

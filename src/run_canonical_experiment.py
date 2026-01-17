from datetime import datetime
from pathlib import Path

from crv_engine.data.mt5_loader import load_pair_ohlc
from crv_engine.experiments import (
    create_baseline_experiment,
    run_walk_forward,
    analyze_edge_boundaries,
)
from crv_engine.utils.serialization import save_json


# ==============================
# LOCKED PARAMETERS (DO NOT EDIT)
# ==============================
PAIR_A = "EURUSD"
PAIR_B = "GBPUSD"
TIMEFRAME = "H4"
BARS = 5000
BLOCK_SIZE = 500
WARMUP = 60
CRR_THRESHOLD = 0.55
MAX_INVALIDATION = 0.40


def main():
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    print("🔹 Loading MT5 data...")
    observations = load_pair_ohlc(
        symbol_a=PAIR_A,
        symbol_b=PAIR_B,
        timeframe=TIMEFRAME,
        bars=BARS,
    )

    assert len(observations) == BARS, "Observation stream length mismatch"

    print("🔹 Creating experiment config...")
    exp_config = create_baseline_experiment(
        name="canonical_edge_eval",
        crr_threshold=CRR_THRESHOLD,
        max_invalidation=MAX_INVALIDATION,
    )

    print("🔹 Running walk-forward...")
    wf_output = run_walk_forward(
        observation_stream=observations,
        experiment_config=exp_config,
        block_size=BLOCK_SIZE,
        warmup=WARMUP,
        verbose=True,
    )

    wf_path = output_dir / f"walkforward_canonical_{timestamp}.json"
    save_json(wf_output, wf_path)
    print(f"✅ Walk-forward saved: {wf_path}")

    print("🔹 Extracting predictions with observables...")
    predictions = wf_output.predictions_with_observables
    assert len(predictions) > 0, "No predictions generated"

    print("🔹 Running edge boundary analysis...")
    eb_output = analyze_edge_boundaries(
        predictions=predictions,
        walk_forward_output=wf_output,
        verbose=True,
    )

    eb_path = output_dir / f"edge_boundary_canonical_{timestamp}.json"
    save_json(eb_output, eb_path)
    print(f"✅ Edge boundary saved: {eb_path}")

    print("\n🏁 CANONICAL EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()

"""Export deterministic PyTorch reference forecasts for the Rust parity tests.

Usage:
    python tools/export_reference.py <model_dir> <output.json>

The reference implementation is expected at TIMESTRAL_REF_SRC (default: the clone
used during development) or installed in the environment.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

DEFAULT_REF = Path(os.environ.get("TIMESTRAL_REF_SRC", r"C:\Users\PC\AppData\Local\Temp\opencode\timestral-ref\src"))
if DEFAULT_REF.is_dir():
    sys.path.insert(0, str(DEFAULT_REF))

from timestral import TimestralForecaster  # noqa: E402


def clean(value):
    if isinstance(value, np.ndarray):
        flat = []
        for v in value.reshape(-1):
            f = float(v)
            flat.append(None if not math.isfinite(f) else f)
        return flat
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _reshape(values, shape):
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32).reshape(shape)


def main() -> None:
    global FORECASTER
    model_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    FORECASTER = TimestralForecaster.from_pretrained(model_dir)
    rng = np.random.default_rng(42)
    t = np.arange(4096, dtype=np.float64)

    cases = []

    def run(name, context, prediction_length, *, observation_mask=None, **kwargs):
        context = np.asarray(context, dtype=np.float32)
        shape = list(context.shape)
        case = {
            "name": name,
            "context": clean(context),
            "context_shape": shape,
            "prediction_length": prediction_length,
            "symmetric_averaging": kwargs.pop("symmetric_averaging", True),
            "clip_nonnegative": kwargs.pop("clip_nonnegative", True),
            "quantile_levels": list(FORECASTER.quantile_levels),
        }
        for key in ("past_covariates", "known_covariates",
                    "categorical_past_covariates", "categorical_known_covariates"):
            value = kwargs.pop(key, None)
            value = None if value is None else np.asarray(value, dtype=np.float32)
            case[key] = None if value is None else clean(value)
            case[key + "_shape"] = None if value is None else list(value.shape)
        case["observation_mask"] = None if observation_mask is None else clean(np.asarray(observation_mask, dtype=np.float32))
        case["observation_mask_shape"] = None if observation_mask is None else list(np.asarray(observation_mask).shape)
        assert not kwargs, kwargs

        forecast = FORECASTER.forecast(
            context,
            prediction_length=prediction_length,
            observation_mask=None if observation_mask is None else np.asarray(observation_mask, dtype=bool),
            past_covariates=_reshape(case["past_covariates"], case["past_covariates_shape"]),
            known_covariates=_reshape(case["known_covariates"], case["known_covariates_shape"]),
            categorical_past_covariates=_reshape(case["categorical_past_covariates"], case["categorical_past_covariates_shape"]),
            categorical_known_covariates=_reshape(case["categorical_known_covariates"], case["categorical_known_covariates_shape"]),
            symmetric_averaging=case["symmetric_averaging"],
            clip_nonnegative=case["clip_nonnegative"],
        )
        q = forecast.quantiles.cpu().numpy()
        case["quantiles"] = clean(q)
        case["quantiles_shape"] = list(q.shape)
        cases.append(case)
        print(f"{name}: context={shape} horizon={prediction_length} -> {q.shape}")

    base = rng.standard_normal(2016).cumsum()
    positive = np.abs(rng.standard_normal(2016).cumsum()) + 1.0

    run("uni_2016_96_sym", base, 96)
    run("uni_2016_96_nosym", base, 96, symmetric_averaging=False)
    run("uni_1000_96_nosym", 10 * np.sin(2 * np.pi * t[:1000] / 24) + rng.standard_normal(1000).cumsum(), 96,
        symmetric_averaging=False)
    run("long_horizon_150_200_nosym", base[:150], 200, symmetric_averaging=False)
    run("block_ar_200_1200_nosym", base[:200], 1200, symmetric_averaging=False)

    multi = np.stack([
        10 * np.sin(2 * np.pi * t[:300] / 24) + rng.standard_normal(300).cumsum(),
        5 * np.cos(2 * np.pi * t[:300] / 7) + rng.standard_normal(300).cumsum() * 0.5,
        rng.standard_normal(300).cumsum() * 2 + 50,
    ])[None].repeat(2, axis=0)
    multi[0, 1, 40:55] = np.nan
    multi[1, 2, 200:210] = np.nan
    run("multi_300_50_nosym", multi, 50, symmetric_averaging=False)
    run("multi_300_50_sym", multi, 50, symmetric_averaging=True)

    past = np.stack([np.sin(t[:200] / 9), np.cos(t[:200] / 17)])[None]
    known = np.stack([np.sin(t[:224] / 11) + 2.0])[None]
    known[0, 0, 30:35] = np.nan
    run("past_known_200_24_nosym", positive[:200][None, None], 24,
        past_covariates=past, known_covariates=known, symmetric_averaging=False)
    run("past_known_200_24_sym", positive[:200][None, None], 24,
        past_covariates=past, known_covariates=known, symmetric_averaging=True)

    cat = rng.integers(0, 4, size=(1, 1, 224)).astype(np.float64)
    cat[0, 0, 40:45] = np.nan
    run("cat_known_200_24_nosym", positive[:200][None, None], 24,
        categorical_known_covariates=cat, symmetric_averaging=False)

    multi_past = np.stack([np.sin(t[:250] / 5), np.cos(t[:250] / 13)])[None].repeat(2, axis=0)
    multi_known = np.stack([np.sin(t[:290] / 7), np.cos(t[:290] / 19)])[None].repeat(2, axis=0)
    multi_ctx = np.stack([
        rng.standard_normal(250).cumsum(),
        20 + rng.standard_normal(250).cumsum(),
    ])[None].repeat(2, axis=0)
    multi_ctx[0, 0, 10:18] = np.nan
    multi_known[0, 1, 255:262] = np.nan
    run("multi_cov_250_40_sym", multi_ctx, 40,
        past_covariates=multi_past, known_covariates=multi_known, symmetric_averaging=True)

    mask = np.ones((1, 1, 100), dtype=np.float32)
    mask[0, 0, 3:9] = 0
    mask[0, 0, 57:60] = 0
    run("explicit_mask_100_12_nosym", base[:100][None, None], 12,
        observation_mask=mask, symmetric_averaging=False)

    short = np.stack([
        10 * np.sin(2 * np.pi * t[:96] / 24) + rng.standard_normal(96).cumsum(),
        rng.standard_normal(96).cumsum(),
    ])[None]
    run("v2_96_32_nosym", short, 32, symmetric_averaging=False)
    short_nan = short.copy()
    short_nan[0, 1, 20:26] = np.nan
    run("v2nan_96_32_nosym", short_nan, 32, symmetric_averaging=False)
    batch2 = short[:, :1].repeat(2, axis=0)
    run("b2_96_32_nosym", batch2, 32, symmetric_averaging=False)
    short3 = np.concatenate([short, short[:, :1] * 0.5], axis=1)
    run("v3_96_32_nosym", short3, 32, symmetric_averaging=False)

    payload = {
        "model": str(model_dir),
        "cases": cases,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

# AGENTS.md

## Project

CPU-only Rust inference engine for Mistral Timestral time-series checkpoints, plus a `timestral` CLI. Checkpoints live under `models/` and are downloaded separately; they are not part of the crate.

## Commands

- Build: `cargo build --release`
- Full verification: `cargo test --release -- --nocapture`
- One parity model: `cargo test --release --test parity matches_pytorch_reference_28m -- --nocapture`
- Lint/format: `cargo fmt --all` then `cargo clippy --release --all-targets`
- Benchmark: `.\target\release\timestral.exe bench --model models\Timestral-5M --context 2016 --horizon 96 --threads 8 --no-sym`
- Forecast: `.\target\release\timestral.exe forecast --input fixtures\reference_5m.json --case uni_2016_96_nosym --output out.json`

`cargo test --release` does not rebuild `target/release/timestral.exe`. Run `cargo build --release` before timing the CLI after library changes.

## Tests and Fixtures

- Tests are integration-only PyTorch parity checks with tolerance `2e-3 + 2e-5 * scale` and a strict quantile-monotonicity check.
- Parity tests skip silently when the checkpoint is absent, so a green run may verify nothing. Confirm the case output or the 5M/28M model paths.
- Model overrides: `TIMESTRAL_MODEL` (5M) and `TIMESTRAL_MODEL_28M` (28M). Defaults are `models/Timestral-5M` and `models/Timestral-28M`.
- Regenerate fixtures only after intentional semantic changes. The script needs the official reference checkout (`https://github.com/mistralai/timestral`):
  `$env:TIMESTRAL_REF_SRC="<checkout>\src"; python tools\export_reference.py models\Timestral-28M fixtures\reference_28m.json`
- Fixtures cover covariates, categorical covariates, explicit masks, multi-variate data, batches, and long-horizon block-AR. Do not remove cases or loosen tolerance to make a change pass.
- 5M and 28M have fixtures; 209M is downloaded but not parity-tested; 123M is not present.

## Non-Obvious Invariants

- `cumulative_welford` in `src/ops.rs` must process each row independently and snapshot the previous prefix within each scan pass. A flattened scan leaked row-0 statistics into later rows and broke all multivariate forecasts.
- `VariateMask.key_valid` is row-major `(rows, enc_len)` and indexed as `key_row * enc_len + slot`. Do not transpose it.
- `exp_fast`, sgemm-based time attention, and the two GEMM parallelization paths are deliberate performance choices. Re-run parity after changing any numerical path.
- `Mat::linear` stays serial below `1 << 24` output elements. Above that it splits columns when `n >= 512` (unsafe `SendPtr`) and rows otherwise. Removing this makes tiny GEMMs much slower on hybrid CPUs.
- `.cargo/config.toml` sets `-C target-cpu=native`. Remove or override it for portable binaries or cross-machine benchmarking.
- Symmetric averaging doubles inference work and reverses context/quantiles.
- The safetensors loader accepts F32 checkpoints only.

## Layout

- `src/forecaster.rs`: public API, preprocessing/assembly, block-autoregressive decode, symmetric averaging, clipping.
- `src/model.rs`: checkpoint loading, patch embedding, transformer, head, covariate and refine paths.
- `src/ops.rs`: GEMM, tensor ops, Welford scans, quantile transforms.
- `src/attention.rs`: time/variate masks, partial RoPE, attention.
- `src/bin/timestral.rs`: CLI and benchmark harness.
- `fixtures/`: reference inputs/outputs. `tools/export_reference.py`: fixture generator.

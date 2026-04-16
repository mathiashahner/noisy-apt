uv run ./scripts/build_maestro_musdb_mixtures.py \
  --maestro_root ./data/maestro-v3.0.0 \
  --musdb_root ./data/musdb18hq \
  --out_root ./data/maestro_musdb_mixed \
  --sr 44100 \
  --target_lufs -14 \
  --n_samples 0 \
  --seed 1377
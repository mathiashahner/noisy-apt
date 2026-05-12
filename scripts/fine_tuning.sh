python -m transkun.createDatasetMaestro \
  data/maestro_musdb_mixed \
  data/maestro_musdb_mixed/maestro-v3.0.0.csv \
  data/maestro_musdb_mixed/meta \
  --noPedalExtension


mkdir -p checkpoint
python -m moduleconf.generate \
  Model:transkun.ModelTransformer > checkpoint/conf.json


python -m transkun.train \
  checkpoint/2.0.pt \
  --nProcess 2 \
  --datasetPath data/maestro_musdb_mixed \
  --datasetMetaFile_train data/maestro_musdb_mixed/meta/train.pickle \
  --datasetMetaFile_val data/maestro_musdb_mixed/meta/val.pickle \
  --modelConf checkpoint/2.0.conf \
  --batchSize 16 \
  --max_lr 3e-5 \
  --nIter 50000 \
  --weight_decay 1e-3 \
  --allow_tf32


python -m transkun.transcribe \
  data/tests/music.mp3 \
  data/tests/music.mid \
  --device cuda


python -m transkun.computeMetrics \
  data/2.3/ data/2.x/ \
  --outputJSON data/2.3/result.json \
  --nProcess 1 \
  --alignOnset \
  --noPedalExtension


python -m transkun.plotDeviation \
  data/2.3/result.json \
  --output data/2.3/result.png \
  --labels "16 epochs"

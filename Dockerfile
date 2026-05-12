FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

WORKDIR /

COPY transkun /transkun

RUN pip install --break-system-packages \
    librosa matplotlib mir-eval moduleconf ncls \
    numpy pandas pretty-midi pydub scipy seaborn \
    soundfile sox soxr tensorboard torch \
    torch-optimizer torchaudio tqdm

ENTRYPOINT ["python", "-m", \
            "transkun.train", "/gcs/noisy-piano-transcription-bucket/checkpoint/checkpoint.pt", \
            "--nProcess", "2", \
            "--datasetPath", "/gcs/noisy-piano-transcription-bucket/maestro_musdb_mixed", \
            "--datasetMetaFile_train", "/gcs/noisy-piano-transcription-bucket/maestro_musdb_mixed/meta/train.pickle", \
            "--datasetMetaFile_val", "/gcs/noisy-piano-transcription-bucket/maestro_musdb_mixed/meta/val.pickle", \
            "--modelConf", "/gcs/noisy-piano-transcription-bucket/checkpoint/model.conf", \
            "--batchSize", "16", \
            "--max_lr", "3e-5", \
            "--weight_decay", "1e-3", \
            "--nIter", "50000", \
            "--allow_tf32"]

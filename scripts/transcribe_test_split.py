import os
import pandas as pd

maestro_base = "data/maestro_musdb_mixed/"
outfolder = "data/2.0/"


def removeExtension(file):
    return ".".join(os.path.basename(file).split(".")[:-1])


def transcribe(index, total, file):
    print(index, "/", total, " - ", file)

    path = os.path.join(outfolder, removeExtension(file) + ".mid")
    os.system('python -m transkun.transcribe "%s" "%s"' % (file, path))


def get_dataset():
    gcs_path = maestro_base + "maestro-v3.0.0.csv"
    df_maestro = pd.read_csv(gcs_path)
    return df_maestro[df_maestro["split"] == "test"].reset_index(drop=True)


def main():
    df_test = get_dataset()

    for i, row in df_test.iterrows():
        audio_rel = row.get("audio_filename")

        if not isinstance(audio_rel, str):
            continue

        audio_path = maestro_base + audio_rel
        transcribe(i, len(df_test), audio_path)


main()

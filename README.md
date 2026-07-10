# Noise Robust Piano Transcription with Neural Semi-CRF

This repo contains code to transcribe expressive piano performances into MIDI, even in the presence of external audio interference.

## pip installation

```bash
pip3 install transkun
```

The pip package provides a quick command for transcribing piano performance audio into midi:

```bash
$ transkun input.mp3 output.mid
```

with cuda:

```bash
$ transkun input.mp3 output.mid --device cuda
```

**Note:** The code/pip package shipped checkpoint is trained **without pedal extension of notes**, and **with data augmentation**,which I believe is closer to a real performance.
Be cautious that the convention for the piano transcription task in previous works is extending all notes by sustain pedal durations. For more checkpoints, e.g, those reported in the paper, see [Model Cards](#model-cards)

**Colab Notebook** [Colab](https://colab.research.google.com/drive/1XuFNdYbcHBy3OyGCmtt2UGa2-PXjkJeo?usp=sharing)

## Basic Usage

### The Semi-CRF Layer

This code includes an neural semi-CRF module that is optimized for the problem domain.

Here is a minimal example for using this module:

```python
import CRF
import torch

T = 200
NBatch = 4

# representing the score for the interval [TBegin, TEnd]
# dimensions: [TEnd, TBegin, NBatch]
# only the lower triangular part is used
score = ((torch.randn(T,  T, NBatch))).cuda()

# representing the score for being not an interval, dimensions [TBegin, TBegin+1]
noiseScore= ((torch.randn(T-1,  NBatch))).cuda()

# a list of list of non-overlapping intervals
intervals = [
        [(0,2), (4,6),(6,6), (7,8)],
        [(1,2), (3,5), (19,19)],
        [(0,0),(4,7)],
        [],
        ]

crf = CRF.NeuralSemiCRFInterval(score, noiseScore)

## log probability
logP = crf.logProb(intervals)

## decoding
decoded = crf.decode()

## decoding starting from a given position, useful for segment based processing
decoded = crf.decode(forcedStartPos = [4]*NBatch)
```

### Transcribing piano performance into a MIDI file

```bash
python3 -m transkun.transcribe -h

usage: transcribe.py [-h] [--weight WEIGHT] [--conf CONF] [--device [DEVICE]] [--segmentHopSize SEGMENTHOPSIZE] [--segmentSize SEGMENTSIZE] audioPath outPath

positional arguments:
  audioPath             path to the input audio file
  outPath               path to the output MIDI file

options:
  -h, --help            show this help message and exit
  --weight WEIGHT       path to the pretrained weight
  --conf CONF           path to the model conf
  --device [DEVICE]     The device used to perform the most computations (optional), DEFAULT: cpu
  --segmentHopSize SEGMENTHOPSIZE
                        The segment hopsize for processing the entire audio file (s), DEFAULT: the value defined in model conf
  --segmentSize SEGMENTSIZE
                        The segment size for processing the entire audio file (s), DEFAULT: the value defined in model conf

```

This script can also be used directly as the command line command 'transkun' if the pip package is installed, e.g.,

```bash
$ transkun input.mp3 output.mid
```

## Model Cards

- onset deviations on MAPS deviates strongly from a Normal distribution, suggesting potential annotation issues. ad hoc align is used to fix this bias. Offset deviations are still abnormal after this correction.
- Aug means the model is trained with data augmentation
- No Ext means without pedal extension
- The default checkpoint shipped with the code/pip package is Transkun V2 No Pedal Ext. Currently it is fine-tuned from Transkun V2 Aug, will train from scratch in the future.
- There's some minor bug fixes since the paper, the results listed here reflect the latest version which may differ slightly from the paper.

## Handling the dataset

### Converting to the same sampling rate

We assume the data contains only the same sampling rate 44100hz. Therefore for the maestro dataset it is necessary to perform sampling rate conversion to 44100hz for the last two years (2017 and 2018) .

### Generating metadata files

Assuming all audio files have already been converted to the same sampling rate, we iterate the entire dataset to combine the groundtruth midi and metadata into a single file.

The following script will generate train.pt, val.pt and test.pt

```bash
python3 -m transkun.createDatasetMaestro -h
usage: createDatasetMaestro.py [-h] [--noPedalExtension] datasetPath metadataCSVPath outputPath

positional arguments:
  datasetPath         folder path to the maestro dataset
  metadataCSVPath     path to the metadata file of the maestro dataset (csv)
  outputPath          path to the output folder

optional arguments:
  -h, --help          show this help message and exit
  --noPedalExtension  Do not perform pedal extension according to the sustain pedal
```

This command will generate train.pt, dev.pt, test.pt in the outputPath.

## Training

After generating the metadata files, we can perform training using the dataset. During training, the audio waveforms will be fetched directly from the original .wav files.

Firstly, generate a config template file for the model.:

```bash
mkdir checkpoint
python3 -m moduleconf.generate Model:transkun.ModelTransformer > checkpoint/conf.json
```

Then we call the training script.

```bash
python3 -m transkun.train -h
```

## The Evaluation Module

### Comparing the output MIDI files and the groundtruth MIDI files

We also provide an out-of-box tool for computing metrics directly from output midi files.

```bash
usage: computeMetrics.py [-h] [--outputJSON OUTPUTJSON] [--noPedalExtension] [--applyPedalExtensionOnEstimated] [--nProcess [NPROCESS]] [--alignOnset] [--dither DITHER]
                         [--pedalOffset PEDALOFFSET] [--onsetTolerance ONSETTOLERANCE]
                         estDIR groundTruthDIR

compute metrics directly from MIDI files.
Note that estDIR should have the same folder structure as the groundTruthDIR.
The MIDI files to evaluate should have the same extension as the ground truth.
Metrics outputed are ordered by precision, recall, f1, overlap.

positional arguments:
  estDIR
  groundTruthDIR

options:
  -h, --help            show this help message and exit
  --outputJSON OUTPUTJSON
                        path to save the output file for detailed metrics per audio file
  --noPedalExtension    Do not perform pedal extension according to the sustain pedal for the ground truth
  --applyPedalExtensionOnEstimated
                        perform pedal extension for the estimated midi
  --nProcess [NPROCESS]
                        number of workers for multiprocessing
  --alignOnset          whether or not to realign the onset.
  --dither DITHER       amount of noise added to the prediction.
  --pedalOffset PEDALOFFSET
                        offset added to the groundTruth sustain pedal when extending notes
  --onsetTolerance ONSETTOLERANCE
                        onset tolerance, default: 0.05 (50ms)

```

Currently, we do not support evaluation of multitrack MIDIs.

This command can also be used directly as the command line script 'transkunEval' if the pip package is installed.

### Citation

If you find this repository helpful, please consider citing:

Bibtex:

```bibtex
@inproceedings{hahner2026Noisy,
  author    = {Mathias Daniel Hahner},
  title     = {TRANSCRIÇÃO AUTOMÁTICA DE PIANO RESISTENTE A RUÍDO: Uma Abordagem com Fine-tuning e Data Augmentation},
  year      = {2026},
}
```

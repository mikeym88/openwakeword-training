"""
OpenWakeWords Training Data Preparation Script. Based on the [OpenWakeWord Jupyter Notebook][openwakeword-jupyter-notebook].

# Requirements

## Dependencies

* Some of the packages depend on audiooop module, which has been [deprecated in Python 3.11][python-audioop].
  To install the audioop-lts package, install the [audioop-lts][audioop-lts] package from PyPI.
  This package is a drop-in replacement for the deprecated audioop module, and it is compatible with Python 3.11 and later versions.
* The `duckgo` package requires a CPU with AVX2 support. If your CPU does not support AVX2, you can (theoretically) install the `duckgo` package with the `--no-binary` option to build it from source.

[python-audioop]: https://docs.python.org/3/library/audioop.html
[audioop-lts]: https://pypi.org/project/audioop-lts/
[flac-to-wav]: https://stackoverflow.com/questions/58424108/with-python-how-can-i-convert-a-flac-stream-to-a-wav-stream
[openwakeword-jupyter-notebook]: https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb

"""

import os
import numpy as np
import torch
import sys
from pathlib import Path
import uuid
import yaml
import datasets
import scipy
from tqdm import tqdm
import subprocess
import duckdb
from pydub import AudioSegment
import io



def process_room_impulse_responses(output_dir: str = "./mit_rirs"):
    '''

    Notes:
    * Download room impulse responses collected by MIT from: <https://mcdermottlab.mit.edu/Reverb/IR_Survey.html>
    * The dataset is available on HuggingFace: <https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses>
    '''

    print("Creating MIT RIR folder")
    # Output Directory for MIT RIR dataset
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print("Loading MIT RIR dataset...")
    rir_dataset = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )

    # Save clips to 16-bit PCM wav files
    print("Saving MIT RIR clips...")
    for row in tqdm(rir_dataset):
        name = row['audio'].metadata.path.split('/')[-1]
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (row['audio']['array'] * 32767).astype(np.int16),
    )
    print("Done saving MIT RIR clips.")


def process_google_audioset_dataset():
    '''

    Notes:
    *
    '''
    ## Download noise and background audio

    # Audioset Dataset (https://research.google.com/audioset/dataset/index.html)
    # Download one part of the audioset .tar files, extract, and convert to 16khz
    # For full-scale training, it's recommended to download the entire dataset from
    # https://huggingface.co/datasets/agkphysics/AudioSet, and
    # even potentially combine it with other background noise datasets (e.g., FSD50k, Freesound, etc.)
    if not os.path.exists("audioset"):
        os.mkdir("audioset")

    dataset_file = "bal_train/00.parquet"
    print(f"Downloading {dataset_file} from HuggingFace AudioSet dataset...")
    out_dir = Path("audioset", dataset_file.split("/")[0])
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    link = f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/{dataset_file}?download=true"

    dataset_path = Path(out_dir, dataset_file.split("/")[-1])
    if os.path.exists(dataset_path):
        print(f"{dataset_path} already exists, skipping download.")
    else:
        subprocess.run(["wget", "-O", str(dataset_path), link])

    parquet_file = Path(out_dir, dataset_file.split("/")[-1])
    with duckdb.connect(parquet_file) as con:
        target_file_type = "wav"
        results = con.execute(f"SELECT audio.bytes, audio.path FROM '{parquet_file}'")
        print("Extracting audio from parquet file...")
        for b, p in results.fetchall():
            print(f"Extracting {p}")
            flac = AudioSegment.from_file(io.BytesIO(b), format='flac')
            dst_file_name = p.split("/")[-1].replace(".flac", f".{target_file_type}")
            dst_file_path = Path(out_dir, dst_file_name)
            audio_16k = flac.set_frame_rate(16000)
            audio_16k.export(dst_file_path, format=target_file_type)


def process_fma_dataset(fma_dataset: str = "benjamin-paine/free-music-archive-full", output_dir: str = "./fma"):
    """Process the Free Music Archive (FMA) dataset.

    The dataset contains a collection of music tracks in various genres and is commonly used for music information retrieval tasks.

    # Notes

    * The FMA dataset is available on [HuggingFace][hf-fma-replacement] and [GitHub][github-fma].
    * The original instructions used a [dataset][hf-fma-original] that has a Python loading sript which is no longer supported by HuggingFace. The replacement dataset is a streaming dataset that can be used to process the FMA dataset in a more efficient manner.

    [github-fma]: https://github.com/mdeff/fma
    [hf-fma-original]: https://huggingface.co/datasets/rudraml/fma
    [hf-fma-replacement]: https://huggingface.co/datasets/benjamin-paine/free-music-archive-full
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fma_dataset = datasets.load_dataset(
        fma_dataset,
        # name="small",
        split="train",
        streaming=True,
    )
    fma_dataset = iter(fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000)))

    n_hours = 5
    i = 0
    for row in fma_dataset:  # this works because the FMA dataset is all 30 second clips
        try:
            # row = next(fma_dataset)
            # print(f"Processing FMA dataset clip for file {row.title}...")
            title: str = row.get("title")
            audio: datasets._torchcodec.AudioDecoder = row.get("audio")
            audio_bytes = audio._hf_encoded.get("bytes")
            audio_path = audio.metadata.path
            audio_name = audio_path.split("/")[-1]
            source_codec = audio.metadata.codec
            dst_file = os.path.join(output_dir, audio_name.replace(f".{source_codec}", ".wav"))
            print(f"Processing FMA dataset clip for file {title} ({audio_name})...", end = " ")
            # Skip file if exists, to avoid re-processing the same file if the script is interrupted and restarted
            if os.path.exists(dst_file):
                print("File already exists, skipping.")
            else:
                flac = AudioSegment.from_file(io.BytesIO(audio_bytes), format=source_codec)
                audio_16k = flac.set_frame_rate(16000)
                audio_16k.export(dst_file,format="wav")
                print("done.")
        except StopIteration:
            print("Finished processing FMA dataset.")
            break
        i += 1
        if i == n_hours*3600//30:
            break

def process_FSD50k_dataset():
    """Process the FSD50k dataset.

    The FSD50k dataset is available on [HuggingFace][hf-fsd50k].
    The dataset contains a collection of audio clips in various
    categories and is commonly used for audio classification tasks.

    [hf-fsd50k]: https://huggingface.co/datasets/Fhrozen/FSD50k
    """
    raise NotImplementedError("FSD50k dataset processing is not yet implemented. Please refer to the HuggingFace documentation for details on how to process this dataset.")

def main():
    # process_room_impulse_responses()
    # process_google_audioset_dataset()
    process_fma_dataset()

if __name__ == "__main__":
    main()

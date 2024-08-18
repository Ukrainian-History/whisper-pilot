import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
from datetime import datetime
from functools import lru_cache
from itertools import product

import tqdm
import whisper
from pydub import AudioSegment

from . import utils

# These are whisper options that we want to perturb.
#
# The defaults are:
#   beam_size: 5
#   patience: 1
#   condition_on_previous_text: True
#   best_of: 5

whisper_options = {
    "model_name": ["medium", "large", "large-v3"],
    "beam_size": [5, 10],
    "patience": [1.0, 2.0],
    "condition_on_previous_text": [True, False],
    "best_of": [5, 10],
}

preprocessing_combinations = [
    "afftdn=nr=10:nf=-25:tn=1",
    "afftdn=nr=10:nf=-25:tn=1,volume=4",
    "anlmdn,volume=4",
    "highpass=200,lowpass=3000,afftdn",
    "volume=4",
    "speechnorm",
]


def run(output_dir, manifest):
    combinations = list(whisper_option_combinations())
    files = utils.get_data_files(manifest)
    total = len(combinations) * len(files)
    progress = tqdm.tqdm(total=total, desc="whisper".ljust(10))

    results = []
    for file_metadata in files:
        for options in combinations:
            file_metadata["run_count"] = len(results) + 1
            result = run_whisper(file_metadata, options, output_dir)
            results.append(result)
            progress.update(1)

    csv_filename = os.path.join(output_dir, "report-whisper.csv")
    utils.write_report(results, csv_filename, extra_cols=["options"])


def run_preprocessing(output_dir, manifest):
    results = []
    files = utils.get_data_files(manifest)
    total = len(files) * len(preprocessing_combinations)
    progress = tqdm.tqdm(total=total, desc="preprocess".ljust(10))

    for file_metadata in files:
        for combination in preprocessing_combinations:
            file = file_metadata["media_filename"]
            logging.info("preprocessing for file %s: %s", file, combination)
            preprocessed_file = (
                os.path.basename(file).rsplit(".", 1)[0]
                + "_filter_"
                + re.sub(r"[^A-Za-z0-9]", "", combination)
                + ".wav"
            )
            subprocess.Popen(
                f"ffmpeg -i {file} -af {combination} {preprocessed_file}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).communicate()
            result = run_whisper(
                {
                    **file_metadata,
                    "media_filename": preprocessed_file,
                    "run_count": len(results) + 1,
                },
                {
                    "model_name": "large",
                    "beam_size": 5,
                    "patience": 1,
                    "condition_on_previous_text": True,
                },
                output_dir,
            )
            result["ffmpeg filer"] = combination
            logging.info("result %s", result)
            results.append(result)

            os.remove(preprocessed_file)
            progress.update(1)

    csv_filename = os.path.join(output_dir, "report-whisper-preprocessing.csv")
    utils.write_report(results, csv_filename, extra_cols=["ffmpeg filer"])


def run_whisper(file_metadata, options, output_dir):
    start_time = datetime.now()
    file = file_metadata["media_filename"]
    logging.info("running whisper on %s with options %s", file, options)
    transcription = transcribe(file_metadata, options)
    runtime = utils.get_runtime(start_time)

    result = utils.compare_transcripts(
        file_metadata, transcription, "whisper", output_dir
    )

    result["druid"] = file_metadata["druid"]
    result["runtime"] = runtime
    result["options"] = str(options)

    # write out the json results
    with open(os.path.join(output_dir, f"{result['run_id']}.json"), "w") as fh:
        json.dump(transcription, fh, ensure_ascii=False)

    logging.info("result: %s", result)
    return result


def transcribe(file_metadata, options):
    model = load_model(options["model_name"])

    whisper_options = options.copy()
    whisper_options.pop("model_name")

    # if the languages of the source media and transcript are different and the
    # transcript is to be in English then we tell Whisper to translate
    if (
        file_metadata["media_language"] != file_metadata["transcript_language"]
        and file_metadata["transcript_language"] == "en"
    ):
        whisper_options["task"] = "translate"

    whisper_options["language"] = file_metadata["media_language"]
    audio = load_audio(file_metadata["media_filename"])

    return whisper.transcribe(audio=audio, model=model, **whisper_options)


def get_language(file, model_name):
    model = load_model(model_name)
    silences = get_silences(file)
    n_mels = 128 if model_name == "large" else 80

    if len(silences) > 0 and int(silences[0]["start_silence"]) == 0:
        audio_segment = AudioSegment.from_file(file)
        start_time = silences[0]["end_silence"] * 1000
        end_time = (30 + silences[0]["end_silence"]) * 1000
        clip = audio_segment[start_time:end_time]
        with tempfile.NamedTemporaryFile() as tmp:
            clip.export(tmp.name, format="wav")
            audio = load_audio(tmp.name)
    else:
        audio = load_audio(file)

    audioclip = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audioclip, n_mels=n_mels).to(model.device)
    _, probs = model.detect_language(mel)

    return max(probs, key=probs.get)


def get_silences(file):
    file = shlex.quote(file)
    p = subprocess.Popen(
        "ffmpeg -i {} -af 'volumedetect' -vn -sn -dn -f null /dev/null".format(file),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()
    detectvolume = p[-1].decode("utf-8")
    meanvolume = ffmpegcontentparse(detectvolume, "mean_volume")
    volume = meanvolume - 1 if meanvolume > -37 else -37
    p2 = subprocess.Popen(
        "ffmpeg -i {} -af silencedetect=n={}dB:d=0.5 -f null -".format(file, volume),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()
    startendsilences = []
    for item in p2[-1].decode("utf-8").split("\n"):
        if "silence_start" in item:
            startendsilences.append(
                {"start_silence": ffmpegcontentparse(item, "silence_start")}
            )
        elif "silence_end" in item:
            startendsilences[-1] = startendsilences[-1] | {
                "end_silence": ffmpegcontentparse(item.split("|")[0], "silence_end"),
                "duration": ffmpegcontentparse(item, "silence_duration"),
            }

    return startendsilences


def ffmpegcontentparse(content, field):
    lines = content.split("\n")
    correctline = [idx for idx, s in enumerate(lines) if field in s][0]
    get_value = lines[correctline].split(":")[-1]
    value_as_float = float(re.sub(r"[^0-9\-.]", "", get_value, 0, re.MULTILINE).strip())
    return value_as_float


@lru_cache(maxsize=1)
def load_model(model_name):
    # cache the response since it takes some time to load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(model_name, device=device)


@lru_cache(maxsize=1)
def load_audio(file):
    return whisper.load_audio(file)


def whisper_option_combinations():
    # generate a list of all possible combinations of the whisper option values
    for values in product(*whisper_options.values()):
        # generate a dict using the combination values and the original keys
        yield dict(zip(whisper_options.keys(), values))

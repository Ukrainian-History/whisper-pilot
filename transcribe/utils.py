import csv
import datetime
import glob
import os
import re
import string

import jiwer

base_csv_columns = [
    "file",
    "language",
    "runtime",
    "wer",
    "mer",
    "wil",
    "wip",
    "hits",
    "substitutions",
    "insertions",
    "deletions",
]


def get_files(bags_dir):
    # TODO: simplify this
    folder = f"{bags_dir}/*"
    files_with_transcript = []
    folders = glob.glob(folder + os.path.sep)
    files = [glob.glob("{}data/content/*_sl*.m*".format(folder)) for folder in folders]
    for folder in files:
        for file in sorted(folder):
            if (
                len(
                    glob.glob(f"{file.rsplit('.', 1)[0].replace('_sl', '')}*script.txt")
                )
                > 0
            ):
                files_with_transcript.append(file)
                break
    return list(sorted(files_with_transcript))


def get_reference_file(file, language):
    reference_files = glob.glob(
        f"{file.rsplit('.', 1)[0].replace('_sl', '')}*script.txt"
    )
    find_file = list(filter(lambda x: "_{}".format(language) in x, reference_files))
    reference_file = find_file if len(find_file) > 0 else reference_files
    return list(sorted(reference_file))[0]


def get_reference(file, language):
    return open(get_reference_file(file, language), "r").read()


def get_runtime(start_time):
    elapsed = datetime.datetime.now() - start_time
    return elapsed.total_seconds()


def write_report(rows, csv_path, extra_cols=[]):
    fieldnames = base_csv_columns.copy()
    if len(extra_cols) > 0:
        fieldnames.extend(extra_cols)
    with open(csv_path, "w") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compare_transcripts(reference, hypothesis):
    output = jiwer.process_words(clean_text(reference), clean_text(hypothesis))
    return {
        "wer": output.wer,
        "mer": output.mer,
        "wil": output.wil,
        "wip": output.wip,
        "hits": output.hits,
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
    }


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"  +", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = text.strip()
    return text

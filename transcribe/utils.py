import csv
import datetime
import difflib
import os
import re
import string
import textwrap
from collections import Counter
from io import StringIO

import jiwer
import webvtt

base_csv_columns = [
    "run_id",
    "druid",
    "file",
    "language",
    "transcript_filename",
    "transcript_language",
    "runtime",
    "wer",
    "mer",
    "wil",
    "wip",
    "hits",
    "substitutions",
    "insertions",
    "deletions",
    "diff",
]


def get_data_files(manifest):
    rows = []
    for row in csv.DictReader(open(manifest)):
        rows.append(row)
    return rows


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


def compare_transcripts(file, transcript, transcript_type, output_dir):
    """
    Compare the given file (a dictionary of file metadata, a row from data.csv).
    To the given transcript result, and the transcript_type in order to
    differentiate the different ways that results are represented. The
    output_dir is supplied because in addition to returning the comparison an
    HTML diff will be written to the output_dir.
    """
    run_id = f"{file['druid']}-{transcript_type}-{file['run_count']:03}"

    if transcript_type == "google":
        hypothesis, lang = parse_google(transcript)
    elif transcript_type == "aws":
        hypothesis, lang = parse_aws(transcript)
    elif transcript_type == "whisper":
        hypothesis, lang = parse_whisper(transcript)
    else:
        raise Exception("Unknown transcript type: {transcript_type}")

    reference = read_reference_file(file["transcript_filename"])

    stats = jiwer.process_words(clean_text(reference), clean_text(hypothesis))

    diff_file = f"{run_id}.html"
    diff_url = f"https://sul-dlss.github.io/whisper-pilot/{os.path.basename(output_dir)}/{diff_file}"
    diff_path = os.path.join(output_dir, diff_file)
    write_diff(file["druid"], reference, hypothesis, diff_path)

    return {
        "run_id": run_id,
        "druid": file["druid"],
        "file": os.path.basename(file["media_filename"]),
        "transcript_filename": os.path.basename(file["transcript_filename"]),
        "transcript_language": file["transcript_language"],
        "wer": stats.wer,
        "mer": stats.mer,
        "wil": stats.wil,
        "wip": stats.wip,
        "hits": stats.hits,
        "substitutions": stats.substitutions,
        "insertions": stats.insertions,
        "deletions": stats.deletions,
        "language": lang,
        "diff": diff_url,
    }


def read_reference_file(path):
    if path.endswith(".txt"):
        return open(path, "r", encoding="utf-8-sig").read().splitlines()
    elif path.endswith(".vtt"):
        return [caption.text for caption in webvtt.read(path)]
    else:
        raise Exception("Unknown reference transcription type {path}")


def write_diff(druid, reference, hypothesis, diff_path):
    from_lines = split_sentences(strip_rev_formatting(reference))
    to_lines = split_sentences(hypothesis)

    diff = difflib.HtmlDiff(wrapcolumn=80)
    diff = diff.make_file(from_lines, to_lines, "reference", "transcript")

    html = StringIO()
    html.writelines(diff)
    html = html.getvalue()

    # embed the media player for this item
    html = html.replace(
        "<body>",
        f'<body>\n\n    <div style="height: 200px;"><iframe style="position: fixed;" src="https://embed.stanford.edu/iframe?url=https://purl.stanford.edu/{druid}" height="200px" width="100%" title="Media viewer" frameborder="0" marginwidth="0" marginheight="0" scrolling="no" allowfullscreen="allowfullscreen" allow="clipboard-write"></iframe></div>',
    )

    # write the diff file
    open(diff_path, "w").write(html)

    return html


def parse_google(data):
    lines = [result["alternatives"][0]["transcript"] for result in data["results"]]
    lang_counts = Counter([result["languageCode"] for result in data["results"]])
    lang = lang_counts.most_common(1)[0][0]

    return lines, lang


def parse_whisper(data):
    lines = [segment["text"] for segment in data["segments"]]
    lang = data["language"]

    return lines, lang


def parse_aws(data):
    lines = [t["transcript"] for t in data["results"]["transcripts"]]
    lang = data["results"]["language_code"]

    return lines, lang


def wrap_lines(lines):
    """
    Fit text onto lines, which is useful if the text lacks any new lines, as
    is the case with Google and AWS transcripts. If we were processing VTT
    files this wouldn't be necessary.
    """
    new_lines = []
    for line in lines:
        new_lines.extend(textwrap.wrap(line.strip(), width=80))
    return new_lines


def clean_text(lines):
    """
    Normalize text for jiwer analysis.
    """
    text = " ".join(lines)
    text = text.replace("\n", " ")
    text = re.sub(r"  +", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = text.strip()
    return text


def strip_rev_formatting(lines):
    """
    Remove initial line formatting including optional diarization.

    So:

        - [Interviewer] And how far did you fall?

    would turn into:

        And how far did you fall?
    """
    new_lines = []
    for line in lines:
        line = re.sub(r"^- (\[.*?\] )?", "", line)
        new_lines.append(line)

    return new_lines


sentence_endings = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s")


def split_sentences(lines):
    """
    Split lines with multiple sentences into multiple lines. So,

        To be or not to be. That is the question.

    would become:

        To be or not to be.
        That is the question.
    """
    text = " ".join(lines)
    text = text.replace("\n", " ")
    text = re.sub(r" +", " ", text)
    sentences = sentence_endings.split(text.strip())
    sentences = [sentence.strip() for sentence in sentences]

    return sentences

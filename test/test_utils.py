import tempfile
from os import path

from transcribe import utils

TEST_DATA = path.join(path.dirname(__file__), "data")


def test_get_data_files():
    #TODO this will obviously fail...
    files = utils.get_data_files("data.csv")
    assert len(files) == 14
    assert path.basename(files[0]["media_filename"]) == "bb158br2509_sl.m4a"


def test_read_txt_reference_file():
    lines = utils.read_reference_file(path.join(TEST_DATA, "en.txt"))
    assert lines == ["This is a test for whisper reading in English."]


def test_read_vtt_reference_file():
    lines = utils.read_reference_file(path.join(TEST_DATA, "en.vtt"))
    assert lines == ["This is a test for whisper reading in English."]


def test_clean_text():
    assert utils.clean_text(["Makes Lowercase"]) == "makes lowercase"
    assert utils.clean_text(["Strips, punctuation."]) == "strips punctuation"
    assert utils.clean_text(["Removes  spaces"]) == "removes spaces"
    assert (
        utils.clean_text(["Removes    extra      spaces  "]) == "removes extra spaces"
    )
    assert (
        utils.clean_text(["removes\nall\nnewlines", "from\nall\nlines"])
        == "removes all newlines from all lines"
    )


def test_strip_rev_formatting():
    assert utils.strip_rev_formatting(["- [interviewer] hi there", "seeya"]) == [
        "hi there",
        "seeya",
    ]


def test_split_sentences():
    assert utils.split_sentences(
        ["Hiya.\n This is a test? This is another test... Onwards.\n", ""]
    ) == ["Hiya.", "This is a test?", "This is another test...", "Onwards."]

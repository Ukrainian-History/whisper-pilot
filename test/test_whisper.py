from os import path

from transcribe import whisper

MODEL_SIZE = "small"
TEST_DATA = path.join(path.dirname(__file__), "data")


def test_get_silences():
    assert whisper.get_silences(path.join(TEST_DATA, "en.wav")) == [
        {"start_silence": 2.70731, "end_silence": 3.22, "duration": 0.512687}
    ]
    assert whisper.get_silences(path.join(TEST_DATA, "en-with-silence.wav")) == [
        {"start_silence": 0.0, "end_silence": 35.1915, "duration": 35.1915}
    ]


def test_transcribe():
    t = whisper.transcribe(
        {
            "media_filename": path.join(TEST_DATA, "en.wav"),
            "media_language": "en",
            "transcript_language": "en",
        },
        {"model_name": MODEL_SIZE},
    )

    assert t[0].text == "This is a test for whisper reading in English."
    # assert t["language"] == "en"


def test_transcribe_fr():
    t = whisper.transcribe(
        {
            "media_filename": path.join(TEST_DATA, "fr.wav"),
            "media_language": "fr",
            "transcript_language": "fr",
        },
        {"model_name": MODEL_SIZE},
    )
    assert (
        t[0].text
        == "Il s'agit d'un test de lecture de Whispel en français."
    )
    # assert t["language"] == "fr"


def test_whisper_option_combinations():
    opts = list(whisper.whisper_option_combinations())
    assert len(opts) == 48
    assert {
        "model_name": "large",
        "beam_size": 10,
        "patience": 1.0,
        "condition_on_previous_text": False,
        "best_of": 10,
    } in opts

# whisper-pilot
 

This repository contains code for testing OpenAI's Whisper for generating transcripts from audio and video files. It is based on the tool of the same name from Stanford University Libraries modified for the specific needs of the Ukrainian History and Education Center. The UHEC does not have "ground truth" transcriptions to check for error rates, and whisper.cpp will be used instead of openai-whisper.

At this point, it is likely that this code is in a broken state while it is being reconfigured for the UHEC's purposes.

## Data

The data used in this analysis was determined ahead of time in this spreadsheet, which has a snapshot included in this repository as `uhec-data.csv`.

The audio files were manually constructed from UHEC preservation/production masters or appropriate mezzanine files. The audio files should be made available in a `data` directory that you create in the same directory you've cloned this repository to. Alternatively you can symlink the location to `data`

## Whisper Options

The whisper options that are perturbed as part of the run are located in the whisper module:

https://github.com/sul-dlss/whisper-pilot/blob/83292dc8f32bc30a003d0e71362ad12733f66473/transcribe/whisper.py#L27-L33

These could have been command line options or a separate configuration file, but we knew what we wanted to test. This is where to make adjustments if you do want to test additional Whisper options.

## Setup

Create or link your data directory:

```
$ ln -s /path/to/exported/data data
```

Create a virtual environment:

```
$ python -m venv env
$ source env/bin/activate
```

Install dependencies:

```
$ pip install -r requirements.txt
```

## Run

Then you can run the report:

```
$ ./run.py
```

If you just want to run one of the report types you can, for example only run preprocessing:

```
$ ./run --only preprocessing
```

## Test

To run the unit tests you should:

```
$ pytest
```

## Analysis

There are some Jupyter notebooks in the `notebooks` directory which you can view here on Github.

* [Caption Providers](https://github.com/sul-dlss/whisper-pilot/blob/main/notebooks/caption-providers.ipynb): an analysis of Word Error Rates for Whisper, Google Speech and Amazon Transcribe.
* [On Prem Estimate](https://github.com/sul-dlss/whisper-pilot/blob/main/notebooks/on-prem-estimate.ipynb): an estimate of how long it will take to run our backlog through Whisper using hardware similar to the RDS GPU work station.
* [Whisper Options](https://github.com/sul-dlss/whisper-pilot/blob/main/notebooks/whisper-options.ipynb) examining the effects of adjusting several Whisper options.

If you want to interact with them, you'll need to run Jupyter Lab which was installed with the dependencies:

```
$ jupyter lab
```

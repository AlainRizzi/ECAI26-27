# Lab 3 – Voice User Interface: Speech Recognition

Records audio from a USB microphone on a Raspberry Pi and turns it into text in two ways: in the cloud, by sending each second of audio to a Whisper server on Deepnote, and on the edge, by running Moonshine directly on the Pi. The goal is to compare how responsive the two are.

- [src/recorder.py](src/recorder.py) records 1-second blocks from the mic (48 kHz, mono, `int32`). Each block is converted to `float32`, resampled to 16 kHz, base64-encoded and sent to the cloud. The prediction is printed. The exercise 1 code that saved each block as a timestamped `.wav` is still in the callback, commented out.
- [src/api.py](src/api.py) has `send()`, which posts the audio to the Deepnote `/predict` endpoint and measures the round-trip time.
- [src/4.2-tts_api.py](src/4.2-tts_api.py) is the LitServe server that runs `openai/whisper-tiny.en` on CPU on port `8080`. It returns the transcription in lowercase and without punctuation. (The name says "TTS", but the model does speech-to-text.)
- [src/4.3 - Lab2 - Cloud Inference Server.ipynb](<src/4.3 - Lab2 - Cloud Inference Server.ipynb>) is the Deepnote notebook that starts the server.
- [src/transcribe.py](src/transcribe.py) runs Moonshine (`ModelArch.TINY_STREAMING`) on the Pi with `MicTranscriber`. It shows partial text while you speak and prints each finished line with its finalization latency. The language can be changed (see [Changing the language](#changing-the-language)).

## Hardware

- Raspberry Pi 4 Model B
- USB microphone, plugged into the Pi

## Setup

### Raspberry Pi (run once)

```bash
cd lab3

# Create a virtual env that can also see the modules already installed on the Pi
python3 -m venv --system-site-packages env

# Activate it
source env/bin/activate

# Install what the scripts need
pip install sounddevice numpy scipy requests moonshine-voice

# Check that everything imports
python -c "import sounddevice, numpy, scipy, requests, moonshine_voice; print('ok')"
```

Find the USB microphone's device index and set it as `device=` in [src/recorder.py](src/recorder.py) (it is `1` by default):

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Deepnote (run once)

1. Upload `4.2-tts_api.py` and the notebook to a private Deepnote project and open the notebook.
2. Start the machine and turn on **Incoming connections** in its settings.
3. Copy the API address and put it in [src/api.py](src/api.py), ending with `/predict`:

   ```python
   'https://<your-project-id>.deepnoteproject.com/predict'
   ```

## Run

### 1. Cloud ASR (Whisper on Deepnote)

Run the notebook cell `!python 4.2-tts_api.py` and leave it running. Then, on the Pi:

```bash
cd lab3/src
source ../env/bin/activate
python recorder.py
```

Every second, the callback sends the last block and prints what the server heard (`print(f'Prediction: {prediction}')`):

```text
Press Q to stop or P to pause/resume saving: Prediction: turn on the
Prediction: light in the kitchen
```

The text is lowercase with no punctuation, because the server removes it. Each prediction covers only one 1-second block, so a sentence can be split across lines. The first prediction appears on the same line as the prompt, because `input()` doesn't end the line.

Type `P` + Enter to pause or resume sending, and `Q` + Enter to stop. To print the round-trip time of each request, uncomment the `print` lines in `send()` in [src/api.py](src/api.py).

The server expects this request:

```json
{ "input": "<base64 of float32 samples>", "sampling_rate": 16000 }
```

and answers with:

```json
{ "output": "turn on the light" }
```

### 2. Edge ASR (Moonshine on the Pi)

```bash
cd lab3/src
source ../env/bin/activate
python transcribe.py
```

The first run downloads the model weights. Partial text updates in place while you speak, and each finished line is printed with its `Finalization latency`. Stop with `Ctrl+C`, and leave the env with `deactivate`.

#### Changing the language

Moonshine transcribes the language set with `.language(...)` in [src/transcribe.py](src/transcribe.py). It is currently Spanish:

```python
mic = (
    MicTranscriber()
    .language('es')   # change this, e.g. 'en' for English (the language the lab uses)
    .model_arch(ModelArch.TINY_STREAMING)
    ...
)
```

Change it to the language code you want and run the script again. The first run in a new language may need to download that language's model.

| Code | Language |
|------|----------|
| `ar` | Arabic |
| `de` | German |
| `en` | English |
| `es` | Spanish |
| `ja` | Japanese |
| `ko` | Korean |
| `tl` | Tagalog |
| `uk` | Ukrainian |
| `vi` | Vietnamese |
| `zh` | Chinese |

The cloud server can't switch languages: `openai/whisper-tiny.en` only understands English.

## Cloud vs edge

Try a few short commands and a few longer sentences with both scripts.

| | Cloud (Whisper) | Edge (Moonshine) |
|---|---|---|
| Where it runs | Deepnote, over HTTP | On the Pi |
| Latency measured | Round-trip time (network + inference) | Finalization latency per line |
| Output | One prediction per 1-second block | Partial text while speaking, then the full line |
| Needs internet | Yes | Only for the first download |

`.wav` and `.env` files are git-ignored.

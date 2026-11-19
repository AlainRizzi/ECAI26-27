import time
from moonshine_voice import MicTranscriber, ModelArch


def show_partial(text):
    print(f'\r{text}', end='', flush=True)


def show_line(line):
    print(f'\r{line.text}')
    print(f'Finalization latency: {line.last_transcription_latency_ms} ms')

def main():
    mic = (
        MicTranscriber()
        .language('es')
        .model_arch(ModelArch.TINY_STREAMING)
        .on_text(show_partial)
        .on_line(show_line)
    )

    mic.load()
    with mic:
        mic.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            mic.stop()

if __name__ == '__main__':
    main()

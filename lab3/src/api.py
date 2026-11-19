import requests
from statistics import median, stdev
from time import sleep, time
import os

import requests


def send(encoded, sampling_rate):
    start = time()
    try:
        response = requests.post(
            'https://15fedb15-1aa2-4858-870b-cef09ad44893.deepnoteproject.com/predict',
            json={'input': encoded, 'sampling_rate': sampling_rate},
        )
        response.raise_for_status()
        ret = response.json()
        elapsed_ms = (time() - start) * 1000
        # print(elapsed_ms)
        # print(f'Round-trip time: {elapsed_ms:.1f} ms')
        return ret
    except requests.exceptions.ConnectionError as e:
        print(f'Connection failed: {e}')
    except requests.exceptions.HTTPError as e:
        print(f'HTTP error occurred: {e}')
    except requests.exceptions.RequestException as e:
        print(f'An error occurred: {e}')
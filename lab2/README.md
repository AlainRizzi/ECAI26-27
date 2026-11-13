# Lab 2 – DHT11 → Redis

Reads temperature and humidity from a DHT11 sensor on a Raspberry Pi every 2 seconds and stores each reading in Redis as a time series.

- [DHT11Sensor.py](DHT11Sensor.py) reads the sensor on GPIO4 (`board.D4`) and uses the Pi's MAC address as the device id.
- [DB.py](DB.py) connects to Redis and writes each value to the keys `<mac>:temperature` and `<mac>:humidity`.
- [\_\_main\_\_.py](__main__.py) loads the `.env` file, then reads and stores in a loop.

## Wiring

| DHT11 | Raspberry Pi |
|-------|--------------|
| VCC   | 3.3V (pin 1) |
| DATA  | GPIO4 (pin 7) |
| GND   | GND (pin 6)  |

## Setup (on the Raspberry Pi, run once)

```bash
cd lab2

# Create a virtual env that can also see the modules already installed on the Pi
# (adafruit_dht, board, redis, ...)
python3 -m venv --system-site-packages env

# Activate it
source env/bin/activate

# Install python-dotenv inside the env
pip install python-dotenv

# Check that everything imports
python -c "import adafruit_dht, board, redis, dotenv; print('ok')"
```

Then create your `.env` from the template and fill in your Redis credentials:

```bash
cp .env.example .env
nano .env
```

`.env` is git-ignored, so your password is not committed.

## Run

```bash
cd lab2
source env/bin/activate
python __main__.py
```

You should see `Redis Connected: True` followed by one line per reading. Stop with `Ctrl+C`, and leave the env with `deactivate`.

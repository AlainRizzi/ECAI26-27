from DHT11Sensor import DHT11Sensor
from DB import DB
import time
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    sensor = DHT11Sensor()
    sensor.start()
    sensor.connect()
    db = DB()
    db.connect()
    while(True):
        id, reading = sensor.read()
        if reading is not None:
            print(id, reading)
            db.store(id, reading)
        time.sleep(2)

if __name__ == '__main__':
    main()
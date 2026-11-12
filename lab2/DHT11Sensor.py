import adafruit_dht
import uuid
import time
from datetime import datetime
from board import D4

class DHT11Sensor:
    def __init__(self):
        self.dht_device = None
        self.id = None
    
    def start(self):
        mac = uuid.getnode()
        self.id = ':'.join(f'{(mac >> i) & 0xff:02x}' for i in range(40, -1, -8))

    def connect(self):
        if self.dht_device is not None:
            self.dht_device.exit()
        self.dht_device = adafruit_dht.DHT11(D4)

    def read(self):
        try:
            timestamp = time.time()
            formatted_datetime = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
            temperature = self.dht_device.temperature
            humidity = self.dht_device.humidity
            return (self.id, {"temperature": temperature, "humidity": humidity})
        except:
            self.connect()
            return (self.id, None)
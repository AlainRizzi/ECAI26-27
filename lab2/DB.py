import redis
import time
import os

class DB:
    def __init__(self):
        self._redisHost = os.getenv("REDIS_HOST")
        self._redisPort = os.getenv("REDIS_PORT")
        self._redisUsername = os.getenv("REDIS_USERNAME")
        self._redisPassword = os.getenv("REDIS_PASSWORD")
        if(self._redisHost is None or self._redisPort is None or self._redisUsername is None or self._redisPassword is None):
            raise Exception("env vars not configured")

    def connect(self):
        self.redis_client = redis.Redis(
            host=self._redisHost,
            port=self._redisPort,
            username=self._redisUsername,
            password=self._redisPassword,
        )
        is_connected = self.redis_client.ping()
        print("Redis Connected:", is_connected)
        if not is_connected:
            raise Exception("Could not connect to DB")

    def store(self, id, readings):
        timestamp_ms = int(time.time() * 1000)

        for key in readings.keys():
            self.redis_client.ts().add(
                f"{id}:{key}", timestamp_ms, readings[key]
            )
import threading
from kafka import KafkaProducer, KafkaConsumer
from json import dumps, loads
from collections import namedtuple

data = namedtuple("news", ("title", "context"))

class Publisher():

    def __init__(self, topic_name, buffer_size=16):
        self.producer = KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda x: dumps(x).encode('utf-8'))
        
        self.topic_name = topic_name
        self.buffer_size = buffer_size
        self.run_thread = None
        self.stop = False
        self.queue = []

        self.run()

    def publish(self, msg):
        if len(self.queue) >= self.buffer_size:
            del self.queue[0]

        self.queue.append(msg)

    def run(self):
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.start()

    def close(self):
        if self.run_thread is not None:
            if self.stop is False:
                self.stop = True
            self.run_thread.join()

    def stop(self):
        self.stop = True

    def _run(self):
        while self.stop is False:
            self.producer.poll(0)

            if len(self.queue) > 0:
                msg = self.queue.pop(0)
                producer.produce(self.topic_name, f"{msg}".encode("utf-8"))
                producer.flush()
                print(f"[INFO] Producer produced at topic {self.topic_name}")


class KafkaHelper(object):
    @classmethod
    def _producer(cls):
        return KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda x: dumps(x).encode('utf-8'))

    @classmethod
    def _consumer(cls, topic):
        return KafkaConsumer(topic, bootstrap_servers=['localhost:9092'], value_deserializer=lambda x: loads(x.decode('utf8')))

    @classmethod
    def pub_noutput(cls, result):
        return cls._producer().send("noutput", result)

    @classmethod
    def pub_foutput(cls, result):
        return cls._producer().send("foutput", result)
    
    # consume functions are blockable til new record come
    @classmethod
    def consume_ninput(cls):
        record = next(iter(cls._consumer("ninput")))
        return record.value

    @classmethod
    def consume_ninput(cls):
        record = next(iter(cls._consumer("finput")))
        return record.value

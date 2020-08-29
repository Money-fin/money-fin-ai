from kafka import KafkaProducer, KafkaConsumer
from json import dumps, loads

class KafkaHelper(object):
    @classmethod
    def _producer(cls):
        return KafkaProducer(bootstrap_servers=['localhost:9092'], value_serializer=lambda x: dumps(x).encode('utf-8'))

    @classmethod
    def _consumer(cls, topic):
        # return KafkaConsumer(topic, bootstrap_servers=['localhost:9092'], value_deserializer=lambda x: loads(x.decode('utf8')))
        return KafkaConsumer(topic, bootstrap_servers=['localhost:9092'], value_deserializer=lambda x: loads(x.decode('utf-8-sig')))

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
    def consume_finput(cls):
        record = next(iter(cls._consumer("finput")))
        return record.value


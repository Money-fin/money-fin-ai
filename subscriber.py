from confluent_kafka import Consumer
import threading


class Subscriber():

    def __init__(self, topic_name, buffer_size=16):
        self.consumer = Consumer({
            # TODO: consumer config
        })
        self.consumer.subscribe([topic_name])

        self.topic_name = topic_name
        self.buffer_size = buffer_size

        self.queue = []
        self.stop = False

        self.run_thread = None

        self.run()

    def run(self):
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.start()

    def stop(self):
        self.stop = True

    def poll(self):
        if len(self.queue) > 0:
            return self.queue.pop(0)
        return None

    def close(self):
        if self.run_thread is not None:
            if self.stop is False:
                self.stop = True
            self.run_thread.join()

    def _run(self):
        while self.stop is False:
            msg = self.consumer.poll(1.0)

            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            print(f"[INFO] Message received in topic {self.topic_name}")

            self.queue.append(msg)
            if len(self.queue) > self.buffer_size:
                del self.queue[0]


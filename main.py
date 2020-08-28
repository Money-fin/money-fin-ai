import sys
import signal
from subscriber import Subscriber
from config import *


subscribers = None
publishers = None


def main():
    global subscribers, publishers

    signal.signal(signal.SIGINT, sighandler)

    subscribers = {
        "n_input": Subscriber("n_input", BUFFER_SIZE),
        "f_input": Subscriber("f_input", BUFFER_SIZE),
    }
    publishers = {
        "n_output": Publisher("n_output", BUFFER_SIZE),
        "f_output": Publisher("f_output", BUFFER_SIZE),
    }

def sighandler(signum, frame):
    print(f"[INFO] All subscribers and publishers are shutting down...")
    
    for sub in subscribers:
        sub.stop()
        sub.close()

    for pub in publishers:
        pub.stop()
        pub.close()

    print(f"[INFO] All subscribers and publishers are shutdowned")


if __name__ == "__main__":
    main()

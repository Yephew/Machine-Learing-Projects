# Threading.

import threading
import time
from queue import Queue
import numpy as np

def job(l,q):
    for i in range(len(l)):
        l[i] = l[i] ** 2
    q.put(l)

def multithreading():
    q = Queue()
    threads = []
    data = list(np.arange(10000).reshape(100,100))
    for i in range(4):
        t = threading.Thread(target=job, args=(data[i], q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    results = []
    for _ in range(4):
          results.append(q.get())
    print(results)

if __name__ == '__main__':
    multithreading()

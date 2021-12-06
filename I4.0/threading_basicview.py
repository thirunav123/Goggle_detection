import threading
import time
x = 0

def thread_task(a):
    global x
    for i in range(3):
        x += a
        time.sleep(3)
t1 = threading.Thread(target=thread_task, args=(1,))
t1.start()
while True:
    
    print(t1.is_alive())
    if not t1.is_alive():
        t1 = threading.Thread(target=thread_task, args=(1,))
        print('Not alive, started')
        t1.start()
    else:
        print("Alive{0}".format(x))
    time.sleep(1)
import torch.multiprocessing as mp
import time

def put(q):
    for i in range(100):
        time.sleep(1)
        q.put(i)
        
if __name__=="__main__":
    q = mp.Queue()
    pp = mp.Process(target=put, args=(q,))
    pp.start()
    while True:
        print("hi")
        time.sleep(3)
        while not q.empty():
            print(q.get())
            
        
        
        
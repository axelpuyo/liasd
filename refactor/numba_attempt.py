import time
from numba import jit

def bar():
    for n in range(100000000):
        if n == 50:
            print('mid')
s = time.time()
bar()
e = time.time()
print(e - s)

@jit
def foo():
    for n in range(100000000):
        if n == 50:
            print('parmid')

s = time.time()
foo()
e = time.time()
print(e - s)
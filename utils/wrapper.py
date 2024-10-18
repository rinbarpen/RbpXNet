import pdb

import warnings
import threading

def deprecated(f):
    def wrapper(*args, **kwargs):
        warnings.warn(f'{f.__name__} is deprecated.', DeprecationWarning, stacklevel=2)
        return f(*args, **kwargs)
    return wrapper


def thread(name=None, daemon=True):
    def wrapper(f):
        def inner_wrapper(*args, **kwargs):
            t = threading.Thread(target=f, name=name, daemon=daemon, args=args, kwargs=kwargs)
            t.start()
            return t
        return inner_wrapper
    return wrapper

# @thread(name='K')
# def add(x, y):
#     tid = threading.current_thread().native_id
#     tname = threading.current_thread().name
#     print(f'{tid=}|{tname=}: {x + y=}')

# add(1, 2)

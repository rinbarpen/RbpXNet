import pdb

import warnings

def deprecated(f):
    def wrapper(*args, **kwargs):
        warnings.warn(f'{f.__name__} is deprecated.', DeprecationWarning, stacklevel=2)
        return f(*args, **kwargs)
    return wrapper

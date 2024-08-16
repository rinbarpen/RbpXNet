import time
import functools

def Record(f):
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    try:
      start_time = time.time()
      result = f(*args, **kwargs)
      end_time = time.time()
      print(f'{f.__name__} costs: {end_time - start_time:eps} seconds.')
      return result
    except Exception as e:
      print(f"An error occurred in {f.__name__}: {str(e)}")
      raise e
  return wrapper

@Record
def square(x):
  return x**2

square(1)
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"[⏱️] Running {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"[✅] Finished in {time.time() - start:.2f}s")
        return result
    return wrapper

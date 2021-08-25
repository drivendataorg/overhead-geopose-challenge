import time
from contextlib import contextmanager


@contextmanager
def timer(name, logger=None):
    t0 = time.time()

    if logger:
        logger.info(f'[{name}] start.')
    else:
        print(f'[{name}] start.')
    yield
    if logger:
        logger.info(f'[{name}] done in {time.time() - t0:.0f} s')
    else:
        print(f'[{name}] done in {time.time() - t0:.0f} s')

import logging
from tqdm.auto import tqdm
from joblib import Parallel

# To do : maybe shift to a Tensorflow preprocessing layer!

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def custom_logger():
    logger = logging.getLogger()
    logger.setLevel('INFO')
    blue = '\u001b[30m'
    yellow = "\x1b[33;20m"
    reset = "\x1b[0m"
    fmt = blue+'%(asctime)s'+reset+'--'+yellow+'%(levelname)-5s'+reset+': %(message)s'
    fmt_date = '%H:%M:%S'
    formatter = logging.Formatter(fmt, fmt_date)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":
    logger = custom_logger()
    logger.info("Hello Word")
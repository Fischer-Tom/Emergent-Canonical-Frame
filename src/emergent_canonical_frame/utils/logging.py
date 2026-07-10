import logging
import sys
from collections import defaultdict, deque
from pathlib import Path


class Logger:

    def __init__(
        self,
        rank: int = 0,
        name: str = "emergent_canonical_frame",
        window: int = 20,
        log_file: Path | str | None = None,
    ):
        self.rank = rank
        self._logger = logging.getLogger(name)
        if rank == 0 and not self._logger.handlers:
            fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)
            if log_file is not None:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(fmt)
                self._logger.addHandler(file_handler)
            self._logger.setLevel(logging.INFO)
        elif rank != 0:
            self._logger.setLevel(logging.CRITICAL + 1)
        self._history = defaultdict(lambda: deque(maxlen=window))

    def info(self, msg):
        self._logger.info(msg)

    def error(self, msg):
        self._logger.error(msg)

    def log_dict(self, d: dict, step: int = -1):
        prefix = f"[iter {step}] " if step >= 0 else ""
        for k, v in d.items():
            val = v.item() if hasattr(v, "item") else v
            hist = self._history[k]
            hist.append(val)
            smoothed = sum(hist) / len(hist)
            self._logger.info(f"{prefix}{k}: {smoothed:.4f}")

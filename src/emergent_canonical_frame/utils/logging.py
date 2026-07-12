import logging
import sys
import time
from collections import defaultdict, deque
from pathlib import Path


class Logger:

    def __init__(
        self,
        rank: int = 0,
        name: str = "emergent_canonical_frame",
        window: int = 100,
        log_file: Path | str | None = None,
        total_iters: int | None = None,
    ):
        self.rank = rank
        self.total_iters = total_iters
        self.start_time = time.time()
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

    def log_dict(self, d: dict, step: int = -1, *, emit: bool = True):
        for k, v in d.items():
            val = v.item() if hasattr(v, "item") else v
            self._history[k].append(float(val))

        if not emit:
            return

        elapsed = time.time() - self.start_time
        total = self.total_iters if self.total_iters is not None else "?"
        speed = step / elapsed if step >= 0 and elapsed > 0 else 0.0
        lines = [
            f"Training Summary - Iteration {step}/{total}",
            f"Elapsed: {elapsed / 3600:.2f}h | Speed: {speed:.2f} iter/s",
        ]
        for key, hist in self._history.items():
            lines.append(f"{key}: {sum(hist) / len(hist):.4g}")
        self._logger.info("\n" + "\n".join(lines))

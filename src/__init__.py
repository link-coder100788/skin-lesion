import os
import sys

# On macOS, torch and xgboost each ship their own libomp; with both loaded, multithreaded
# OpenMP regions segfault. One OpenMP thread avoids it (heavy work runs on MPS anyway).
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

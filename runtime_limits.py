"""Keep the app's memory footprint small. Import this before numpy or TensorFlow.

On a 12-core machine OpenBLAS reserves one work buffer per core as soon as numpy is
imported (about 750 MB committed before any model loads), and TensorFlow/oneDNN add
per-thread scratch space on top. On a laptop with little free memory that made the
app fail with "OpenBLAS error: Memory allocation still failed".

Measured on this project (Windows, 12 logical CPUs, both ResNet50 models loaded and one
scan analysed): 1,658 MB committed with defaults vs 668 MB with these limits, with the
same prediction. A single uploaded slice gains almost nothing from more threads.

Each value is only a default: set the environment variable yourself to override it.
"""

import os

THREADS = "2"
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS"):
    os.environ.setdefault(_name, THREADS)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # oneDNN scratch buffers; results are unchanged
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")  # hide TensorFlow's informational start-up lines

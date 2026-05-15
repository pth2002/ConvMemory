"""Train or export a LoCoMo ConvMemory checkpoint.

This is a thin public training entrypoint over `reproduce_locomo.py`. It keeps a
clear `train_*` command in the repository while preserving the tested training
implementation and all of its existing flags.

Example:

python experiments/train_locomo.py \
  --device cuda \
  --data data/locomo10.json \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --teacher-cache results/cache/repro_teacher_mpnet_top500_seed23.json \
  --epochs 1 \
  --seed 23 \
  --save-pretrained checkpoints/convmemory-locomo-mpnet \
  --out results/train_locomo_seed23
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reproduce_locomo import main


if __name__ == "__main__":
    main()

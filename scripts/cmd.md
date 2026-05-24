SRC="/root/.cache/hippius/hub/models--a51--teutonic-q3-4b-5hgmbeef-t5108/snapshots/t5108-long-raw131k-lr3e6-20260522T120031Z"
DST="/root/teutonic/s1-work/merged"

mkdir -p "$DST"
cp -av "$SRC/"* "$DST/"

push

hippius-hub upload

hippius-hub login --hippius-token 52c631b308b1bc231448fc216725472cf961c9d2

boris-ai-dev/Teutonic-LXXX-5Ev4MnNC-vv02

hippius-hub download teutonic/teutonic-q3-4b-genesis <filename> --revision main

5Ev4MnNC
5ev4mnnc

hippius-hub upload yoko/teutonic-q3-4b-5ev4mnnc-v13 /root/.cache/hippius/hub/models--scoutminer--teutonic-q3-4b-5hbdijfd-auto18/snapshots/auto18 --revision v13

mkdir -p teutonic-q3-download
cd teutonic-q3-download

python -m venv .venv
source .venv/bin/activate

pip install hippius-hub

REPO="aetheling/teutonic-q3-4b-5cdd5hdj-v5"
REV="main"

for f in \
  model.safetensors \
  tokenizer.json \
  model.safetensors.index.json \
  tokenizer_config.json \
  config.json \
  generation_config.json
do
  echo "Downloading $f..."
  hippius-hub download "$REPO" "$f" --revision "$REV"
done

cp -avL /root/.cache/hippius/hub/models--aetheling--teutonic-q3-4b-5cdd5hdj-v5/snapshots/main /root/teutonic/s1-work

python submit_challenger.py \
  --verdict /root/teutonic/s1-work/verdict.json \
  --hotkey default \
  --wallet-name silvanus-hs1 \
  --netuid 3 \
  --network finney


  hf upload silvanus0930/Teutonic-Q3-4B-Prod10T /root/teutonic/qwen3-fineweb-edu-prod10 . --repo-type dataset --commit-message "Upload"

export LOCAL_DATASET_MANIFEST="/path/to/your/5shard/manifest.json"

export LOCAL_KING_DIR="/root/teutonic/s1-work/auto18"
export TEUTONIC_SIM_HOTKEY="5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv"
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO="Qwen/Qwen3-4B"

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-v2 \
  --bundle /root/teutonic/scripts/training_bundle \
  --dataset-mode auto \
  --n-shards 4 --shard-start 0 --eval-shard 4 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --n-score 10000 \
  --train-per-iter 5000 \
  --val-size 500 \
  --n-eval 1000 \
  --eval-gpus 0 \
  --n-gpus 1 \
  --micro-batch 2 \
  --grad-accum 16 \
  --lr 2e-5 \
  --epochs 1.0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --max-iters 1 \
  --report-out /root/teutonic/s1-work/verdict.json

  python -u scripts/mining/validator_eval.py \
  --king "${LOCAL_KING_DIR}" \
  --challenger /root/teutonic/s1-work-v2/iter_00/merged \
  --hotkey "${TEUTONIC_SIM_HOTKEY}"

---

## Resume without re-scoring (after crash)

If scoring finished (`iter_00/train.jsonl` exists) but training crashed:

```bash
export LOCAL_DATASET_MANIFEST="/root/teutonic/dataset/manifest.json"
export LOCAL_KING_DIR="/root/teutonic/s1-work/main"
export TEUTONIC_SIM_HOTKEY="5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv"
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-v2 \
  --bundle scripts/training_bundle \
  --skip-scoring \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --n-gpus 1 --micro-batch 2 --grad-accum 16 \
  --lr 2e-5 --epochs 1.0 --lora-r 32 --lora-alpha 64 \
  --max-iters 1 \
  --report-out /root/teutonic/s1-work/verdict.json
```

Skips ~18 min scoring and does **not** reload 4×2G shards.


python scripts/mining/submit_challenger.py \
  --verdict "/root/teutonic/s1-work/verdict.json" \
  --wallet-name silvanus-hs2 \
  --hotkey hotkey30 \
  --netuid 3 \
  --network finney 
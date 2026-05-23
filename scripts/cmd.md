SRC="/root/.cache/hippius/hub/models--a51--teutonic-q3-4b-5hgmbeef-t5108/snapshots/t5108-long-raw131k-lr3e6-20260522T120031Z"
DST="/root/teutonic/s1-work/merged"

mkdir -p "$DST"
cp -av "$SRC/"* "$DST/"

push

hippius-hub upload

hippius-hub login --hippius-token 52c631b308b1bc231448fc216725472cf961c9d2

boris-ai-dev/Teutonic-LXXX-5Ev4MnNC-vv02

hippius-hub download teutonic/teutonic-q3-4b-genesis <filename> --revision main

hippius-hub upload yoko/teutonic-q3-4b-5Ev4MnNC-v1 /root/teutonic/s1-work/merged --revision v1

mkdir -p teutonic-q3-download
cd teutonic-q3-download

python -m venv .venv
source .venv/bin/activate

pip install hippius-hub

REPO="teutonic/teutonic-q3-4b-genesis"
REV="main"

for f in \
  model-00002-of-00003.safetensors \
  model-00001-of-00003.safetensors \
  model-00003-of-00003.safetensors \
  tokenizer.json \
  model.safetensors.index.json \
  tokenizer_config.json \
  config.json \
  generation_config.json
do
  echo "Downloading $f..."
  hippius-hub download "$REPO" "$f" --revision "$REV"
done

cp -avL /root/.cache/hippius/hub/models--teutonic--teutonic-q3-4b-genesis/snapshots/main /root/teutonic/s1-work

python submit_challenger.py \
  --verdict /root/teutonic/s1-work/verdict.json \
  --hotkey hotkey25 \
  --wallet-name silvanus-hs1 \
  --netuid 3 \
  --network finney


  hf upload silvanus0930/Teutonic-Q3-4B-Prod10T /root/teutonic/qwen3-fineweb-edu-prod10 . --repo-type dataset --commit-message "Upload"

  export LOCAL_DATASET_MANIFEST="/path/to/your/5shard/manifest.json"
export LOCAL_KING_DIR="/path/to/uid47-king"   # sync from dashboard
export TEUTONIC_SIM_HOTKEY="YOUR_HOTKEY_SS58"
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO="Qwen/Qwen3-4B"

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-v2 \
  --bundle /root/teutonic/scripts/training_bundle \
  --dataset-mode auto \
  --n-shards 4 --shard-start 0 --eval-shard 4 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --n-score 12000 \
  --train-per-iter 10000 \
  --val-size 500 \
  --n-eval 5000 \
  --eval-gpus 0 \
  --n-gpus 1 \
  --micro-batch 2 \
  --grad-accum 16 \
  --lr 1e-4 \
  --epochs 3.0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --max-iters 5 \
  --target-mu 0.015 \
  --report-out /root/teutonic/s1-work-v2/verdict.json
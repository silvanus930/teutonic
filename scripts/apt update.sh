apt update
apt install -y pipx
pipx ensurepath
source ~/.bashrc
pipx install bittensor-cli
source ~/.bashrc

git clone https://github.com/unarbos/teutonic.git
cd teutonic

python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -r ./scripts/mining/requirements.txt

cp -avL /root/.cache/hippius/hub/models--iris999--teutonic-q3-4b-5cqlruvc-base-cpt-r64s200/snapshots/5cqlruvc-base-cpt-r64s200 /root/teutonic/s1-work

btcli subnet register --netuid 3 --wallet.name silvanus-hs1 --hotkey default
btcli subnet register --netuid 3 --wallet.name silvanus-hs1 --hotkey hotkey1
btcli wallet overview --netuid 3 --wallet.name silvanus-hs1

export HF_XET_HIGH_PERFORMANCE=1

export LOCAL_KING_DIR="/root/teutonic/s1-work/merged"
export LOCAL_DATASET_MANIFEST="/root/teutonic/qwen3-fineweb-edu-npy-big/manifest.json"
python /root/teutonic/scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work \
  --bundle /root/teutonic/scripts/training_bundle \
  --n-gpus 1 \
  --n-shards 8 \
  --shard-start 0 \
  --eval-shard 14 \
  --n-eval 500 \
  --n-score 5000 \
  --train-per-iter 10000 \
  --val-size 200 \
  --max-iters 1 \
  --micro-batch 2 \
  --grad-accum 8 \
  --lr 2e-4 \
  --epochs 1 \
  --lora-r 16 \
  --lora-alpha 32

hippius-hub login --hippius-token 52c631b308b1bc231448fc216725472cf961c9d2
eaa8fNGT1yS5svPvHiEftGRKwx66ZcUvsm1SfXYc

hippius-hub upload yoko/teutonic-q3-4B-5Ev4MnNC-v1 /root/teutonic/s1-work/merged --revision v1
hippius-hub upload yoko/teutonic-q3-4b-5fxjcgb1-v1 /root/teutonic/s1-work/iter_00/merged --revision v1
hippius-hub upload yoko/teutonic-q3-4b-5fxjcgb1-v2 /root/teutonic/s1-work/iter_00/merged --revision v2


hippius-hub login --username 'robot$yoko+yoko-bot' --password 'IeNszjh538f3DxCjh6EnBSyzsVUO7qmB'
hippius-hub login --hippius-token 52c631b308b1bc231448fc216725472cf961c9d2


btcli subnet register --netuid 3 --wallet.name silvanus-hs1 --hotkey hotkey31
btcli subnet register --netuid 3 --wallet.name silvanus-hs1 --hotkey hotkey32

# Dry-run first
python /root/teutonic/scripts/mining/submit_challenger.py \
  --verdict /root/teutonic/s1-work/verdict.json \
  --wallet-name silvanus-hs2 \
  --hotkey hotkey4 \
  --netuid 3 \
  --network finney


cd /root/teutonic
source .venv/bin/activate

export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO="Qwen/Qwen3-4B"
export TEUTONIC_RAW_MAX_FILES_PER_EVAL=32
export TEUTONIC_SIM_HOTKEY="5FxJCGB1eaa8fNGT1yS5svPvHiEftGRKwx66ZcUvsm1SfXYc"

python -u scripts/mining/validator_eval.py \
  --king /root/teutonic/s1-work/5gzdd7oy-v3-s400 \
  --challenger /root/teutonic/s1-work-v3/iter_00/merged \
  --report-out /root/teutonic/s1-work-v3/verdict.json


  cd /root/teutonic

# Do NOT point at local .npy shards — eval does not use them.
unset LOCAL_DATASET_MANIFEST

# Current on-chain king (download or copy merged base weights here).
export LOCAL_KING_DIR="/root/teutonic/s1-work/5gzdd7oy-v3-s400"   # <-- sync from dashboard king digest

# Match ecosystem.config.js / eval_server.py
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_DATASET_PREFIX="hf-mirrors/HuggingFaceFW/fineweb-edu/data"
export TEUTONIC_RAW_DATASET_MANIFEST="hf-mirrors/HuggingFaceFW/fineweb-edu/data/_manifest.json"
export TEUTONIC_RAW_TOKENIZER_REPO="Qwen/Qwen3-4B"
export TEUTONIC_RAW_MAX_FILES_PER_EVAL=24   # mining needs ~12k+ seqs; eval uses 8 for 5k

# HF only for tokenizer download (not for king if LOCAL_KING_DIR is set)
export HF_TOKEN="$(cat /root/teutonic/.hf_token)"   # or: export HF_TOKEN="$(cat /root/teutonic/.hf_token)"

# Hippius upload after training (lowercase repo path)
# Coldkey prefix 5FxJCGB1 must appear in the repo name (validator gate).
export UPLOAD_REPO="yoko/teutonic-q3-4b-5fxjcgb1-v3"

cd /root/teutonic

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work \
  --bundle /root/teutonic/scripts/training_bundle \
  --dataset-mode auto \
  --raw-max-files 24 \
  --n-shards 1 \
  --shard-start 0 \
  --eval-shard 0 \
  --n-gpus 1 \
  --micro-batch 2 \
  --grad-accum 16 \
  --n-score 3500 \
  --train-per-iter 5000 \
  --val-size 400 \
  --n-eval 2500 \
  --max-iters 3 \
  --target-mu 0.01 \
  --lr 2e-4 \
  --epochs 1.0 \
  --lora-r 16 \
  --lora-alpha 32 \
  --seed 42 \
  --report-out /root/teutonic/s1-work/verdict1.json

  cd /root/teutonic
source .venv/bin/activate

# Your 5× ~2G Qwen shards (directory with manifest.json + shard_*.npy)
export LOCAL_DATASET_MANIFEST="/root/teutonic/qwen3-fineweb-edu-prod10/manifest.json"

# Current on-chain king (sync from dashboard before each run)
export LOCAL_KING_DIR="/root/teutonic/s1-work/5gzdd7oy-v3-s400"

# Validator holdout seeds (your submit hotkey)
export TEUTONIC_SIM_HOTKEY="5FxJCGB1eaa8fNGT1yS5svPvHiEftGRKwx66ZcUvsm1SfXYc"

# Still set for validator eval inside train_challenger
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO="Qwen/Qwen3-4B"
export TEUTONIC_RAW_MAX_FILES_PER_EVAL=32

export HF_TOKEN="$(cat /root/teutonic/.hf_token)"   # tokenizer / HF if needed

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-v3 \
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
  --report-out /root/teutonic/s1-work-v3/verdict.json


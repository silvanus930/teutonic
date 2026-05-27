SRC="/root/.cache/hippius/hub/models--a51--teutonic-q3-4b-5hgmbeef-t5108/snapshots/t5108-long-raw131k-lr3e6-20260522T120031Z"
DST="/root/teutonic/s1-work/merged"

mkdir -p "$DST"
cp -av "$SRC/"* "$DST/"

push

hippius-hub upload

hippius-hub login --hippius-token 52c631b308b1bc231448fc216725472cf961c9d2
hippius-hub registry rotate-token --docker-login

boris-ai-dev/Teutonic-LXXX-5Ev4MnNC-vv02

hippius-hub download teutonic/teutonic-q3-4b-genesis <filename> --revision main

5Ev4MnNC
5ev4mnnc

hippius-hub upload yoko/teutonic-q3-8b-5ev4mnnc-v20 /root/teutonic/s1-work-v8/iter_00/merged --revision v20_2
hf download silvanus0930/Teutonic-Q3-8B-Test /root/teutonic/s1-work/merged

hf download "silvanus0930/Teutonic-Q3-8B-Test" --local-dir "/root/teutonic/s1-work/merged"

mkdir -p teutonic-q3-download
cd teutonic-q3-download

python -m venv .venv
source .venv/bin/activate

pip install hippius-hub

python <<'EOF'
from hippius_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="mastertensor/teutonic-q3-8b-5ek5koe5-11x3977-rn",
    revision="5Ek5KoE5-11x3977-rn",
    allow_patterns=["*.safetensors", "*.json"],
    ignore_patterns="optimizer*",
    max_workers=8,
)
print(local_dir)
EOF

REPO="mastertensor/teutonic-q3-8b-5ek5koe5-62x4059-rn"
REV="5Ek5KoE5-62x4059-rn"

for f in \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  tokenizer.json \
  model.safetensors.index.json \
  tokenizer_config.json \
  config.json \
  generation_config.json
do
  echo "Downloading $f..."
  hippius-hub download "$REPO" "$f" --revision "$REV"
done

tech-dev-ai/Teutonic-VIII-5CXiauzN-cru1

cp -avL /root/.cache/hippius/hub/models--mastertensor--teutonic-q3-8b-5ek5koe5-22x3581-rn/snapshots/5Ek5KoE5-22x3581-rn /root/teutonic/s1-work

python submit_challenger.py \
  --verdict /root/teutonic/s1-work/verdict.json \
  --hotkey default \
  --wallet-name silvanus-hs1 \
  --netuid 3 \
  --network finney


  hf upload silvanus0930/Teutonic-Q3-4B-Prod10T /root/teutonic/qwen3-fineweb-edu-prod10 . --repo-type dataset --commit-message "Upload"
  hf upload silvanus0930/Teutonic-Q3-8B-Test /root/.cache/hippius/hub/models--mastertensor--teutonic-q3-8b-5ek5koe5-11x3977-rn/snapshots/5Ek5KoE5-11x3977-rn . --repo-type model --commit-message "Upload"

  hf download taoism99/Teutonic-VIII-5FnnkHKa-t01 /root/teutonic/s1-work/merged123
  hf download silvanus0930/Bad-set --local-dir /root/teutonic/s1-work/dataset

  hf download silvanus0930/Bad-set \
  --repo-type dataset \
  --local-dir /root/teutonic/s1-work/dataset

//pretokenized 

export LOCAL_DATASET_MANIFEST="/root/teutonic/s1-work/dataset/manifest.json"
export LOCAL_KING_DIR="/root/.cache/hippius/hub/models--mastertensor--teutonic-q3-8b-5ek5koe5-62x4059-rn/snapshots/5Ek5KoE5-62x4059-rn"
export TEUTONIC_SIM_HOTKEY="5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv"
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO="Qwen/Qwen3-4B"
export TEUTONIC_RAW_MAX_FILES_PER_EVAL=32

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-v8 \
  --bundle /root/teutonic/scripts/training_bundle \
  --dataset-mode auto \
  --n-shards 10 --shard-start 0 --eval-shard 15 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --n-score 1000 \
  --train-per-iter 800 \
  --val-size 200 \
  --n-eval 200 \
  --eval-gpus 0 \
  --n-gpus 1 \
  --micro-batch 8 \
  --grad-accum 4 \
  --lr 6e-6 \
  --epochs 1.0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --max-iters 1 \
  --report-out /root/teutonic/s1-work/verdict.json

# Hippius raw parquet holdout (download + tokenize)
python -u scripts/mining/validator_eval.py \
  --king /root/teutonic/s1-work-v5/king \
  --challenger /root/.cache/hippius/hub/models--mastertensor--teutonic-q3-8b-5ek5koe5-44x7285-rn/snapshots/5Ek5KoE5-44x7285-rn \
  --n-public 500 \
  --hotkey 5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv \
  --mix 4

  --king /root/.cache/hippius/hub/models--mastertensor--teutonic-q3-8b-5ek5koe5-44x15505-rn/snapshots/5Ek5KoE5-44x15505-rn \
  --king /root/teutonic/s1-work-v7/king \

python -u scripts/mining/validator_eval.py \
  --king /root/teutonic/s1-work-v7/merged \
  --challenger /root/.cache/hippius/hub/models--mastertensor--teutonic-q3-8b-5ek5koe5-62x4059-rn/snapshots/5Ek5KoE5-62x4059-rn \
  --n-public 500 \
  --local-dataset /root/teutonic/s1-work/dataset \
  --hotkey 5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv \
  --local-dataset /root/teutonic/s1-work/dataset \
  --mix 4

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
  --uploaded-repo yoko/teutonic-q3-8b-5ev4mnnc-v20 \
  --uploaded-digest sha256:3f9754272ada8ab3f1fdecedc6412ecf6b24b285750293977db82b4683c9b981 \
  --wallet-name silvanus-hs1 \
  --hotkey hotkey33 \
  --netuid 3 \
  --network finney
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

hippius-hub upload yoko/teutonic-5ev4mnnc-012 /root/teutonic/s1-work/iter_00/merged --revision main
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

REPO="mastertensor/teutonic-q3-10b-5ek5koe5-40037154544-rn"
REV="5Ek5KoE5-40037154544-rn"

for f in \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors \
  modeling_qwen3_5.py \
  configuration_qwen3_5.py \
  model.safetensors.index.json \
  config.json \
  generation_config.json
do
  echo "Downloading $f..."
  hippius-hub download "$REPO" "$f" --revision "$REV"
done

tech-dev-ai/Teutonic-VIII-5CXiauzN-cru1

cp -avL /root/.cache/hippius/hub/models--kyou--teutonic-5g1dh3iz-test119/snapshots/test119 /root/teutonic/s1-work

python submit_challenger.py \
  --verdict /root/teutonic/s1-work/verdict.json \
  --hotkey default \
  --wallet-name silvanus-hs1 \
  --netuid 3 \
  --network finney


  hf upload silvanus0930/Teutonic-Q3-8B-Full /root/teutonic/dataset/fineweb_edu_qwen3_2048 . --repo-type dataset --commit-message "Upload"
  hf upload silvanus0930/Teutonic-Q3-8B-Test /root/.cache/hippius/hub/models--mastertensor--teutonic-q3-8b-5ek5koe5-11x3977-rn/snapshots/5Ek5KoE5-11x3977-rn . --repo-type model --commit-message "Upload"

  hf download silvanus0930/Teutonic-Q35-8B-Test --local-dir /root/teutonic/s1-work/dataset --repo-type dataset
  hf download tech-dev-ai/teutonic-5g6x3hrj-top30 --local-dir /root/teutonic/s1-work/temp

  hf download silvanus0930/Bad-set \
  --repo-type dataset \
  --local-dir /root/teutonic/s1-work/dataset

//pretokenized local shards (fineweb_edu_qwen3_2048: 4 train + 1 eval)

export LOCAL_DATASET_MANIFEST="/root/teutonic/s1-work/dataset/manifest.json"
export LOCAL_KING_DIR="/root/teutonic/s1-work/king1"
export TEUTONIC_SIM_HOTKEY="5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv"
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO="${LOCAL_KING_DIR}" 
export TEUTONIC_RAW_MAX_FILES_PER_EVAL=4

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work \
  --bundle /root/teutonic/scripts/training_bundle \
  --dataset-mode local \
  --local-dataset-manifest "${LOCAL_DATASET_MANIFEST}" \
  --n-shards 4 --shard-start 0 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --n-score 15000 \
  --train-per-iter 10000 \
  --val-size 600 \
  --n-eval 500 \
  --eval-gpus 0 \
  --n-gpus 1 \
  --micro-batch 2 \
  --grad-accum 8 \
  --lr 2e-6 \
  --epochs 1.0 \
  --lora-r 32 \
  --lora-alpha 64 \
  --max-iters 5 \
  --use-local-score-cache \
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

hippius-hub upload yoko/teutonic-5ev4mnnc-013 /root/teutonic/s1-work/merged1 --revision main

python scripts/mining/submit_challenger.py \
  --uploaded-repo yoko/teutonic-5ev4mnnc-013 \
  --uploaded-digest sha256:d46b852f45124435589d293d6097b9d9167e8bc47c288d118bc80fd04574941d \
  --wallet-name silvanus-hs1 \
  --hotkey hotkey41 \
  --netuid 3 \
  --network finney

python scripts/mining/submit_challenger.py \
  --uploaded-repo "silvanus0930/teutonic-q3-4b-5ev4mnnc-006" \
  --uploaded-digest "sha256:YOUR_HIPPIUS_DIGEST" \
  --wallet YOUR_WALLET \
  --hotkey YOUR_HOTKEY \
  --netuid 3 \
  --network finney

python scripts/mining/submit_external_model.py \
  --hf silvanus0930/teutonic-5ev4mnnc-006 \
  --hippius-namespace silvanus0930 \
  --wallet silvanus-hs1 \
  --hotkey hotkey38 \
  --workdir /root/teutonic/s1-wor1/iter_00/merged \
  --skip-download \
  --dry-run

python scripts/mining/submit_external_model.py \
  --hf silvanus0930/teutonic-5ev4mnnc-006 \
  --hippius-namespace silvanus0930 \
  --wallet silvanus-hs1 \
  --hotkey hotkey37 \
  --workdir /root/teutonic/s1-wor1/iter_00/merged \
  --skip-download \
  --suffix 006 \
  --dry-run


  python -u scripts/mining/validator_eval.py \
  --king /root/teutonic/s1-work/king1 \
  --challenger /root/teutonic/s1-work/iter_00/merged \
  --hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --local-dataset /root/teutonic/s1-work/dataset/manifest.json \
  --n-public 500 \
  --batch-size 4 \
  --gpus 0 \
  --report-out /root/teutonic/s1-work/verdict.json


export LOCAL_KING_DIR="/root/teutonic/s1-work/king2"
export LOCAL_DATASET_MANIFEST="/root/teutonic/s1-work/dataset/manifest.json"
export TEUTONIC_SIM_HOTKEY="5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv"
export TRANSFORMERS_TRUST_REMOTE_CODE=true

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work \
  --bundle /root/teutonic/scripts/training_bundle \
  --dataset-mode local \
  --local-dataset-manifest /root/teutonic/s1-work/dataset/manifest.json \
  --candidate-preset strong_followup \
  --n-score 64000 \
  --hard-frac 0.35 \
  --max-iters 1 \
  --n-gpus 1 --micro-batch 1 --grad-accum 16 \
  --report-out /root/teutonic/s1-work/verdict.json

  cd /root/teutonic
source .venv/bin/activate

export LOCAL_KING_DIR="/root/teutonic/s1-work/iter_00/merged"
export TEUTONIC_SIM_HOTKEY="5GNJfDvtqqAjhwy9knrf9xTAhUu6hdZqwXELuY425Xio8a24"
export TRANSFORMERS_TRUST_REMOTE_CODE=true
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO="${LOCAL_KING_DIR}"
unset LOCAL_DATASET_MANIFEST

python -u scripts/mining/validator_eval.py \
  --king "${LOCAL_KING_DIR}" \
  --challenger /root/.cache/hippius/hub/models--mastertensor--teutonic-q3-10b-5ek5koe5-57645162954-rn/snapshots/5Ek5KoE5-57645162954-rn \
  --hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --n-public 500 \
  --batch-size 8 \
  --gpus 0 \
  --report-out /root/teutonic/s1-work/verdict.json


  cd /root/teutonic && source .venv/bin/activate

export LOCAL_KING_DIR="/root/.cache/hippius/hub/models--mastertensor--teutonic-q3-10b-5ek5koe5-40037154544-rn/snapshots/5Ek5KoE5-40037154544-rn"
export TEUTONIC_SIM_HOTKEY="5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv"
export TRANSFORMERS_TRUST_REMOTE_CODE=true

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work \
  --bundle /root/teutonic/scripts/training_bundle \
  --dataset-mix /root/teutonic/scripts/mining/dataset_mix_quasar_v4.json \
  --mix-shards-per-dataset 12 \
  --candidate-preset strong_followup \
  --n-score 5000 \
  --train-per-iter 4000 \
  --val-size 400 \
  --hard-frac 0.35 \
  --general-frac 0.55 \
  --easy-frac 0.10 \
  --epochs 1.5 \
  --lr 5e-5 \
  --max-iters 1 \
  --n-gpus 1 \
  --micro-batch 1 \
  --grad-accum 16 \
  --eval-mode validator \
  --final-eval-n 1000 \
  --n-eval 1000 \
  --report-out /root/teutonic/s1-work/verdict.json


  python scripts/mining/reshard_merged.py \
  --in /root/.cache/hippius/hub/models--mastertensor--teutonic-q3-10b-5ek5koe5-57645162954-rn/snapshots/5Ek5KoE5-57645162954-rn \
  --out /root/teutonic/s1-work/merged1
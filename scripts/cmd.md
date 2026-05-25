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

hippius-hub upload yoko/teutonic-q3-8b-5ev4mnnc-v1 /root/teutonic/s1-work-v2/iter_00/merged --revision v1

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

tech-dev-ai/Teutonic-VIII-5CXiauzN-cru1

cp -avL /root/.cache/hippius/hub/models--aetheling--teutonic-q3-4b-5cdd5hdj-v5/snapshots/main /root/teutonic/s1-work

python submit_challenger.py \
  --verdict /root/teutonic/s1-work/verdict.json \
  --hotkey default \
  --wallet-name silvanus-hs1 \
  --netuid 3 \
  --network finney


  hf upload silvanus0930/Teutonic-Q3-4B-Prod10T /root/teutonic/qwen3-fineweb-edu-prod10 . --repo-type dataset --commit-message "Upload"

  hf download taoism99/Teutonic-VIII-5FnnkHKa-t01 /root/teutonic/s1-work/merged123

export LOCAL_DATASET_MANIFEST="/path/to/your/5shard/manifest.json"

export LOCAL_KING_DIR="/root/teutonic/s1-work/main"
export TEUTONIC_SIM_HOTKEY="5FhMoUmcE9ed4p1it7xebF1y1SHdC5hYFbD1Gk44wuiX88hv"
export TEUTONIC_EVAL_DATASET_MODE=raw_hippius
export TEUTONIC_RAW_TOKENIZER_REPO="Qwen/Qwen3-4B"

python -u scripts/mining/train_challenger.py \
  --work /root/teutonic/s1-work-v2 \
  --bundle /root/teutonic/scripts/training_bundle \
  --dataset-mode auto \
  --n-shards 1 --shard-start 0 --eval-shard 1 \
  --eval-mode validator \
  --sim-hotkey "${TEUTONIC_SIM_HOTKEY}" \
  --n-score 3000 \
  --train-per-iter 2000 \
  --val-size 100 \
  --n-eval 100 \
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

python scripts/mining/submit_challenger.py \
  --uploaded_repo yoko/teutonic-q3-4b-5ev4mnnc-v17 \
  --uploaded_hash sha256:eba5ce9df9cf12243099cca68606fe3822e0ca03ee839758650e9b13b5549f12 \
  --wallet-name silvanus-hs1 \
  --hotkey default \
  --netuid 3 \
  --network finney 

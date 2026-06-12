REPO="mastertensor/teutonic-q3-10b-5ek5koe5-40037154544-rn"
REV="5Ek5KoE5-40037154544-rn"

for f in \
  model-00001-of-00005.safetensors \
  model-00002-of-00005.safetensors \
  model-00003-of-00005.safetensors \
  model-00004-of-00005.safetensors \
  model-00005-of-00005.safetensors \
  modeling_qwen3_5.py \
  configuration_qwen3_5.py \
  model.safetensors.index.json \
  config.json \
  tokenizer.json \
  tokenizer_config.json \
  generation_config.json
do
  echo "Downloading $f..."
  hippius-hub download "$REPO" "$f" --revision "$REV"
done

REPO="mastertensor/teutonic-q3-10b-5ek5koe5-57645162954-rn"
REV="5Ek5KoE5-57645162954-rn"

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
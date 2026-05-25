REPO="teutonic/teutonic-q3-8b-genesis"
REV="main"

for f in \
  model-00001-of-00005.safetensors \
  model-00002-of-00005.safetensors \
  model-00003-of-00005.safetensors \
  model-00004-of-00005.safetensors \
  model-00005-of-00005.safetensors \
  tokenizer.json \
  model.safetensors.index.json \
  tokenizer_config.json \
  config.json \
  vocab.json \
  merges.txt \
  generation_config.json
do
  echo "Downloading $f..."
  hippius-hub download "$REPO" "$f" --revision "$REV"
done

cp -avL /root/.cache/hippius/hub/models--teutonic--teutonic-q3-8b-genesis/snapshots/main /root/teutonic/s1-work
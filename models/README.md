# ImageNet baseline model

`mobilenetv2-12-qdq.onnx` is the ONNX Model Zoo MobileNetV2 QDQ classifier, trained for ImageNet-1k. Its model card specifies RGB NCHW input, ImageNet mean/std normalization, and 1,000 output classes.

`imagenet_labels.txt` contains the matching ImageNet class names, one class per line.

The model is distributed under Apache-2.0. Model source: <https://huggingface.co/onnxmodelzoo/mobilenetv2-12-qdq>. Label source: <https://github.com/pytorch/hub/blob/master/imagenet_classes.txt>.

Run a scan with:

```bash
cargo run --release -- \
  --tier1-model models/mobilenetv2-12-qdq.onnx \
  --labels-file models/imagenet_labels.txt \
  scan --input-dir All-Images --output scan_results.ndjson
```

This is a general ImageNet classifier. It does not guarantee perfect results, and it may not recognize specialized classes absent from ImageNet.

## Visual search encoder

`dinov2_vits14.onnx` is an ONNX export of the DINOv2 ViT-S/14 feature encoder. It returns a 384-value visual descriptor for a 224 x 224 RGB image. The project uses these descriptors for nearest-neighbor lookup and uses each indexed filename stem as the photo ID. The encoder is based on Meta's DINOv2 model and is distributed by its ONNX publisher under Apache-2.0.

Build the photo index:

```bash
cargo run --release -- \
  --tier1-model models/dinov2_vits14.onnx \
  --embedding-dim 384 \
  index --input-dir All-Images --index-file vectors.rkyv
```

Start the visual search API with the same model and dimension:

```bash
cargo run --release -- \
  --tier1-model models/dinov2_vits14.onnx \
  --embedding-dim 384 \
  serve --mode embedding-search --index-file vectors.rkyv --image-dir All-Images
```

Open <http://127.0.0.1:8080/test> to upload a screenshot or crop and inspect matching IDs, cosine scores, and photo previews. The page sends the image to `/v1/search?top_k=5`. Rebuild the index whenever the encoder or its preprocessing changes.

Model source and model card: <https://huggingface.co/sefaburak/dinov2-small-onnx>. DINOv2 research and feature details: <https://github.com/facebookresearch/dinov2>.

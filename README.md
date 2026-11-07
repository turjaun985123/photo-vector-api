# Photo Vector API

An image-only API that converts uploaded product photos into vector embeddings. It accepts one or more images and returns the photo embedding data with a UUID for correlating the response to a product in your main system.

The API does not store uploaded photos, product details, or uploaded vectors. The caller owns product records and decides where returned vector data is stored.

## API

- Form: `/products`
- Upload: `POST /v1/products` (`multipart/form-data`, repeat the `images` field)
- Health: `/healthz`

Each image produces four 384-dimensional vectors: full image, center crop, top crop, and bottom crop. The response includes one `request_id` UUID for the whole upload. Every vector ID contains that UUID and its image number.

The form and server use the hardcoded development bearer token `npc-image-index-dev-token`. Uploads may contain any number of images, limited to 10 MB per image and 30 MB total.

See [PRODUCT_API.md](PRODUCT_API.md) for the response shape and calling example.

## Render deployment

1. Create a Render **Web Service** from this repository.
2. Set the Start Command to `bash start.sh`.
3. Deploy. `start.sh` downloads the latest Linux x86_64 release bundle, which contains the API binary and DINOv2 model, then starts the server on Render's `PORT` (default 10000).

The release workflow builds a portable x86_64 Linux binary, downloads and SHA-256 verifies the DINOv2 ONNX model, bundles both, and publishes a GitHub release.

To create a release, push a `v*` tag or run the **Build and publish Render release** workflow with `gh workflow run release.yml -f tag=v0.1.0`.

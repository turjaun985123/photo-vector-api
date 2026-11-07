# Photo vector indexing API

Open `/products` for the photo-only form. Upload any number of JPG, PNG, or WebP image files, with at least one image. No product IDs, names, or other text fields are accepted. The main server that calls this API is responsible for collecting product details and associating them with the response `request_id`.

The form includes the hardcoded bearer token `npc-image-index-dev-token`. Programmatic clients send it as `Authorization: Bearer npc-image-index-dev-token`.

## Endpoint

`POST /v1/products` accepts `multipart/form-data` with one or more repeated `images` fields. Each image must be under 10 MB, with up to 30 MB total image data.

The server generates four 384-dimensional embeddings per photo: the full image, center crop, top crop, and bottom crop. Each request receives a unique UUID in `request_id`; all returned vector IDs include the same UUID and the uploaded image number. The main server can use `request_id` to associate the batch with its product record. Uploaded files are processed in memory and are not saved. The request does not update this server's catalog index or store product data.

Example response shape:

```json
{
  "vectors": [
    {
      "vector_id": "<request-id>-image-0001-full",
      "vector_indexing_data": [0.01, -0.02]
    }
  ]
}
```

Each vector contains 384 floating-point values. One uploaded photo returns 4 vectors; N uploaded photos return 4N vectors. A response includes the shared `request_id` and `vectors` array.

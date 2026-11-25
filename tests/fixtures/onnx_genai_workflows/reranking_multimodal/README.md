# Native multimodal relevance fixture

This model combines the existing tokenizer pair distance with decoded image
pixel distances. The query supplies one PNG; a document supplies two ordered
PNGs. The image distance is `mean(abs(query-primary)) +
2*mean(abs(query-secondary))`. The final score is sigmoid of the negative text
and image distances. The graph also emits its actual non-padding token count.

The package reuses one image-preprocess@2 component three times. PNG byte lengths
are deliberately different (the black PNG contains an ancillary text chunk),
while decoded geometry is identical. Runtime code must obtain dimensions and
binding names from the package. `truncated_pixels.png` has a valid header and
container but lacks complete pixel data; it is used only for the HTTP decode
validation regression.

Tests cover native scores, ordering, optional workflow-owned image fallback,
Jina/TEI HTTP shapes, raw base64/data URI/HTTP images and request resource bounds.
These tiny fixture weights do not establish real-model quality or performance.

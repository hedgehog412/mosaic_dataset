# mosaic_dataset
Converting mosaic GUI logs to PyTorch datasets

## Camera-like Cropping

The dataset now includes camera-like cropping to reduce token usage for transformer training:

- **Automatic bounding box calculation**: Crops around visible tiles instead of rendering the full canvas
- **Padding**: Adds configurable padding around tiles (default 30%) to simulate camera field of view
- **Crop jitter**: Random offsets during augmentation (default 10%) to simulate camera movement
- **Center crop and pad**: Crops are placed in the center of a fixed-size canvas with padding (no scaling to preserve tile size for VQVAE)

### Parameters

- `crop_padding_frac` (default: 0.3): Padding around tiles as fraction of scene size
- `crop_jitter_frac` (default: 0.1): Random crop offset as fraction of padding for augmentation
- `output_size` (default: (256, 256)): Final output size after cropping and center padding (tiles are NOT scaled)
- `min_tiles` (default: 1): Minimum number of tiles required per sample - filters out empty canvas states automatically

### Preventing Empty Canvases

The dataset automatically filters out any events with fewer than `min_tiles` tiles during initialization. This ensures:
- No empty white canvases in your dataset
- All samples contain meaningful content
- No wasted training iterations on empty images

Set `min_tiles=0` to disable filtering (not recommended).

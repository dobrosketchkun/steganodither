# Halftone 2D Code

Embed messages in images using dual-mode halftone dithering. This script can hide text in both grayscale and color images using a clever dual-mode dithering approach that embeds data while maintaining visual quality.

## Features

- **Dual-Mode Dithering**: Uses two visually similar but detectable patterns to encode binary data
- **Reed-Solomon Error Correction**: Both header and payload use RS error correction for robustness
- **Encryption**: Optional password-based encryption using Fernet + PBKDF2HMAC
- **Adaptive Contrast**: Automatically adjusts image blocks to improve encoding quality
- **Automatic Resizing**: Resizes input images to fit block dimensions
- **Header Redundancy**: Repeats the header with delimiters for reliable decoding
- **Payload Padding**: Fills available space through repetition for visual consistency across the entire image
- **Color Support**: Works with both grayscale and color images (encoding in luminance channel)

## Image Example 

[<img src="misc/output.png" alt="An example of output image" />](## "An example of output image")  


## Requirements

```bash
pip install opencv-python numpy reedsolo cryptography
```

## Usage

### Encoding Messages

#### Grayscale Encoding

```bash
python steganodither.py encode \
  --input input.jpg \
  --output encoded.png \
  --message "Your secret message here" \
  --block_size 16 \
  --header_rows 15 \
  --rs_header_nsym 8 \
  --rs_payload_nsym 16 \
  --scale_factor 2.0
```

#### Color Encoding

```bash
python steganodither.py encode \
  --input input.jpg \
  --output encoded.png \
  --message "Your secret message here" \
  --block_size 16 \
  --header_rows 15 \
  --color \
  --rs_header_nsym 8 \
  --rs_payload_nsym 16 \
  --scale_factor 2.0
```

#### With Encryption

Add `--password your_secret_password` to either command to encrypt the message.

### Decoding Messages

#### Grayscale Decoding

```bash
python steganodither.py decode \
  --input encoded.png \
  --block_size 16 \
  --header_rows 15 \
  --rs_header_nsym 8 \
  --rs_payload_nsym 16
```

#### Color Decoding

```bash
python steganodither.py decode \
  --input encoded.png \
  --block_size 16 \
  --header_rows 15 \
  --color \
  --rs_header_nsym 8 \
  --rs_payload_nsym 16
```

#### With Decryption

Add `--password your_secret_password` to decrypt (must be the same as used for encoding).

## How It Works

1. **Header Encoding**: The header (containing payload length) is RS-encoded and repeated with delimiters in the header area.
2. **Payload Processing**: 
   - If a password is provided, the message is encrypted and then RS-encoded
   - Otherwise, the message is directly RS-encoded
3. **Dithering**: The script uses two different Bayer matrix patterns to create visually similar but detectable halftones
4. **Adaptive Contrast**: Automatically adjusts block contrast to improve pattern detection
5. **Decoding Process**: 
   - Extracts and decodes the header to determine payload length
   - Extracts exactly that many payload bits
   - RS-decodes and (if needed) decrypts to recover the original message

## Parameters

- `--block_size`: Size of each dithering block in pixels (default: 16)
- `--header_rows`: Number of rows reserved for header data (default: 15)
- `--rs_header_nsym`: Reed-Solomon error correction symbols for header (default: 8)
- `--rs_payload_nsym`: Reed-Solomon error correction symbols for payload (default: 16)
- `--scale_factor`: Image scaling factor before encoding (default: 1.0)
- `--pixel_algo`: Pixelization algorithm: "nearest", "bilinear", or "bicubic" (default: "nearest")
- `--color`: Process as color image instead of grayscale
- `--password`: Optional encryption password


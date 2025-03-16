#!/usr/bin/env python3
import argparse
import sys
import cv2
import numpy as np
import logging
import os
import base64

try:
    import reedsolo
    RS_AVAILABLE = True
except ImportError:
    RS_AVAILABLE = False

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ----- Cryptographic Functions -----
def derive_key(password: str, salt: bytes, iterations: int = 100000) -> bytes:
    password_bytes = password.encode()
    kdf = PBKDF2HMAC(
         algorithm=hashes.SHA256(),
         length=32,
         salt=salt,
         iterations=iterations,
         backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
    return key

def encrypt_message(message: str, password: str) -> bytes:
    salt = os.urandom(16)
    key = derive_key(password, salt)
    fernet = Fernet(key)
    token = fernet.encrypt(message.encode())
    return salt + token

def decrypt_message(token: bytes, password: str) -> str:
    salt = bytes(token[:16])
    actual_token = bytes(token[16:])
    key = derive_key(password, salt)
    fernet = Fernet(key)
    decrypted = fernet.decrypt(actual_token)
    return decrypted.decode()

# ----- Utility Functions for Bit Conversion -----
def text_to_bits(text):
    bits = []
    for char in text:
        char_bits = format(ord(char), '08b')
        logging.debug(f"Character '{char}' -> bits {char_bits}")
        bits.extend([int(x) for x in char_bits])
    return bits

def bits_to_text(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char = chr(int("".join(str(b) for b in byte), 2))
        chars.append(char)
    return "".join(chars)

def int_to_bits(n, num_bits):
    bits = [int(x) for x in format(n, f'0{num_bits}b')]
    logging.debug(f"Integer {n} to {num_bits} bits: {bits}")
    return bits

def bits_to_int(bits):
    n = int("".join(str(b) for b in bits), 2)
    logging.debug(f"Bits {bits} to integer: {n}")
    return n

# ----- RS Encoding/Decoding for Header and Payload -----
def rs_encode_data(data_bytes, nsym):
    if not RS_AVAILABLE:
        logging.warning("Reed–Solomon library not available; using raw data")
        return data_bytes
    rsc = reedsolo.RSCodec(nsym)
    encoded = rsc.encode(data_bytes)
    logging.debug(f"RS-encoded data (nsym={nsym}): {list(encoded)}")
    return encoded

def rs_decode_data(encoded_bytes, nsym):
    if not RS_AVAILABLE:
        logging.warning("Reed–Solomon library not available; using raw data")
        return encoded_bytes
    rsc = reedsolo.RSCodec(nsym)
    try:
        decoded = rsc.decode(encoded_bytes)
    except reedsolo.ReedSolomonError as e:
        logging.error("RS decoding failed: " + str(e))
        sys.exit(1)
    if isinstance(decoded, tuple):
        decoded = decoded[0]
    logging.debug(f"RS-decoded data: {list(decoded)}")
    return decoded

# ----- Delimiter for Header Copies (plain fixed sequence) -----
DELIMITER_BITS = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]  # 10-bit delimiter

# ----- Adaptive Contrast Adjustment -----
def adaptive_adjust(block, desired_mode):
    avg = np.mean(block)
    if desired_mode == 'A' and avg < 35:
        logging.debug(f"Adaptive adjust Mode A: avg {avg:.2f} < 35, adding 20")
        block = np.clip(block + 20, 0, 255)
    elif desired_mode == 'B' and avg > 220:
        logging.debug(f"Adaptive adjust Mode B: avg {avg:.2f} > 220, subtracting 20")
        block = np.clip(block - 20, 0, 255)
    return block

# ----- Dual-Mode Dithering Functions with Adaptive Adjustment -----
def dither_mode_A(block):
    block = adaptive_adjust(block, 'A')
    bayer = np.array([[0,  8,  2, 10],
                      [12, 4, 14, 6],
                      [3, 11, 1, 9],
                      [15, 7, 13, 5]], dtype=np.float32) / 16.0
    threshold = bayer * 255
    h, w = block.shape
    tile_rows = int(np.ceil(h/4))
    tile_cols = int(np.ceil(w/4))
    tiled = np.tile(threshold, (tile_rows, tile_cols))[:h, :w]
    output = (block > tiled).astype(np.uint8) * 255
    return output

def dither_mode_B(block):
    block = adaptive_adjust(block, 'B')
    bayer = np.array([[0,  8,  2, 10],
                      [12, 4, 14, 6],
                      [3, 11, 1, 9],
                      [15, 7, 13, 5]], dtype=np.float32) / 16.0
    rotated = np.rot90(bayer)
    threshold = rotated * 255
    h, w = block.shape
    tile_rows = int(np.ceil(h/4))
    tile_cols = int(np.ceil(w/4))
    tiled = np.tile(threshold, (tile_rows, tile_cols))[:h, :w]
    output = (block > tiled).astype(np.uint8) * 255
    return output

# ----- Header/ Payload Block Encoding -----
def encode_data_block(block, bit, mode_func):
    return mode_func(block)

def encode_header_block(block, bit, mode_func):
    return mode_func(block)

def encode_delimiter_block(block, bit):
    mode_func = dither_mode_A if bit == 0 else dither_mode_B
    return mode_func(block)

# ----- Pixelization -----
def pixelize_image(image, scale_factor=1.0, algorithm='nearest'):
    if scale_factor != 1.0:
        h, w = image.shape[:2]
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))
        if algorithm == 'nearest':
            interp = cv2.INTER_NEAREST
        elif algorithm == 'bilinear':
            interp = cv2.INTER_LINEAR
        elif algorithm == 'bicubic':
            interp = cv2.INTER_CUBIC
        else:
            raise ValueError("Unsupported pixelation algorithm: " + algorithm)
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        logging.debug(f"Image pixelized to {new_w}x{new_h} using {algorithm}")
    return image

# ----- Automatic Resizing to Multiples of Block Size -----
def adjust_image_size(image, block_size):
    h, w = image.shape[:2]
    new_w = round(w / block_size) * block_size
    new_h = round(h / block_size) * block_size
    if new_w != w or new_h != h:
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        logging.info(f"Resized image from {w}x{h} to {new_w}x{new_h} to be divisible by block size {block_size}")
    return image

# ----- Header Area Encoding with Repeated RS-Coded Header and Delimiters -----
def encode_header_area_rs(output, rs_header_bits, block_size, num_blocks_x, header_rows):
    total_header_blocks = header_rows * num_blocks_x
    copy_len = len(rs_header_bits)
    delim_len = len(DELIMITER_BITS)
    segment_len = copy_len + delim_len
    num_copies = total_header_blocks // segment_len
    logging.info(f"Header area: {total_header_blocks} blocks; storing {num_copies} full header copies (each {segment_len} blocks)")
    block_index = 0
    for c in range(num_copies):
        for bit in rs_header_bits:
            row = block_index // num_blocks_x
            col = block_index % num_blocks_x
            y0 = row * block_size
            x0 = col * block_size
            block = output[y0:y0+block_size, x0:x0+block_size]
            mode_func = dither_mode_A if bit == 0 else dither_mode_B
            output[y0:y0+block_size, x0:x0+block_size] = mode_func(block)
            logging.debug(f"Header copy {c}, block ({row},{col}) encoded bit {bit}")
            block_index += 1
        for bit in DELIMITER_BITS:
            row = block_index // num_blocks_x
            col = block_index % num_blocks_x
            y0 = row * block_size
            x0 = col * block_size
            block = output[y0:y0+block_size, x0:x0+block_size]
            output[y0:y0+block_size, x0:x0+block_size] = encode_delimiter_block(block, bit)
            logging.debug(f"Delimiter at block ({row},{col}) encoded bit {bit}")
            block_index += 1
    while block_index < total_header_blocks:
        row = block_index // num_blocks_x
        col = block_index % num_blocks_x
        y0 = row * block_size
        x0 = col * block_size
        output[y0:y0+block_size, x0:x0+block_size] = dither_mode_A(output[y0:y0+block_size, x0:x0+block_size])
        logging.debug(f"Filling extra header block ({row},{col}) with neutral Mode A")
        block_index += 1
    return output

def extract_header_area_bits(output, block_size, header_rows, num_blocks_x):
    bits = []
    total_header_blocks = header_rows * num_blocks_x
    for i in range(total_header_blocks):
        row = i // num_blocks_x
        col = i % num_blocks_x
        y0 = row * block_size
        x0 = col * block_size
        block = output[y0:y0+block_size, x0:x0+block_size]
        avg = np.mean(block)
        constant = np.full((block_size, block_size), avg, dtype=np.uint8)
        expected_A = dither_mode_A(constant)
        expected_B = dither_mode_B(constant)
        actual_bin = (block > 128).astype(np.uint8)
        exp_A_bin = (expected_A > 128).astype(np.uint8)
        exp_B_bin = (expected_B > 128).astype(np.uint8)
        diff_A = np.sum(actual_bin != exp_A_bin)
        diff_B = np.sum(actual_bin != exp_B_bin)
        bit = 0 if diff_A <= diff_B else 1
        bits.append(bit)
    return bits

def decode_header_area_rs(output, block_size, header_rows, num_blocks_x, rs_header_nsym):
    header_bits = extract_header_area_bits(output, block_size, header_rows, num_blocks_x)
    logging.debug(f"Extracted header bits from area: {header_bits}")
    candidates = []
    current = []
    delim_len = len(DELIMITER_BITS)
    i = 0
    while i < len(header_bits):
        if header_bits[i:i+delim_len] == DELIMITER_BITS:
            if current:
                candidates.append(current)
                current = []
            i += delim_len
        else:
            current.append(header_bits[i])
            i += 1
    if current:
        candidates.append(current)
    logging.debug(f"Found {len(candidates)} candidate header copies: {candidates}")
    expected_length = None
    valid_candidates = []
    for cand in candidates:
        if expected_length is None:
            expected_length = len(cand)
        if len(cand) == expected_length:
            valid_candidates.append(cand)
    if not valid_candidates:
        logging.error("No valid header candidate found")
        sys.exit(1)
    for idx, bits in enumerate(valid_candidates):
        header_bytes = bytearray()
        for j in range(0, len(bits), 8):
            byte = int("".join(str(b) for b in bits[j:j+8]), 2)
            header_bytes.append(byte)
        logging.debug(f"Candidate {idx} header bytes: {list(header_bytes)}")
        decoded = rs_decode_data(header_bytes, nsym=rs_header_nsym)
        if len(decoded) >= 2:
            logging.info(f"Candidate {idx} successfully RS-decoded header")
            return decoded[:2]
    logging.error("No candidate header could be RS-decoded")
    sys.exit(1)

def decode_header_process_gray(image, block_size=16, header_rows=15, rs_header_nsym=8):
    h, w = image.shape
    num_blocks_x = w // block_size
    decoded_header = decode_header_area_rs(image, block_size, header_rows, num_blocks_x, rs_header_nsym)
    payload_length = int.from_bytes(decoded_header, byteorder='big')
    logging.info(f"Decoded header indicates payload length: {payload_length} bits")
    return payload_length

# ----- Grayscale Full Encoding/Decoding Functions with Padded Payload and Repeated RS Header -----
def encode_image_process_gray(image, message, block_size=16, header_rows=15, rs_header_nsym=8, rs_payload_nsym=16, password=None):
    logging.info("Starting dual-mode encoding with repeated RS-protected header and payload RS encoding (grayscale)")
    image = adjust_image_size(image, block_size)
    h, w = image.shape
    num_blocks_x = w // block_size
    num_blocks_y = h // block_size
    total_blocks = num_blocks_x * num_blocks_y
    logging.debug(f"Image: {w}x{h} pixels; Grid: {num_blocks_x}x{num_blocks_y}; Total blocks: {total_blocks}")

    if password:
        encrypted = encrypt_message(message, password)
        logging.info("Message encrypted with password")
        payload_bytes = rs_encode_data(encrypted, nsym=rs_payload_nsym)
    else:
        payload_bytes = rs_encode_data(message.encode(), nsym=rs_payload_nsym)
    payload_bits = []
    for byte in payload_bytes:
        payload_bits.extend([int(x) for x in format(byte, '08b')])
    logging.info(f"Payload length: {len(payload_bits)} bits")
    
    header_int = len(payload_bits)
    header_bytes = header_int.to_bytes(2, byteorder='big')
    logging.debug(f"Original header (2 bytes): {list(header_bytes)}")
    rs_header_bytes = rs_encode_data(header_bytes, nsym=rs_header_nsym)
    rs_header_bits = []
    for byte in rs_header_bytes:
        rs_header_bits.extend([int(x) for x in format(byte, '08b')])
    logging.debug(f"RS-encoded header bits ({len(rs_header_bits)} bits): {rs_header_bits}")

    header_capacity = header_rows * num_blocks_x
    delimiter_length = len(DELIMITER_BITS)
    segment_length = len(rs_header_bits) + delimiter_length
    num_copies = header_capacity // segment_length
    if num_copies < 1:
        logging.error(f"Not enough header area to store at least one full RS header copy (needs {segment_length} blocks)")
        sys.exit(1)
    logging.info(f"Header area will store {num_copies} copies of the RS header (each {segment_length} blocks)")
    payload_capacity = total_blocks - header_capacity
    if len(payload_bits) > payload_capacity:
        logging.error(f"Not enough payload blocks: capacity {payload_capacity} vs. payload bits {len(payload_bits)}")
        sys.exit(1)
    logging.debug(f"Header capacity: {header_capacity} blocks; Payload capacity: {payload_capacity} blocks")

    padded_payload = (payload_bits * ((payload_capacity // len(payload_bits)) + 1))[:payload_capacity]
    logging.debug(f"Padded payload length: {len(padded_payload)} bits (original {len(payload_bits)} bits)")

    output = image.copy()
    output = encode_header_area_rs(output, rs_header_bits, block_size, num_blocks_x, header_rows)
    logging.info("Header encoding complete")

    payload_index = 0
    for by in range(header_rows, num_blocks_y):
        for bx in range(num_blocks_x):
            y0 = by * block_size
            x0 = bx * block_size
            block = output[y0:y0+block_size, x0:x0+block_size]
            if payload_index < len(padded_payload):
                bit = padded_payload[payload_index]
                mode_func = dither_mode_A if bit == 0 else dither_mode_B
                encoded_block = mode_func(block)
                output[y0:y0+block_size, x0:x0+block_size] = encoded_block
                logging.debug(f"Payload block ({by},{bx}) encoded bit {bit}")
                payload_index += 1
            else:
                logging.debug(f"Payload block ({by},{bx}) no payload data; using Mode A neutral")
                output[y0:y0+block_size, x0:x0+block_size] = dither_mode_A(block)
    logging.info("Payload encoding complete")
    return output

def decode_image_process_gray(image, block_size=16, header_rows=15, rs_header_nsym=8, rs_payload_nsym=16, password=None):
    logging.info("Starting dual-mode decoding with repeated RS-protected header (grayscale)")
    image = adjust_image_size(image, block_size)
    h, w = image.shape
    num_blocks_x = w // block_size
    num_blocks_y = h // block_size
    total_blocks = num_blocks_x * num_blocks_y
    logging.debug(f"Image: {w}x{h} pixels; Grid: {num_blocks_x}x{num_blocks_y}; Total blocks: {total_blocks}")

    payload_length = decode_header_process_gray(image, block_size, header_rows, rs_header_nsym)
    
    payload_bits = []
    for by in range(header_rows, num_blocks_y):
        for bx in range(num_blocks_x):
            if len(payload_bits) >= payload_length:
                break
            y0 = by * block_size
            x0 = bx * block_size
            block = image[y0:y0+block_size, x0:x0+block_size]
            avg = np.mean(block)
            constant = np.full((block_size, block_size), avg, dtype=np.uint8)
            expected_A = dither_mode_A(constant)
            expected_B = dither_mode_B(constant)
            actual_bin = (block > 128).astype(np.uint8)
            exp_A_bin = (expected_A > 128).astype(np.uint8)
            exp_B_bin = (expected_B > 128).astype(np.uint8)
            diff_A = np.sum(actual_bin != exp_A_bin)
            diff_B = np.sum(actual_bin != exp_B_bin)
            bit = 0 if diff_A <= diff_B else 1
            payload_bits.append(bit)
            logging.debug(f"Payload block ({by},{bx}): avg={avg:.2f}, diff_A={diff_A}, diff_B={diff_B}, decoded bit={bit}")
        if len(payload_bits) >= payload_length:
            break
    if len(payload_bits) < payload_length:
        logging.error("Insufficient payload data extracted")
        sys.exit(1)
    payload_bits = payload_bits[:payload_length]
    payload_bytes = bytearray()
    for i in range(0, len(payload_bits), 8):
        byte = int("".join(str(b) for b in payload_bits[i:i+8]), 2)
        payload_bytes.append(byte)
    if password:
        decoded_payload = rs_decode_data(bytes(payload_bytes), nsym=rs_payload_nsym)
        decrypted = decrypt_message(decoded_payload, password)
        logging.info("Payload decrypted successfully")
        return decrypted
    else:
        return payload_bytes.decode()

# ----- Color Processing Functions (Luminance Only) -----
def encode_image_process_color(image, message, block_size=16, header_rows=15, rs_header_nsym=8, rs_payload_nsym=16, password=None):
    logging.info("Starting color encoding in luminance mode")
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    encoded_Y = encode_image_process_gray(Y, message, block_size, header_rows, rs_header_nsym, rs_payload_nsym, password)
    encoded_ycrcb = cv2.merge([encoded_Y, Cr, Cb])
    encoded_bgr = cv2.cvtColor(encoded_ycrcb, cv2.COLOR_YCrCb2BGR)
    return encoded_bgr

def decode_image_process_color(image, block_size=16, header_rows=15, rs_header_nsym=8, rs_payload_nsym=16, password=None):
    logging.info("Starting color decoding in luminance mode")
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, _, _ = cv2.split(ycrcb)
    message = decode_image_process_gray(Y, block_size, header_rows, rs_header_nsym, rs_payload_nsym, password)
    return message

# ----- Main CLI with Sub-commands -----
def main():
    parser = argparse.ArgumentParser(
        description="Dual-Mode Halftone 2D Code with RS-Protected Header and Payload, Adaptive Contrast, Resizing, Payload Padding, Repeated RS Header with Delimiters, and Optional Encryption"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands: encode, decode")

    # Encode sub-command.
    encode_parser = subparsers.add_parser("encode", help="Encode a message into an image (default: grayscale; use --color for color)")
    encode_parser.add_argument("--input", required=True, help="Input image file (default: grayscale)")
    encode_parser.add_argument("--output", required=True, help="Output image file for encoded data")
    encode_parser.add_argument("--message", required=True, help="Message to encode")
    encode_parser.add_argument("--block_size", type=int, default=16, help="Block size in pixels (default: 16)")
    encode_parser.add_argument("--header_rows", type=int, default=15, help="Number of block rows reserved for header (default: 15)")
    encode_parser.add_argument("--scale_factor", type=float, default=1.0, help="Pixelization scale factor (default: 1.0, no pixelation)")
    encode_parser.add_argument("--pixel_algo", type=str, default="nearest", choices=["nearest", "bilinear", "bicubic"], help="Pixelization algorithm (default: nearest)")
    encode_parser.add_argument("--color", action="store_true", help="Process image as color (default: grayscale)")
    encode_parser.add_argument("--password", type=str, help="Password for encrypting the message (optional)")
    encode_parser.add_argument("--rs_header_nsym", type=int, default=8, help="RS nsym for header (default: 8)")
    encode_parser.add_argument("--rs_payload_nsym", type=int, default=16, help="RS nsym for payload (default: 16)")

    # Decode sub-command.
    decode_parser = subparsers.add_parser("decode", help="Decode a message from an image (default: grayscale; use --color for color)")
    decode_parser.add_argument("--input", required=True, help="Input encoded image file (default: grayscale)")
    decode_parser.add_argument("--block_size", type=int, default=16, help="Block size in pixels (default: 16)")
    decode_parser.add_argument("--header_rows", type=int, default=15, help="Number of block rows reserved for header (default: 15)")
    decode_parser.add_argument("--color", action="store_true", help="Process image as color (default: grayscale)")
    decode_parser.add_argument("--password", type=str, help="Password for decrypting the message (if used during encoding)")
    decode_parser.add_argument("--rs_header_nsym", type=int, default=8, help="RS nsym for header (default: 8)")
    decode_parser.add_argument("--rs_payload_nsym", type=int, default=16, help="RS nsym for payload (default: 16)")

    args = parser.parse_args()
    if args.command == "encode":
        if args.color:
            image = cv2.imread(args.input)
            if image is None:
                logging.error("Could not load input image")
                sys.exit(1)
            logging.info("Input color image loaded successfully")
        else:
            image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logging.error("Could not load input image")
                sys.exit(1)
            logging.info("Input grayscale image loaded successfully")
        image = pixelize_image(image, args.scale_factor, args.pixel_algo)
        image = adjust_image_size(image, args.block_size)
        if args.color:
            encoded = encode_image_process_color(image, args.message, block_size=args.block_size, header_rows=args.header_rows, rs_header_nsym=args.rs_header_nsym, rs_payload_nsym=args.rs_payload_nsym, password=args.password)
        else:
            encoded = encode_image_process_gray(image, args.message, block_size=args.block_size, header_rows=args.header_rows, rs_header_nsym=args.rs_header_nsym, rs_payload_nsym=args.rs_payload_nsym, password=args.password)
        cv2.imwrite(args.output, encoded)
        logging.info(f"Encoded image saved to {args.output}")
    elif args.command == "decode":
        if args.color:
            image = cv2.imread(args.input)
            if image is None:
                logging.error("Could not load input image")
                sys.exit(1)
            logging.info("Input color image loaded successfully")
        else:
            image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
            if image is None:
                logging.error("Could not load input image")
                sys.exit(1)
            logging.info("Input grayscale image loaded successfully")
        image = adjust_image_size(image, args.block_size)
        if args.color:
            decoded = decode_image_process_color(image, block_size=args.block_size, header_rows=args.header_rows, rs_header_nsym=args.rs_header_nsym, rs_payload_nsym=args.rs_payload_nsym, password=args.password)
        else:
            decoded = decode_image_process_gray(image, block_size=args.block_size, header_rows=args.header_rows, rs_header_nsym=args.rs_header_nsym, rs_payload_nsym=args.rs_payload_nsym, password=args.password)
        logging.info(f"Decoded Message:\n{decoded}")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

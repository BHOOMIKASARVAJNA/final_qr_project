"""
inference.py — QR Tampering Detection using ELA (Error Level Analysis)
No TensorFlow. No heavy dependencies. Render-deployable.

Techniques used:
  1. ELA (Error Level Analysis) — detects pixel inconsistencies from compression artifacts
  2. Noise Variance Analysis     — tampered regions show abnormal noise patterns
  3. Edge Sharpness Analysis     — spliced regions have inconsistent edge sharpness
  4. UPI Merchant Verification   — checks extracted UPI ID against merchant registry
"""

import io
import os
import json
import numpy as np
import urllib.parse as urlparse
from PIL import Image, ImageChops, ImageEnhance
import cv2

# ─────────────────────────────────────────
# Load merchant registry once at startup
# ─────────────────────────────────────────
try:
    REG_PATH = os.path.join(os.path.dirname(__file__), "merchant_registry.json")
    with open(REG_PATH, "r", encoding="utf-8") as f:
        raw_registry = json.load(f)
    VERIFIED_UPI_IDS = {entry["verified_upi_id"].strip().lower() for entry in raw_registry}
    UPI_TO_NAME = {
        entry["verified_upi_id"].strip().lower(): entry["merchant_name"]
        for entry in raw_registry
    }
    print(f"Merchant registry loaded: {len(VERIFIED_UPI_IDS)} verified UPI IDs.")
except Exception as e:
    VERIFIED_UPI_IDS = set()
    UPI_TO_NAME = {}
    print(f"Could not load merchant registry: {e}")


# ─────────────────────────────────────────
# ELA — Error Level Analysis
# ─────────────────────────────────────────
def ela_score(image: Image.Image, quality: int = 90) -> float:
    """
    Saves image at reduced JPEG quality, compares with original.
    Tampered regions compress differently → higher ELA values.
    Returns a normalized score between 0.0 and 1.0.
    """
    # Save as JPEG at lower quality
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")

    # Pixel difference between original and re-compressed
    diff = ImageChops.difference(image.convert("RGB"), compressed)

    # Amplify for visibility
    enhanced = ImageEnhance.Brightness(diff).enhance(20)
    ela_array = np.array(enhanced).astype(np.float32)

    # Mean brightness of ELA image — higher = more inconsistency
    ela_mean = ela_array.mean()

    # Normalize to 0–1 range (empirically, tampered QRs score > 15 raw)
    score = min(ela_mean / 40.0, 1.0)
    return round(score, 4)


# ─────────────────────────────────────────
# Noise Variance Analysis
# ─────────────────────────────────────────
def noise_variance_score(image: Image.Image) -> float:
    """
    Splits image into blocks and checks noise variance consistency.
    Tampered images show localized spikes in noise.
    Returns a normalized score between 0.0 and 1.0.
    """
    gray = np.array(image.convert("L")).astype(np.float32)
    h, w = gray.shape
    block_size = max(16, h // 8)

    variances = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            variances.append(np.var(block))

    if len(variances) < 2:
        return 0.0

    variances = np.array(variances)
    # High coefficient of variation = inconsistent noise = likely tampered
    cv = np.std(variances) / (np.mean(variances) + 1e-6)

    # Normalize: CV above 2.0 is highly suspicious for a QR code
    score = min(cv / 2.0, 1.0)
    return round(float(score), 4)


# ─────────────────────────────────────────
# Edge Sharpness Consistency
# ─────────────────────────────────────────
def edge_inconsistency_score(image: Image.Image) -> float:
    """
    Measures edge sharpness variation across the image.
    Genuine QR codes have uniform edge sharpness.
    Tampered ones may have a sticker/overlay with different blur level.
    Returns a normalized score between 0.0 and 1.0.
    """
    gray = np.array(image.convert("L"))
    h, w = gray.shape
    block_size = max(32, h // 4)

    laplacian_vars = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            lap = cv2.Laplacian(block, cv2.CV_64F)
            laplacian_vars.append(lap.var())

    if len(laplacian_vars) < 2:
        return 0.0

    laplacian_vars = np.array(laplacian_vars)
    cv = np.std(laplacian_vars) / (np.mean(laplacian_vars) + 1e-6)

    score = min(cv / 3.0, 1.0)
    return round(float(score), 4)


# ─────────────────────────────────────────
# QR Decoder (OpenCV only)
# ─────────────────────────────────────────
def decode_qr(img_array: np.ndarray) -> str | None:
    """
    Decode QR code from image array using OpenCV.
    Returns decoded string or None.
    """
    try:
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(img_array)
        if data:
            return data.strip()
    except Exception:
        pass
    return None


# ─────────────────────────────────────────
# UPI Verification
# ─────────────────────────────────────────
def verify_upi(qr_data: str | None) -> tuple[bool, str]:
    """
    Returns (is_verified, merchant_name_or_message).
    """
    if not qr_data:
        return False, "No UPI ID found in QR"

    upi_id = qr_data
    if "pa=" in qr_data.lower():
        try:
            parsed = urlparse.urlparse(qr_data)
            params = urlparse.parse_qs(parsed.query)
            upi_id = params.get("pa", [qr_data])[0].strip()
        except Exception:
            pass

    lookup = upi_id.lower()
    if lookup in VERIFIED_UPI_IDS:
        return True, UPI_TO_NAME.get(lookup, "Unknown Merchant")
    return False, "UPI ID not in merchant registry"


# ─────────────────────────────────────────
# Main Analysis Function
# ─────────────────────────────────────────
async def analyze_qr(file) -> dict:
    """
    Accepts a FastAPI UploadFile.
    Returns structured analysis result.
    """
    result = {
        "status": "Unknown",
        "risk_score": 0,
        "extracted_upi": None,
        "merchant_name": None,
        "upi_verified": False,
        "tampering_probability": 0.0,
        "reasons": []
    }

    try:
        # ── 1. Read image ──
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((256, 256))  # normalize size
        img_array = np.array(image)

        # ── 2. Decode QR ──
        qr_data = decode_qr(img_array)
        result["extracted_upi"] = qr_data

        # ── 3. UPI Verification ──
        upi_verified, upi_msg = verify_upi(qr_data)
        result["upi_verified"] = upi_verified
        if upi_verified:
            result["merchant_name"] = upi_msg
        else:
            result["reasons"].append(f"UPI check: {upi_msg}")

        # ── 4. Tampering Analysis (3 signals) ──
        ela       = ela_score(image)
        noise     = noise_variance_score(image)
        edge      = edge_inconsistency_score(image)

        # Weighted combination (ELA is most reliable for QR tampering)
        tamper_prob = round((ela * 0.5) + (noise * 0.3) + (edge * 0.2), 4)
        result["tampering_probability"] = tamper_prob

        print(f"ELA: {ela:.3f} | Noise: {noise:.3f} | Edge: {edge:.3f} | Combined: {tamper_prob:.3f}")

        # ── 5. Risk Score (0–100) ──
        visual_risk = tamper_prob * 100

        if upi_verified:
            risk = int(visual_risk * 0.75)  # verified UPI reduces risk
        else:
            risk = int(visual_risk + 20)    # unverified UPI adds penalty

        risk = min(max(risk, 0), 100)
        result["risk_score"] = risk

        # ── 6. Verdict ──
        if risk < 25:
            result["status"] = "✅ Safe QR Code"
        elif risk < 55:
            result["status"] = "⚠️ Suspicious QR Code"
        else:
            result["status"] = "🚨 High Risk — Likely Tampered"

        # ── 7. Reasons ──
        if ela >= 0.5:
            result["reasons"].append(f"High compression artifact inconsistency (ELA: {ela:.2f})")
        if noise >= 0.5:
            result["reasons"].append(f"Abnormal noise pattern detected (score: {noise:.2f})")
        if edge >= 0.5:
            result["reasons"].append(f"Inconsistent edge sharpness (score: {edge:.2f})")
        if qr_data is None:
            result["reasons"].append("QR content could not be decoded")

    except Exception as e:
        result["status"] = "Error analyzing QR"
        result["reasons"].append(str(e))

    return result

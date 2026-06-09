"""
Teknofest IKA - Ortak Yardimci Fonksiyonlar (utils.py)

Degisiklikler:
  - load_params: DEFAULT_PARAMS ile deep merge (eksik key'ler otomatik tamamlanir)
  - save_params: Gecirilen dict'i degistirmez (copy uzerinde calisir)
  - Yeni: _deep_merge yardimci fonksiyonu
"""

import copy
import json
import os
from datetime import datetime

import numpy as np

DEFAULT_PARAMS = {
    "version": 1,
    "hsv_params": {
        "red_h_low_1": 0,
        "red_h_high_1": 10,
        "red_h_low_2": 160,
        "red_h_high_2": 180,
        "red_s_low": 70,
        "red_s_high": 255,
        "red_v_low": 50,
        "red_v_high": 255,
        "white_s_max": 40,
        "white_v_min": 200,
    },
    "morphology": {
        "kernel_size": 5,
        "kernel_shape": "ellipse"
    },
    "roi": {
        "top_ratio": 0.45
    },
    "canny": {
        "low": 50,
        "high": 150
    },
    "bev": {
        "src_points": None,
        "dst_points": None,
        "matrix_3x3": None
    },
    "notes": ""
}


def _deep_merge(base, override):
    """base uzerine override'i derinlemesine birlestir. Yeni dict dondurur."""
    result = {}
    all_keys = set(list(base.keys()) + list(override.keys()))
    for key in all_keys:
        if key in override and key in base:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                result[key] = _deep_merge(base[key], override[key])
            else:
                result[key] = override[key]
        elif key in override:
            result[key] = override[key]
        else:
            result[key] = base[key]
    return result


def save_params(params, filepath="calibration.json"):
    """Parametreleri JSON dosyasina kaydet. Orijinal dict'i degistirmez."""
    out = copy.deepcopy(params)
    out["timestamp"] = datetime.now().isoformat(timespec="seconds")

    bev = out.get("bev", {})
    for key in ("src_points", "dst_points"):
        if bev.get(key) is not None:
            bev[key] = np.array(bev[key]).tolist()
    if bev.get("matrix_3x3") is not None:
        bev["matrix_3x3"] = np.array(bev["matrix_3x3"]).tolist()

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[✓] Parametreler kaydedildi: {filepath}")


def load_params(filepath="calibration.json"):
    """
    JSON dosyasindan parametreleri yukle.
    Eksik key'ler DEFAULT_PARAMS'tan tamamlanir (eski JSON'larla uyumlu).
    """
    if not os.path.exists(filepath):
        print(f"[!] Dosya bulunamadi: {filepath} — varsayilanlar kullaniliyor.")
        return copy.deepcopy(DEFAULT_PARAMS)

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    merged = _deep_merge(copy.deepcopy(DEFAULT_PARAMS), data)

    bev = merged.get("bev", {})
    if bev.get("matrix_3x3") is not None:
        bev["matrix_3x3"] = np.array(bev["matrix_3x3"], dtype=np.float64)

    print(f"[✓] Parametreler yuklendi: {filepath}")
    return merged


def build_hsv_ranges(p):
    """HSV parametre dict'inden OpenCV mask araliklarini dondur."""
    hp = p["hsv_params"]
    red_low_1 = np.array([hp["red_h_low_1"], hp["red_s_low"], hp["red_v_low"]])
    red_high_1 = np.array([hp["red_h_high_1"], hp["red_s_high"], hp["red_v_high"]])
    red_low_2 = np.array([hp["red_h_low_2"], hp["red_s_low"], hp["red_v_low"]])
    red_high_2 = np.array([hp["red_h_high_2"], hp["red_s_high"], hp["red_v_high"]])
    white_low = np.array([0, 0, hp["white_v_min"]])
    white_high = np.array([180, hp["white_s_max"], 255])
    return red_low_1, red_high_1, red_low_2, red_high_2, white_low, white_high
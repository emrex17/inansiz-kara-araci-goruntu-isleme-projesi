"""
Teknofest 2026 — Yol Tespiti Kalibrasyon Araci (Duzeltilmis)

Degisiklikler:
  [BUG] Shallow copy → copy.deepcopy (DEFAULT_PARAMS artik bozulmuyor)
  [BUG] Cift kernel boyutu → tek sayiya yuvarlanir
  [BUG] Pipeline tutarsizligi → lane_detector ile ayni contour-tabanli tespit
  [YENİ] B: BEV onizleme toggle (matrix varsa)
  [YENİ] Sol/Sag ok: Frame ileri/geri (duraklatma modunda)
  [YENİ] R: Parametreleri varsayilana sifirla
  [YENİ] Overlay panelinde bariyer + yol merkezi + hata gosterimi

Kullanim: python calibrator.py --input video.mp4
"""

import copy
import sys
import os

import cv2
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from utils import DEFAULT_PARAMS, save_params, load_params, build_hsv_ranges

WINDOW_MAIN = "Kalibrasyon — 2×2 Grid"
WINDOW_PANEL = "Parametreler"
JSON_PATH = "calibration.json"
FRAME_W, FRAME_H = 848, 480
BEV_W, BEV_H = 640, 480
PANEL_W, PANEL_H = 424, 240

# ── Renk paleti ──────────────────────────────────────────────────────────────
C_BG = (24, 24, 32)
C_SECTION = (34, 34, 45)
C_TRACK = (55, 55, 68)
C_HANDLE = (230, 230, 240)
C_LABEL = (155, 155, 175)
C_VALUE = (220, 220, 235)
C_HINT = (85, 85, 105)
C_TITLE_BG = (18, 18, 26)
C_DIVIDER = (45, 45, 58)

GROUP_ACCENT = {
    "HSV - Kirmizi": (90, 80, 220),
    "HSV - Beyaz": (50, 190, 160),
    "Morfoloji": (80, 180, 90),
    "Ilgi Bolgesi": (200, 100, 170),
    "Kenar Algilama": (60, 170, 220),
}

# ── Slider tanimlari ──────────────────────────────────────────────────────────
SLIDER_DEFS = [
    ("HSV - Kirmizi", ("hsv_params", "red_h_low_1"), "H-Alt1", 0, 180),
    ("HSV - Kirmizi", ("hsv_params", "red_h_high_1"), "H-Ust1", 0, 180),
    ("HSV - Kirmizi", ("hsv_params", "red_h_low_2"), "H-Alt2", 0, 180),
    ("HSV - Kirmizi", ("hsv_params", "red_h_high_2"), "H-Ust2", 0, 180),
    ("HSV - Kirmizi", ("hsv_params", "red_s_low"), "S-Alt", 0, 255),
    ("HSV - Kirmizi", ("hsv_params", "red_s_high"), "S-Ust", 0, 255),
    ("HSV - Kirmizi", ("hsv_params", "red_v_low"), "V-Alt", 0, 255),
    ("HSV - Kirmizi", ("hsv_params", "red_v_high"), "V-Ust", 0, 255),
    ("HSV - Beyaz", ("hsv_params", "white_s_max"), "S-Maks", 0, 255),
    ("HSV - Beyaz", ("hsv_params", "white_v_min"), "V-Min", 0, 255),
    ("Morfoloji", ("morphology", "kernel_size"), "Cekirdek", 1, 31),
    ("Ilgi Bolgesi", ("roi", "__top_pct__"), "Ust %", 0, 90),
    ("Kenar Algilama", ("canny", "low"), "Alt", 0, 500),
    ("Kenar Algilama", ("canny", "high"), "Ust", 0, 500),
]

# Layout sabitleri
LABEL_W = 80
TRACK_X0 = 96
TRACK_W = 210
TRACK_H = 5
HANDLE_R = 7
ROW_H = 36
HDR_H = 30
TITLE_H = 38
PAD = 10
PW = TRACK_X0 + TRACK_W + 60 + PAD


# ─── Parametre erisim yardimcilari ───────────────────────────────────────────
def _get(params, path):
    if path[1] == "__top_pct__":
        return int(round(params["roi"]["top_ratio"] * 100))
    return params[path[0]][path[1]]


def _set(params, path, val):
    if path[1] == "__top_pct__":
        params["roi"]["top_ratio"] = max(0.0, min(0.9, val / 100.0))
    elif path == ("morphology", "kernel_size"):
        v = max(1, int(val))
        # Tek sayi zorunlulugu (morphologyEx icin)
        if v % 2 == 0:
            v += 1
        params["morphology"]["kernel_size"] = v
    else:
        params[path[0]][path[1]] = int(val)


# ─── SliderPanel sinifi ───────────────────────────────────────────────────────
class SliderPanel:
    def __init__(self, params):
        self.params = params
        self.dragging = None
        self.layout = []
        self._ph = 0
        self._build_layout()

    def _build_layout(self):
        self.layout = []
        y = TITLE_H + PAD
        group = None
        groups = []

        for sd in SLIDER_DEFS:
            grp, path, label, mn, mx = sd
            if grp != group:
                groups.append((grp, y))
                y += HDR_H
                group = grp
            yc = y + ROW_H // 2
            self.layout.append({
                "group": grp, "path": path, "label": label,
                "min": mn, "max": mx, "y": y, "yc": yc,
            })
            y += ROW_H

        self._ph = y + PAD + 22
        self._groups = groups

    def size(self):
        return PW, self._ph

    def _vx(self, v, mn, mx):
        r = (v - mn) / max(1, mx - mn)
        return int(TRACK_X0 + r * TRACK_W)

    def _xv(self, x, mn, mx):
        r = (x - TRACK_X0) / TRACK_W
        r = max(0.0, min(1.0, r))
        return int(round(mn + r * (mx - mn)))

    def draw(self):
        img = np.full((self._ph, PW, 3), C_BG, dtype=np.uint8)

        cv2.rectangle(img, (0, 0), (PW, TITLE_H), C_TITLE_BG, -1)
        cv2.putText(img, "PARAMETRELER", (PAD, TITLE_H - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 200), 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, TITLE_H - 2), (PW, TITLE_H), C_DIVIDER, -1)

        for gi, (grp, gy) in enumerate(self._groups):
            members = [s for s in self.layout if s["group"] == grp]
            y_end = members[-1]["y"] + ROW_H
            accent = GROUP_ACCENT[grp]

            cv2.rectangle(img, (PAD // 2, gy), (PW - PAD // 2, y_end), C_SECTION, -1)
            hdr_col = tuple(int(c * 0.45) for c in accent)
            cv2.rectangle(img, (PAD // 2, gy), (PW - PAD // 2, gy + HDR_H), hdr_col, -1)
            cv2.rectangle(img, (PAD // 2, gy), (PAD // 2 + 3, gy + HDR_H), accent, -1)
            cv2.putText(img, grp, (PAD // 2 + 9, gy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        tuple(min(255, int(c * 1.3)) for c in accent),
                        1, cv2.LINE_AA)

        for i, sd in enumerate(self.layout):
            v = _get(self.params, sd["path"])
            mn, mx = sd["min"], sd["max"]
            yc = sd["yc"]
            accent = GROUP_ACCENT[sd["group"]]
            hx = self._vx(v, mn, mx)

            cv2.putText(img, sd["label"], (PAD // 2 + 6, yc + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_LABEL, 1, cv2.LINE_AA)

            cv2.line(img, (TRACK_X0, yc), (TRACK_X0 + TRACK_W, yc), C_TRACK, TRACK_H,
                     cv2.LINE_AA)

            shadow = tuple(int(c * 0.5) for c in accent)
            cv2.line(img, (TRACK_X0, yc), (hx, yc), shadow, TRACK_H + 2, cv2.LINE_AA)
            cv2.line(img, (TRACK_X0, yc), (hx, yc), accent, TRACK_H, cv2.LINE_AA)

            cv2.circle(img, (hx, yc), HANDLE_R + 2, (10, 10, 18), -1, cv2.LINE_AA)
            cv2.circle(img, (hx, yc), HANDLE_R, accent, -1, cv2.LINE_AA)
            cv2.circle(img, (hx, yc), HANDLE_R - 3, C_HANDLE, -1, cv2.LINE_AA)

            txt = str(v) if sd["path"][1] != "__top_pct__" else f"{v}%"
            cv2.putText(img, txt, (TRACK_X0 + TRACK_W + 8, yc + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, accent, 1, cv2.LINE_AA)

        cv2.putText(img,
                    "S:Kaydet L:Yukle R:Sifirla B:BEV Space:Dur Ok:Frame Q:Cik",
                    (PAD // 2 + 2, self._ph - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_HINT, 1, cv2.LINE_AA)

        return img

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, sd in enumerate(self.layout):
                yc = sd["yc"]
                hx = self._vx(_get(self.params, sd["path"]), sd["min"], sd["max"])
                hit_handle = (x - hx) ** 2 + (y - yc) ** 2 <= (HANDLE_R + 6) ** 2
                hit_track = (TRACK_X0 <= x <= TRACK_X0 + TRACK_W
                             and abs(y - yc) <= TRACK_H + 6)
                if hit_handle or hit_track:
                    self.dragging = i
                    v = self._xv(x, sd["min"], sd["max"])
                    _set(self.params, sd["path"], v)
                    break

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            sd = self.layout[self.dragging]
            v = self._xv(x, sd["min"], sd["max"])
            _set(self.params, sd["path"], v)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None


# ─── Goruntu isleme pipeline ─────────────────────────────────────────────────
# lane_detector.py ile AYNI mantik: contour tabanli bariyer tespiti
def process_frame(frame, params, use_bev=False):
    """
    Kalibrasyon isleme pipeline'i.
    lane_detector.py ile ayni tespit mantigi kullanilir boylece
    kalibrasyonda gordugunuz = gercek calistirmada aldiginiz sonuc.
    """
    # ── BEV uygulamasi ────────────────────────────────────────────────
    bev = params.get("bev", {})
    bev_active = False

    if use_bev and bev.get("matrix_3x3") is not None:
        matrix = np.array(bev["matrix_3x3"], dtype=np.float64)
        work = cv2.warpPerspective(frame, matrix, (BEV_W, BEV_H))
        bev_active = True
    else:
        work = frame.copy()

    h, w = work.shape[:2]
    roi_top = int(h * params["roi"]["top_ratio"])

    # ── HSV maskeleme ─────────────────────────────────────────────────
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    r1, r2, r3, r4, wl, wh = build_hsv_ranges(params)

    mask_red = cv2.inRange(hsv, r1, r2) | cv2.inRange(hsv, r3, r4)
    mask_white = cv2.inRange(hsv, wl, wh)
    mask = mask_red | mask_white
    mask[:roi_top, :] = 0

    # ── Morfoloji (tek sayi kernel) ───────────────────────────────────
    ks = max(1, int(params["morphology"]["kernel_size"]))
    if ks % 2 == 0:
        ks += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    morph = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)

    # ── Canny ─────────────────────────────────────────────────────────
    edges = cv2.Canny(morph, int(params["canny"]["low"]),
                      int(params["canny"]["high"]))

    # ── Contour tabanli bariyer tespiti (lane_detector ile ayni) ──────
    y_start = roi_top
    bottom_roi = morph[y_start:h, :]

    contours, _ = cv2.findContours(
        bottom_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    clean_roi = np.zeros_like(bottom_roi)
    min_area = max(80, (w * h) // 5000)

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean_roi, [contour], -1, 255, -1)

    ys, xs = np.where(clean_roi > 0)
    left_x = None
    right_x = None

    if len(xs) >= 40:
        left_pixels = xs[xs < w // 2]
        right_pixels = xs[xs >= w // 2]

        if len(left_pixels) > 40:
            left_x = int(np.percentile(left_pixels, 90))
        if len(right_pixels) > 40:
            right_x = int(np.percentile(right_pixels, 10))

    # ── Yol merkezi hesaplama ─────────────────────────────────────────
    lane_center = None
    error = None
    direction = "YOL BULUNAMADI"
    camera_cx = w // 2

    if left_x is not None and right_x is not None:
        lane_center = (left_x + right_x) // 2
        error = lane_center - camera_cx
        if error > 25:
            direction = "SAGA DON"
        elif error < -25:
            direction = "SOLA DON"
        else:
            direction = "DUZ"

    # ── Overlay cizimi ────────────────────────────────────────────────
    result = work.copy()

    # HoughLines (gorsel referans)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=40, maxLineGap=20)
    if lines is not None:
        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.25:
                continue
            cv2.line(result, (x1, y1), (x2, y2), (0, 220, 90), 2)

    # Yardimci cizgiler
    cv2.line(result, (camera_cx, 0), (camera_cx, h), (255, 220, 0), 1)
    cv2.line(result, (0, roi_top), (w, roi_top), (255, 220, 0), 1)

    # Bariyerler
    if left_x is not None:
        cv2.line(result, (left_x, y_start), (left_x, h), (0, 255, 0), 2)
        cv2.putText(result, "SOL", (left_x + 4, y_start + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

    if right_x is not None:
        cv2.line(result, (right_x, y_start), (right_x, h), (0, 255, 0), 2)
        cv2.putText(result, "SAG", (max(4, right_x - 40), y_start + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

    # Yol merkezi + bilgi paneli
    if lane_center is not None:
        cv2.line(result, (lane_center, 0), (lane_center, h), (0, 0, 255), 3)
        steering = float(np.clip(error / 220.0, -1.0, 1.0))
        info = f"Hata:{error:+d}px  Yon:{direction}  Dir:{steering:+.2f}"
        cv2.rectangle(result, (6, 6), (w - 6, 34), (0, 0, 0), -1)
        cv2.putText(result, info, (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    else:
        cv2.rectangle(result, (6, 6), (w - 6, 34), (0, 0, 0), -1)
        cv2.putText(result, "YOL MERKEZI BULUNAMADI", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

    if bev_active:
        cv2.putText(result, "BEV", (w - 50, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ── 2x2 grid panelleri ────────────────────────────────────────────
    def panel(img):
        return cv2.resize(img, (PANEL_W, PANEL_H))

    def gray_panel(g):
        return panel(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))

    def label(img, txt):
        cv2.putText(img, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 220, 80), 1, cv2.LINE_AA)

    p1 = panel(work)
    p2 = gray_panel(morph)
    p3 = gray_panel(edges)
    p4 = panel(result)

    for img, txt in [(p1, "Islenen Goruntu"),
                     (p2, "Maske + Morfoloji"),
                     (p3, "Canny Kenar"),
                     (p4, "Tespit Sonucu")]:
        label(img, txt)

    return p1, p2, p3, p4


# ─── Ana dongu ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Yol Tespiti Kalibrasyon Araci")
    parser.add_argument("--input", default="0")
    parser.add_argument("--output", default=JSON_PATH)
    args = parser.parse_args()

    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[!] Video acilamadi: {args.input}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── deepcopy ile baslat (DEFAULT_PARAMS korunur) ──────────────────
    params = copy.deepcopy(DEFAULT_PARAMS)

    # Varsa kayitli parametreleri yukle
    if os.path.exists(args.output):
        params = load_params(args.output)

    panel = SliderPanel(params)
    pw, ph = panel.size()

    cv2.namedWindow(WINDOW_PANEL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_PANEL, pw, ph)
    cv2.setMouseCallback(WINDOW_PANEL, panel.mouse)

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, PANEL_W * 2, PANEL_H * 2)

    paused = False
    use_bev = False
    frame = None

    print("=" * 58)
    print("  Teknofest 2026 — Kalibrasyon Araci (Duzeltilmis)")
    print("  S: Kaydet  L: Yukle  R: Sifirla  B: BEV Toggle")
    print("  Space: Duraklat  Sol/Sag Ok: Frame Adimla  Q: Cik")
    print("=" * 58)

    while True:
        if not paused:
            ret, fr = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame = cv2.resize(fr, (FRAME_W, FRAME_H))

        if frame is None:
            continue

        p1, p2, p3, p4 = process_frame(frame, params, use_bev=use_bev)
        top = np.hstack([p1, p2])
        bottom = np.hstack([p3, p4])
        grid = np.vstack([top, bottom])

        # Durum bilgisi
        if paused:
            cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            status = f"|| DURAKLATILDI  Frame: {cur_pos}/{total_frames}"
            cv2.putText(grid, status,
                        (10, PANEL_H * 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_MAIN, grid)
        cv2.imshow(WINDOW_PANEL, panel.draw())

        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key in (ord('q'), 27):
            break

        elif key == ord(' '):
            paused = not paused
            print("[Space]", "Duraklatildi" if paused else "Devam ediyor")

        elif key in (ord('s'), ord('S')):
            save_params(params, args.output)

        elif key in (ord('l'), ord('L')):
            loaded = load_params(args.output)
            # Mevcut params referansini guncelle (panel baglantisi kopmaz)
            params.clear()
            params.update(loaded)
            print("[L] Parametreler yuklendi")

        elif key in (ord('r'), ord('R')):
            fresh = copy.deepcopy(DEFAULT_PARAMS)
            params.clear()
            params.update(fresh)
            print("[R] Parametreler varsayilana sifirlandi")

        elif key in (ord('b'), ord('B')):
            bev_cfg = params.get("bev", {})
            if bev_cfg.get("matrix_3x3") is not None:
                use_bev = not use_bev
                print(f"[B] BEV: {'Aktif' if use_bev else 'Pasif'}")
            else:
                print("[B] BEV matrisi bulunamadi — once BEV kalibrasyonu yapin")

        # Sol ok: bir frame geri (duraklatma modunda)
        elif key == 81 or key == 2:  # Sol ok (platform bagimliligi)
            if paused:
                pos = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, fr = cap.read()
                if ret:
                    frame = cv2.resize(fr, (FRAME_W, FRAME_H))

        # Sag ok: bir frame ileri (duraklatma modunda)
        elif key == 83 or key == 3:  # Sag ok
            if paused:
                ret, fr = cap.read()
                if ret:
                    frame = cv2.resize(fr, (FRAME_W, FRAME_H))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
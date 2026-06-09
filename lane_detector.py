"""
Teknofest IKA - Gercek Zamanli Yol Takip Sistemi

Degisiklik: `if writenot None` syntax hatasi duzeltildi.

Kullanim:
    python lane_detector.py --input ika_videosu.mp4
    python lane_detector.py --input 0
    python lane_detector.py --input ika_videosu.mp4 --no-bev
    python lane_detector.py --input ika_videosu.mp4 --record outputs/test_sonucu.mp4
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import load_params, build_hsv_ranges

FRAME_W, FRAME_H = 848, 480
BEV_W, BEV_H = 640, 480
WINDOW_NAME = "IKA - Gercek Zamanli Yol Takip"


class LaneDetector:
    def __init__(self, calibration_path="calibration.json", use_bev=True):
        self.params = load_params(calibration_path)
        self.use_bev = use_bev

        self.last_center = None
        self.smooth_alpha = 0.25

        self.lane_width_px = self._estimate_lane_width_from_bev()

    def _bev_matrix_ready(self):
        bev = self.params.get("bev", {})
        return bev.get("matrix_3x3") is not None

    def _estimate_lane_width_from_bev(self):
        """
        BEV dst_points sirasi:
        [sol-yakin, sag-yakin, sol-uzak, sag-uzak]
        """
        bev = self.params.get("bev", {})
        dst = bev.get("dst_points")

        if dst is None:
            return None

        pts = np.array(dst, dtype=np.float32)
        if pts.shape != (4, 2):
            return None

        near_width = abs(pts[1][0] - pts[0][0])
        far_width = abs(pts[3][0] - pts[2][0])
        return int((near_width + far_width) / 2)

    def apply_bev_if_available(self, frame):
        if not self.use_bev or not self._bev_matrix_ready():
            return frame.copy(), False

        matrix = np.array(self.params["bev"]["matrix_3x3"], dtype=np.float64)
        bev_frame = cv2.warpPerspective(frame, matrix, (BEV_W, BEV_H))
        return bev_frame, True

    def build_mask(self, frame):
        h, w = frame.shape[:2]
        roi_top = int(h * self.params["roi"]["top_ratio"])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r1, r2, r3, r4, white_low, white_high = build_hsv_ranges(self.params)

        mask_red = cv2.inRange(hsv, r1, r2) | cv2.inRange(hsv, r3, r4)
        mask_white = cv2.inRange(hsv, white_low, white_high)
        mask = mask_red | mask_white

        mask[:roi_top, :] = 0

        k = max(1, int(self.params["morphology"]["kernel_size"]))
        if k % 2 == 0:
            k += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask, roi_top

    def find_barriers_from_mask(self, mask, roi_top=None):
        """
        Maskeden sol ve sag bariyerin x konumunu bulur.
        roi_top verilirse bariyer aramasi oradan baslar (ROI ile tutarli).
        """
        h, w = mask.shape[:2]
        y_start = roi_top if roi_top is not None else int(h * 0.55)
        bottom_roi = mask[y_start:h, :]

        contours, _ = cv2.findContours(bottom_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        clean_roi = np.zeros_like(bottom_roi)
        min_area = max(80, (w * h) // 5000)

        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(clean_roi, [contour], -1, 255, -1)

        ys, xs = np.where(clean_roi > 0)

        if len(xs) < 40:
            return None, None, y_start, clean_roi

        left_pixels = xs[xs < w // 2]
        right_pixels = xs[xs >= w // 2]

        left_x = None
        right_x = None

        if len(left_pixels) > 40:
            left_x = int(np.percentile(left_pixels, 90))

        if len(right_pixels) > 40:
            right_x = int(np.percentile(right_pixels, 10))

        return left_x, right_x, y_start, clean_roi

    def calculate_lane_center(self, left_x, right_x, image_width, bev_used):
        reliable = True

        if left_x is not None and right_x is not None:
            center = int((left_x + right_x) / 2)

        elif bev_used and self.lane_width_px is not None and left_x is not None:
            center = int(left_x + self.lane_width_px / 2)
            reliable = False

        elif bev_used and self.lane_width_px is not None and right_x is not None:
            center = int(right_x - self.lane_width_px / 2)
            reliable = False

        elif self.last_center is not None:
            center = self.last_center
            reliable = False

        else:
            return None, None, False

        center = max(0, min(image_width - 1, center))

        if self.last_center is not None:
            center = int((1 - self.smooth_alpha) * self.last_center + self.smooth_alpha * center)

        self.last_center = center
        error = center - (image_width // 2)

        return center, error, reliable

    def direction_from_error(self, error):
        if error is None:
            return "YOL BULUNAMADI", 0.0

        threshold = 25
        steering = float(np.clip(error / 220.0, -1.0, 1.0))

        if error > threshold:
            return "SAGA DON", steering
        elif error < -threshold:
            return "SOLA DON", steering
        return "DUZ", steering

    def draw_hough_lines(self, overlay, edges):
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=40,
            minLineLength=40,
            maxLineGap=20
        )

        if lines is None:
            return

        for seg in lines:
            x1, y1, x2, y2 = seg[0]
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.25:
                continue

            cv2.line(overlay, (x1, y1), (x2, y2), (0, 220, 90), 2)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        work_frame, bev_used = self.apply_bev_if_available(frame)
        h, w = work_frame.shape[:2]

        mask, roi_top = self.build_mask(work_frame)
        edges = cv2.Canny(
            mask,
            int(self.params["canny"]["low"]),
            int(self.params["canny"]["high"])
        )

        left_x, right_x, y_start, clean_roi = self.find_barriers_from_mask(mask, roi_top)
        lane_center, error, reliable = self.calculate_lane_center(left_x, right_x, w, bev_used)
        direction, steering = self.direction_from_error(error)

        overlay = work_frame.copy()

        self.draw_hough_lines(overlay, edges)

        camera_center = w // 2
        cv2.line(overlay, (camera_center, 0), (camera_center, h), (255, 220, 0), 1)
        cv2.line(overlay, (0, roi_top), (w, roi_top), (255, 220, 0), 1)

        if left_x is not None:
            cv2.line(overlay, (left_x, y_start), (left_x, h), (0, 255, 0), 2)
            cv2.putText(overlay, "SOL BARIYER", (left_x + 5, y_start + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if right_x is not None:
            cv2.line(overlay, (right_x, y_start), (right_x, h), (0, 255, 0), 2)
            cv2.putText(overlay, "SAG BARIYER", (max(5, right_x - 150), y_start + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        if lane_center is not None:
            cv2.line(overlay, (lane_center, 0), (lane_center, h), (0, 0, 255), 3)

            status = "GUVENILIR" if reliable else "TAHMIN"
            info_1 = f"Kamera Merkez: {camera_center}px | Yol Merkez: {lane_center}px"
            info_2 = f"Hata: {error:+d}px | Yon: {direction} | Direksiyon: {steering:+.2f} | {status}"

            cv2.rectangle(overlay, (8, 8), (w - 8, 72), (0, 0, 0), -1)
            cv2.putText(overlay, info_1, (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(overlay, info_2, (18, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        else:
            cv2.rectangle(overlay, (8, 8), (w - 8, 46), (0, 0, 0), -1)
            cv2.putText(overlay, "YOL MERKEZI BULUNAMADI",
                        (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        debug = {
            "bev_used": bev_used,
            "left_x": left_x,
            "right_x": right_x,
            "lane_center": lane_center,
            "error": error,
            "direction": direction,
            "steering": steering,
            "reliable": reliable
        }

        return work_frame, mask, edges, overlay, debug


def put_label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (255, 220, 80), 1, cv2.LINE_AA)


def make_grid(original, mask, edges, overlay):
    panel_w, panel_h = 424, 240

    def resize_bgr(img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (panel_w, panel_h))

    p1 = resize_bgr(original)
    p2 = resize_bgr(mask)
    p3 = resize_bgr(edges)
    p4 = resize_bgr(overlay)

    put_label(p1, "Islenen Goruntu")
    put_label(p2, "HSV + ROI + Morfoloji Maske")
    put_label(p3, "Canny Kenar")
    put_label(p4, "Yol Takip Sonucu")

    top = np.hstack([p1, p2])
    bottom = np.hstack([p3, p4])
    return np.vstack([top, bottom])


def main():
    parser = argparse.ArgumentParser(description="IKA Gercek Zamanli Yol Takip Sistemi")
    parser.add_argument("--input", default="0", help="Video dosyasi veya kamera indeksi")
    parser.add_argument("--calibration", default="calibration.json", help="Kalibrasyon JSON dosyasi")
    parser.add_argument("--no-bev", action="store_true", help="BEV kullanmadan calistir")
    parser.add_argument("--record", default="", help="Sonucu video olarak kaydetmek icin dosya yolu")
    args = parser.parse_args()

    source = int(args.input) if str(args.input).isdigit() else args.input

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Video/kamera acilamadi: {args.input}")
        sys.exit(1)

    detector = LaneDetector(
        calibration_path=args.calibration,
        use_bev=not args.no_bev
    )

    writer = None
    last_time = time.time()
    fps = 0.0

    print("=" * 60)
    print("  IKA Gercek Zamanli Yol Takip Sistemi")
    print("  Q/Esc: Cik")
    print("  BEV:", "Aktif" if detector.use_bev and detector._bev_matrix_ready() else "Pasif")
    print("=" * 60)

    while True:
        ret, frame = cap.read()

        if not ret:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        original, mask, edges, overlay, debug = detector.process_frame(frame)
        grid = make_grid(original, mask, edges, overlay)

        now = time.time()
        dt = max(1e-6, now - last_time)
        last_time = now
        current_fps = 1.0 / dt
        fps = current_fps if fps == 0 else (0.9 * fps + 0.1 * current_fps)

        cv2.putText(grid, f"FPS: {fps:.1f}", (10, grid.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        if args.record:
            if writer is None:
                out_dir = os.path.dirname(args.record)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    args.record,
                    fourcc,
                    25,
                    (grid.shape[1], grid.shape[0])
                )

            writer.write(grid)

        cv2.imshow(WINDOW_NAME, grid)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()

    if writer is not None:
        writer.release()
        print(f"[✓] Video kaydedildi: {args.record}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()  
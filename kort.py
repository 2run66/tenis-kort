import cv2
import numpy as np
from sympy import Line

# -----------------------------------------------------------------------------
#  Yardımcı Fonksiyonlar
# -----------------------------------------------------------------------------

def line_intersection(line1, line2):
    """İki doğru kesişim noktasını (x, y) float olarak döndürür."""
    l1, l2 = Line(line1[0], line1[1]), Line(line2[0], line2[1])
    p = l1.intersection(l2)
    if not p:
        return None, None
    return float(p[0].x), float(p[0].y)


def display_lines_on_frame(frame, horizontal=(), vertical=()):
    """Bulunan çizgileri ekranda gösterir."""
    for x1, y1, x2, y2 in horizontal:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for x1, y1, x2, y2 in vertical:
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("Court lines", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
#  Ana Sınıf
# -----------------------------------------------------------------------------

class CourtDetector:
    """Beyaz tenis kort çizgilerini tespit eder (CourtReference olmadan)."""

    def __init__(self, verbose=0, threshold_mode="auto"):
        self.verbose = verbose
        self.threshold_mode = threshold_mode.lower()
        self.dist_tau = 3
        self.intensity_threshold = 40

    # ------------------------------------------------------------------
    # 1. Beyaz maske çıkarma (birden çok yöntem)
    # ------------------------------------------------------------------
    def _threshold(self, frame):
        mode = self.threshold_mode
        candidates = {}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, candidates["gray"] = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        _, candidates["otsu"] = cv2.threshold(gray, 0, 255,
                                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates["adaptive"] = cv2.adaptiveThreshold(gray, 255,
                                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv2.THRESH_BINARY, 31, -5)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        candidates["hsv"] = cv2.inRange(s, 0, 60) & cv2.inRange(v, 200, 255)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(lab)
        mask_lab = (cv2.inRange(L, 200, 255) &
                    cv2.inRange(a, 128-10, 128+10) &
                    cv2.inRange(b, 128-10, 128+10))
        candidates["lab"] = mask_lab

        if mode == "auto":
            mode = max(candidates, key=lambda k: np.count_nonzero(candidates[k]))
        if mode not in candidates:
            raise ValueError(f"threshold_mode='{self.threshold_mode}' tanımsız.")
        binary = candidates[mode]
        if self.verbose:
            print(f"[threshold] mode={mode}, white_px={np.count_nonzero(binary)}")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    # ------------------------------------------------------------------
    # 2. Çizgi yapısına göre piksel filtresi (opsiyonel)
    # ------------------------------------------------------------------
    def _filter_pixels(self, gray):
        g = gray.copy()
        h, w = g.shape
        for i in range(self.dist_tau, h - self.dist_tau):
            for j in range(self.dist_tau, w - self.dist_tau):
                if g[i, j] == 0:
                    continue
                if (g[i, j] - g[i + self.dist_tau, j] > self.intensity_threshold and
                        g[i, j] - g[i - self.dist_tau, j] > self.intensity_threshold):
                    g[i, j] = 0
                    continue
                if (g[i, j] - g[i, j + self.dist_tau] > self.intensity_threshold and
                        g[i, j] - g[i, j - self.dist_tau] > self.intensity_threshold):
                    g[i, j] = 0
        return g

    # ------------------------------------------------------------------
    # 3. Hough ile çizgi tespiti
    # ------------------------------------------------------------------
    def _detect_lines(self, gray):
        lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 80,
                                minLineLength=100, maxLineGap=20)
        if lines is None:
            return [], []
        lines = np.squeeze(lines)
        horiz, vert = self._classify_lines(lines)
        horiz, vert = self._merge_lines(horiz, vert)
        if self.verbose:
            print(f"[detect_lines] horizontal={len(horiz)}, vertical={len(vert)})")
        return horiz, vert

    def _classify_lines(self, lines):
        horizontal, vertical = [], []
        highest_y, lowest_y = np.inf, 0
        for x1, y1, x2, y2 in lines:
            dx, dy = abs(x1 - x2), abs(y1 - y2)
            if dx > 2 * dy:  # yatay
                horizontal.append([x1, y1, x2, y2])
            else:
                vertical.append([x1, y1, x2, y2])
                highest_y = min(highest_y, y1, y2)
                lowest_y  = max(lowest_y,  y1, y2)
        # Yatay çizgileri, dikeylerin kapsadığı aralıkla filtrele
        clean_h = []
        if vertical:
            h = lowest_y - highest_y
            lowest_y  += h / 15
            highest_y -= h * 2 / 15
            for x1, y1, x2, y2 in horizontal:
                if highest_y < y1 < lowest_y and highest_y < y2 < lowest_y:
                    clean_h.append([x1, y1, x2, y2])
        return clean_h, vertical

    def _merge_lines(self, horizontal_lines, vertical_lines):
        # Yatay birleştirme
        horizontal_lines.sort(key=lambda l: l[0])
        mask = [True] * len(horizontal_lines)
        merged_h = []
        for i, line in enumerate(horizontal_lines):
            if not mask[i]:
                continue
            for j, s_line in enumerate(horizontal_lines[i+1:], start=i+1):
                if not mask[j]:
                    continue
                x1, y1, x2, y2 = line
                x3, y3, x4, y4 = s_line
                if abs(y3 - y2) < 10:  # neredeyse aynı seviye
                    pts = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda p: p[0])
                    line = [*pts[0], *pts[-1]]
                    mask[j] = False
            merged_h.append(line)

        # Dikey birleştirme
        vertical_lines.sort(key=lambda l: l[1])
        mask = [True] * len(vertical_lines)
        merged_v = []
        xl, yl, xr, yr = 0, int(self.frame.shape[0]*6/7), self.frame.shape[1], int(self.frame.shape[0]*6/7)
        for i, line in enumerate(vertical_lines):
            if not mask[i]:
                continue
            for j, s_line in enumerate(vertical_lines[i+1:], start=i+1):
                if not mask[j]:
                    continue
                x1, y1, x2, y2 = line
                x3, y3, x4, y4 = s_line
                xi, _ = line_intersection(((x1, y1), (x2, y2)), ((xl, yl), (xr, yr)))
                xj, _ = line_intersection(((x3, y3), (x4, y4)), ((xl, yl), (xr, yr)))
                if xi is None or xj is None:
                    continue
                if abs(xi - xj) < 10:  # neredeyse aynı x
                    pts = sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], key=lambda p: p[1])
                    line = [*pts[0], *pts[-1]]
                    mask[j] = False
            merged_v.append(line)
        return merged_h, merged_v

    # ------------------------------------------------------------------
    # 4. Dış API
    # ------------------------------------------------------------------
    def detect(self, frame, verbose=None):
        """Bir karedeki yatay ve dikey kort çizgilerini döndürür."""
        if verbose is not None:
            self.verbose = verbose
        self.frame = frame  # bazı metodlar için kayıt
        mask = self._threshold(frame)
        filtered = self._filter_pixels(mask)
        horiz, vert = self._detect_lines(filtered)
        return horiz, vert

# -----------------------------------------------------------------------------
#  Örnek kullanım
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    img_path = r"C:\Users\eknbe\Downloads\frame.jpg"  # kendi yolunuzu yazın
    frame = cv2.imread(img_path)
    if frame is None:
        raise FileNotFoundError(img_path)

    detector = CourtDetector(verbose=1, threshold_mode="auto")
    h_lines, v_lines = detector.detect(frame)

    print(f"Bulunan yatay çizgi sayısı: {len(h_lines)}  dikey: {len(v_lines)}")

    out = frame.copy()
    display_lines_on_frame(out, h_lines, v_lines)

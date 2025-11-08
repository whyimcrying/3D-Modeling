import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.image import imread
from skimage.transform import resize

width, height = 1000, 1000
canvas = np.ones((height, width, 3), dtype=np.float32)

def put_pixel(x, y, color=(0, 0, 0), thickness=2):
    for dx in range(-thickness, thickness + 1):
        for dy in range(-thickness, thickness + 1):
            xx, yy = x + dx, y + dy
            if 0 <= xx < width and 0 <= yy < height:
                canvas[height - 1 - yy, xx] = color

def bresenham_line(x1, y1, x2, y2, color=(0, 0, 0), thickness=2, mask_hide=None):
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    sx, sy = (1 if x1 < x2 else -1), (1 if y1 < y2 else -1)
    err = dx - dy
    while True:
        if 0 <= x1 < width and 0 <= y1 < height:
            if mask_hide is None or not mask_hide[height - 1 - y1, x1]:
                put_pixel(x1, y1, color, thickness)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

def bresenham_circle(xc, yc, r, color=(0, 0, 0), thickness=2):
    x, y = 0, r
    d = 3 - 2 * r
    while y >= x:
        for dx, dy in [(x, y), (y, x), (-x, y), (-y, x),
                       (x, -y), (y, -x), (-x, -y), (-y, -x)]:
            put_pixel(xc + dx, yc + dy, color, thickness)
        x += 1
        if d > 0:
            y -= 1
            d += 4 * (x - y) + 10
        else:
            d += 4 * x + 6

def rotate_point(x, y, cx, cy, angle_deg):
    angle = math.radians(angle_deg)
    x_new = cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle)
    y_new = cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
    return int(round(x_new)), int(round(y_new))

def point_in_triangle(px, py, A, B, C):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    a = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / denom
    b = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / denom
    c = 1 - a - b
    return (0 <= a <= 1) and (0 <= b <= 1) and (0 <= c <= 1)

def expand_triangle(A, B, C, scale=1.12):
    cx = (A[0] + B[0] + C[0]) / 3
    cy = (A[1] + B[1] + C[1]) / 3
    def scale_point(p):
        x, y = p
        return (int(cx + (x - cx) * scale), int(cy + (y - cy) * scale))
    return scale_point(A), scale_point(B), scale_point(C)

def cyrus_beck_clip(p1, p2, polygon):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    t_enter, t_exit = 0, 1
    n = len(polygon)
    for i in range(n):
        xA, yA = polygon[i]
        xB, yB = polygon[(i + 1) % n]
        nx, ny = yA - yB, xB - xA
        wx, wy = x1 - xA, y1 - yA
        DdotN = dx * nx + dy * ny
        WdotN = wx * nx + wy * ny
        if DdotN == 0:
            if WdotN < 0:
                return None
            else:
                continue
        t = -WdotN / DdotN
        if DdotN > 0:
            t_enter = max(t_enter, t)
        else:
            t_exit = min(t_exit, t)
        if t_enter > t_exit:
            return None
    if t_enter > 1 or t_exit < 0:
        return None
    return t_enter, t_exit

A = (300, 650)
B = (500, 300)
C = (700, 650)
O = (500, 500)
r, r_, R = 85, 70, 350
A, B, C = [rotate_point(x, y, O[0], O[1], 180) for (x, y) in [A, B, C]]
A_, B_, C_ = expand_triangle(A, B, C, scale=1.12)

image_path = r"C:\Users\katam\PycharmProjects\PythonProject1\.venv\images.jpg"
try:
    img = imread(image_path)
    if img.dtype != np.uint8 and img.max() > 1.0:
        img = img / 255.0
    img_resized = resize(img, (height, width), anti_aliasing=True)
    mask = np.zeros((height, width), dtype=bool)
    for x in range(width):
        for y in range(height):
            if point_in_triangle(x, y, A, B, C):
                dist = math.sqrt((x - O[0])**2 + (y - O[1])**2)
                if dist > r:
                    mask[height - 1 - y, x] = True
    canvas[mask] = img_resized[mask]
except FileNotFoundError:
    print(f"нет картинки")

bresenham_line(*A, *B, thickness=3)
bresenham_line(*B, *C, thickness=3)
bresenham_line(*C, *A, thickness=3)

def dashed_line(p1, p2, dash_len=60, gap_len=20):
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    pos = 0.0
    while pos < length:
        end = min(pos + dash_len, length)
        t1 = pos / length
        t2 = end / length
        x_start = int(x1 + dx * t1)
        y_start = int(y1 + dy * t1)
        x_end = int(x1 + dx * t2)
        y_end = int(y1 + dy * t2)
        bresenham_line(x_start, y_start, x_end, y_end, thickness=2)
        pos += dash_len + gap_len

dashed_line(A_, B_)
dashed_line(B_, C_)
dashed_line(C_, A_)

for angle in range(0, 360, 60):
    for a in np.linspace(angle, angle + 30, 300):
        x = int(O[0] + r_ * math.cos(math.radians(a)))
        y = int(O[1] + r_ * math.sin(math.radians(a)))
        put_pixel(x, y, thickness=2)

bresenham_circle(O[0], O[1], r, thickness=2)

for a in np.linspace(-50, 180, 600):
    x = int(O[0] + R * math.cos(math.radians(a + 180)))
    y = int(O[1] + R * math.sin(math.radians(a + 180)))
    put_pixel(x, y, thickness=2)

axis_start = (80, 700)
axis_end = (950, 475)
clip = cyrus_beck_clip(axis_start, axis_end, [A, B, C])
if clip:
    t_in, t_out = clip
    dx, dy = axis_end[0] - axis_start[0], axis_end[1] - axis_start[1]
    enter_point = (int(axis_start[0] + dx * t_in), int(axis_start[1] + dy * t_in))
    exit_point = (int(axis_start[0] + dx * t_out), int(axis_start[1] + dy * t_out))
    bresenham_line(axis_start[0], axis_start[1], enter_point[0], enter_point[1], thickness=2)
    bresenham_line(exit_point[0], exit_point[1], axis_end[0], axis_end[1], thickness=2)
else:
    bresenham_line(*axis_start, *axis_end, thickness=2)

plt.figure(figsize=(10, 10))
plt.imshow(canvas)
plt.axis("off")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 400, 400

vertices = np.array([
    [0, 0, 0, 1],  # V0: Origin
    [100, 0, 0, 1],  # V1: Ось X
    [100, 100, 0, 1],  # V2: (100, 100)
    [0, 100, 0, 1],  # V3: Ось Y
    [0, 0, 150, 1]  # V4: Ось Z (вершина)
])

# Грани для заливки
faces_for_fill = [
    {'indices': [0, 1, 2], 'color': [0.85, 0.65, 0.95]},  # Очень светлый фиолетовый (lavender)
    {'indices': [0, 2, 3], 'color': [0.85, 0.65, 0.95]},  # Светлый фиолетовый
    {'indices': [1, 2, 4], 'color': [0.50, 0.20, 0.70]},  # Средний фиолетовый
    {'indices': [2, 3, 4], 'color': [0.30, 0.10, 0.45]},  # Тёмный фиолетовый
]

# Контуры для обводки (квадрат + треугольники)
outlines_to_draw = [
    [0, 1, 2, 3],
    [1, 2, 4],
    [2, 3, 4]
]

# Матрицы
def get_rotation_y(degrees):
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]])
def get_rotation_x(degrees):
    rad = np.radians(degrees)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

# Z-buffer (Барицентрические координаты)
def barycentric(x, y, p0, p1, p2):
    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-6: return -1, -1, -1
    lambda0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
    lambda1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2
def project_vertices(vertices, width, height):
    projected = []
    offset_x = width / 2
    offset_y = height / 2
    for v in vertices:
        px = int(v[0] + offset_x)
        py = int(v[1] + offset_y)
        pz = v[2]
        projected.append((px, py, pz))
    return projected
def draw_line_zbuffered(p0, p1, z_buffer, frame_buffer, color=[0, 0, 0], thickness=2):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    dist = int(np.hypot(x1 - x0, y1 - y0))
    if dist == 0: return
    xs = np.linspace(x0, x1, dist + 1)
    ys = np.linspace(y0, y1, dist + 1)
    zs = np.linspace(z0, z1, dist + 1)
    z_bias = 2.0
    brush_range = range(-(thickness // 2), (thickness // 2) + 1)
    for i in range(len(xs)):
        bx, by = int(xs[i]), int(ys[i])
        bz = zs[i]
        for ty in brush_range:
            for tx in brush_range:
                px, py = bx + tx, by + ty
                # Проверка границ экрана
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    # ПРОВЕРКА Z-БУФЕРА!
                    # Если Z текущей точки линии (+ смещение) больше того, что уже нарисовано,
                    # значит линия ближе к нам. Рисуем.
                    # Если линия за стеной, то условие не выполнится.
                    if (bz + z_bias) >= z_buffer[py, px]:
                        frame_buffer[py, px] = color
                        # Обновляем Z-буфер, чтобы линия перекрывала стыки
                        z_buffer[py, px] = bz + z_bias

def render_scene(projected_verts, faces, outlines, width, height):
    z_buffer = np.full((height, width), -np.inf)
    frame_buffer = np.ones((height, width, 3))
    for face in faces:
        inds = face['indices']
        p0, p1, p2 = projected_verts[inds[0]], projected_verts[inds[1]], projected_verts[inds[2]]
        min_x = max(0, min(p0[0], p1[0], p2[0]))
        max_x = min(width - 1, max(p0[0], p1[0], p2[0]))
        min_y = max(0, min(p0[1], p1[1], p2[1]))
        max_y = min(height - 1, max(p0[1], p1[1], p2[1]))
        color = np.array(face['color'])
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                l0, l1, l2 = barycentric(x, y, p0, p1, p2)
                if l0 >= 0 and l1 >= 0 and l2 >= 0:
                    z = l0 * p0[2] + l1 * p1[2] + l2 * p2[2]
                    if z > z_buffer[y, x]:
                        z_buffer[y, x] = z
                        frame_buffer[y, x] = color
    for loop_indices in outlines:
        for i in range(len(loop_indices)):
            idx1 = loop_indices[i]
            idx2 = loop_indices[(i + 1) % len(loop_indices)]  # Следующая вершина (замыкаем на начало)

            p_start = projected_verts[idx1]
            p_end = projected_verts[idx2]

            draw_line_zbuffered(p_start, p_end, z_buffer, frame_buffer, color=[0, 0, 0], thickness=3)

    return frame_buffer
M_total = get_rotation_x(10) @ get_rotation_y(135)
transformed_vertices = (M_total @ vertices.T).T
proj_verts = project_vertices(transformed_vertices, WIDTH, HEIGHT)
img = render_scene(proj_verts, faces_for_fill, outlines_to_draw, WIDTH, HEIGHT)

plt.figure(figsize=(6, 6))
plt.imshow(img, origin='lower')
plt.axis('off')
plt.show()
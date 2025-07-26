import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from colorsys import hls_to_rgb
from IPython.display import display, HTML

n_colors = 5


def hsl_to_rgb255(h, s, l):
    """Convert HSL (degrees, [0-1], [0-1]) to RGB [0-255]"""
    r, g, b = hls_to_rgb(h / 360.0, l, s)
    return np.array([r, g, b]) * 255


def pastel_distance_objective(hsl_flat, n):
    hsl = hsl_flat.reshape((n, 3))
    rgb_colors = np.array([hsl_to_rgb255(*hsl_i) for hsl_i in hsl])

    total_dist = 0
    for i, j in combinations(range(n), 2):
        total_dist += np.linalg.norm(rgb_colors[i] - rgb_colors[j])
    return -total_dist  # Negative for minimization


def generate_pastel_colors_hsl(n, seed=None):
    if seed is not None:
        np.random.seed(seed)

    bounds = []
    for _ in range(n):
        bounds.append((0, 360))  # Hue
        bounds.append((0.2, 0.5))  # Saturation
        bounds.append((0.75, 1.0))  # Lightness

    initial_guess = []
    for i in range(n):
        h = (360 / n) * i
        s = np.random.uniform(0.2, 0.5)
        l = np.random.uniform(0.75, 1.0)
        initial_guess.extend([h, s, l])

    result = minimize(
        pastel_distance_objective,
        np.array(initial_guess),
        args=(n,),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    hsl_colors = result.x.reshape((n, 3))
    rgb_colors = np.array([hsl_to_rgb255(*hsl_i) for hsl_i in hsl_colors])
    return np.clip(np.round(rgb_colors), 0, 255).astype(int)


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def display_color_palette(colors):
    html = "<div style='font-family: monospace'>"
    for i, rgb in enumerate(colors):
        hex_color = rgb_to_hex(rgb)
        html += f"""
        <div style='display: flex; align-items: center; margin: 4px 0'>
            <div style='width: 40px; height: 20px; background-color: {hex_color}; border: 1px solid #aaa; margin-right: 10px'></div>
            <span>Color {i+1}: RGB{tuple(rgb)} &nbsp; {hex_color}</span>
        </div>
        """
    html += "</div>"
    display(HTML(html))


pastels = generate_pastel_colors_hsl(n_colors, seed=42)
display_color_palette(pastels)

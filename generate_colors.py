import numpy as np
from scipy.optimize import minimize
from itertools import combinations
from IPython.display import display, HTML

# Example usage
n_colors = 10

def pastel_distance_objective(flat_rgb, n):
    """Objective function: negative of minimum pairwise distance."""
    colors = flat_rgb.reshape((n, 3))
    min_dist = np.inf
    for i, j in combinations(range(n), 2):
        dist = np.linalg.norm(colors[i] - colors[j])
        if dist < min_dist:
            min_dist = dist
    return -min_dist


def generate_pastel_colors(n, lower_bound=150, upper_bound=255, seed=None):
    if seed:
        np.random.seed(seed)

    bounds = [(lower_bound, upper_bound)] * (n * 3)
    initial_guess = np.random.uniform(lower_bound, upper_bound, size=(n * 3))

    result = minimize(
        pastel_distance_objective,
        initial_guess,
        args=(n,),
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    colors = result.x.reshape((n, 3))
    return np.clip(np.round(colors), 0, 255).astype(int)


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


pastels = generate_pastel_colors(n_colors, seed=42)
display_color_palette(pastels)

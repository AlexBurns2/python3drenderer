"""
Procedural Unity-like sky (improved)
- Left click + drag to look (sensitivity adjustable)
- A/D: move sun left/right
- W/S: move sun up/down
- [, ]: decrease/increase FOV
- +/-: change mouse sensitivity
- Q or Esc: quit
"""
import cv2
import numpy as np
import math

# --- CONFIG (tweak these) ---
WIDTH, HEIGHT = 900, 500
FOV_DEGREES = 90.0
MOUSE_SENSITIVITY = 0.1

# Color presets (RGB)
UNITY_NOON_TOP = np.array([90, 155, 255], dtype=np.float32)     # sky blue
UNITY_NOON_HORIZON = np.array([255, 255, 255], dtype=np.float32)  # white horizon
UNITY_NOON_GROUND = np.array([120, 120, 120], dtype=np.float32)   # grey ground

SUN_COLOR_DAY = np.array([255, 255, 220], dtype=np.float32)
SUN_COLOR_SUNSET = np.array([255, 150, 100], dtype=np.float32)

SUN_SIZE_RAD = 0.008   # angular radius of sun (radians)
SUN_INTENSITY = 4.5

# sunrise / sunset colors
SUNRISE_TOP = np.array([220, 120, 60], dtype=np.float32)   # warm orange top at sunrise
SUNRISE_HORIZON = np.array([255, 140, 80], dtype=np.float32)
SUNRISE_GROUND = np.array([120, 70, 30], dtype=np.float32)  # brown ground

# night colors
NIGHT_TOP = np.array([0, 0, 0], dtype=np.float32)
NIGHT_HORIZON = np.array([0, 0, 0], dtype=np.float32)
NIGHT_GROUND = np.array([0, 0, 0], dtype=np.float32)

# --- STATE ---
yaw = 0.0        # degrees
pitch = 0.0      # degrees
dragging = False
last_x, last_y = 0, 0

sun_yaw = 0.0    # degrees around horizon
sun_pitch = 60.0 # degrees elevation (-90..90). default near noon
fov_deg = FOV_DEGREES
sensitivity = MOUSE_SENSITIVITY

# ---------- utility ----------
def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

def lerp(a, b, t):
    return a * (1.0 - t) + b * t

def clamp(x, a, b):
    return max(a, min(b, x))

# ---------- main generator ----------
def generate_sky(width, height, fov_deg, pitch_deg, yaw_deg, sun_yaw_deg, sun_pitch_deg):
    """
    Render sky with:
      - Correct camera rotation (pitch applied first, then yaw)
      - Flat ground plane (grey/brown/black depending on time)
      - Sun disc (angular falloff, circular)
      - Day/night color mapping: night->sunrise->noon->sunset->night
    """

    # Convert to radians & constants
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    sun_yaw = math.radians(sun_yaw_deg)
    sun_pitch = math.radians(sun_pitch_deg)
    fov = math.radians(fov_deg)
    aspect = width / height

    # Screen space coordinates (IMPORTANT: y positive is UP)
    # so top row should have positive y value -> use ys descending from +1 to -1
    xs = np.linspace(-1.0, 1.0, width)
    ys = np.linspace(1.0, -1.0, height)   # top = +1, bottom = -1
    xv, yv = np.meshgrid(xs, ys)

    # Convert to camera-space direction vectors using angular mapping
    # x_angle horizontally, y_angle vertically (account for aspect)
    x_angle = xv * (fov / 2.0)
    y_angle = yv * (fov / 2.0) / aspect
    dir_x = np.sin(x_angle)
    dir_y = np.sin(y_angle)
    dir_z = np.cos(x_angle) * np.cos(y_angle)
    dirs = np.stack([dir_x, dir_y, dir_z], axis=-1)
    dirs = normalize(dirs)   # shape (H,W,3)

    # APPLY CAMERA ROTATION: pitch FIRST (local X), then yaw (world Y)
    # Rotation matrices:
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    rot_pitch = np.array([
        [1,     0,      0],
        [0,  cos_p, -sin_p],
        [0,  sin_p,  cos_p]
    ], dtype=np.float32)

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rot_yaw = np.array([
        [ cos_y, 0, sin_y],
        [ 0,     1,   0 ],
        [-sin_y, 0, cos_y]
    ], dtype=np.float32)

    # Compose so pitch applied first: R_total = R_yaw @ R_pitch
    R = rot_yaw @ rot_pitch   # rotate vectors: v' = R * v  (we will do dirs @ R.T)
    dirs = dirs @ R.T

    # SUN DIRECTION
    sun_dir = np.array([
        math.sin(sun_yaw) * math.cos(sun_pitch),
        math.sin(sun_pitch),
        math.cos(sun_yaw) * math.cos(sun_pitch)
    ], dtype=np.float32)
    sun_dir = sun_dir / np.linalg.norm(sun_dir)

    # Determine sun elevation factor and daylight
    # sun_pitch in radians: -pi/2..pi/2. Use sin(sun_pitch) ~ elevation (-1..1)
    sun_elev = math.sin(sun_pitch)  # -1..1
    # daylight_scalar roughly 0 at night, 1 at high noon
    daylight = clamp((sun_elev + 0.15) / 1.15, 0.0, 1.0)  # small bias so near horizon still dim

    # Determine phase mapping for colors:
    # We'll smoothly map according to sun_pitch:
    #  - sun_pitch <= -5deg  -> NIGHT
    #  - -5 < sun_pitch < 10 -> SUNRISE/SUNSET (orange -> transition),
    #  - >=10deg -> NOON (blue)
    sp_deg = math.degrees(sun_pitch)
    if sp_deg <= -5.0:
        # Night
        top_color = NIGHT_TOP
        horizon_color = NIGHT_HORIZON
        ground_color = NIGHT_GROUND
        sun_strength = 0.0
    elif sp_deg < 10.0:
        # Sunrise / Sunset region: interpolate between sunrise palette and noon
        t = (sp_deg + 5.0) / 15.0   # 0 @ -5deg, 1 @ +10deg
        # top moves SUNRISE_TOP -> UNITY_NOON_TOP
        top_color = lerp(SUNRISE_TOP, UNITY_NOON_TOP, t)
        horizon_color = lerp(SUNRISE_HORIZON, UNITY_NOON_HORIZON, t)
        ground_color = lerp(SUNRISE_GROUND, UNITY_NOON_GROUND, t)
        sun_strength = clamp(t * 1.3, 0.0, 1.0)  # stronger as it climbs
    else:
        # Noon-ish
        top_color = UNITY_NOON_TOP
        horizon_color = UNITY_NOON_HORIZON
        ground_color = UNITY_NOON_GROUND
        sun_strength = 1.0

    # Additional evening warm tint when sun near horizon on the far side:
    # If daylight low but sun near horizon (sp_deg near 0), make sun_color warm
    sunset_warm = 0.0
    if -5.0 < sp_deg < 20.0:
        sunset_warm = clamp(1.0 - abs((sp_deg - 5.0) / 15.0), 0.0, 1.0)

    sun_color = lerp(SUN_COLOR_SUNSET, SUN_COLOR_DAY, sun_strength)
    # Add a bit more warmth at low elevation
    sun_color = lerp(sun_color, SUN_COLOR_SUNSET, sunset_warm * (1.0 - daylight))

    # Sky rendering:
    upness = dirs[..., 1]   # +1 up, -1 down
    t_up = (upness + 1.0) / 2.0  # 0..1

    # Smooth horizon scattering term (strongest near t_up==0.5)
    scatter = np.exp(-((t_up - 0.5) ** 2) / 0.012) * (0.7 + 0.3 * daylight)

    # Interpolate between top and horizon using t_up (sky above)
    sky_color = top_color * (t_up[..., None]) + horizon_color * (1.0 - t_up[..., None])

    # Ground color: flat plane - always shown for directions pointing down (upness<0)
    # For downward directions we want a constant ground color, not spherical ground.
    # We'll make ground only for rays whose direction points below horizon (upness < 0)
    ground = np.ones_like(sky_color) * ground_color  # broadcast

    # Compose image: if upness>0 => sky_color, else ground
    img = np.where(upness[..., None] > 0.0, sky_color, ground)

    # Apply scattering glow to horizon band (adds brightness/color toward horizon)
    # Blend horizon_color into the sky near horizon
    img += (scatter[..., None] * (horizon_color * 0.15))

    # SUN DISC: compute angular distance between pixel direction and sun_dir
    dot = np.sum(dirs * sun_dir.reshape((1,1,3)), axis=-1)  # cos(angle)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.arccos(dot)  # angle in radians between ray and sun direction

    # Soft circular sun: gaussian falloff based on angular radius
    sun_mask = np.exp(-0.5 * (ang / SUN_SIZE_RAD) ** 2)
    # Sun intensity scales with daylight
    img += (sun_mask[..., None] * sun_color * SUN_INTENSITY * (0.2 + 0.8 * daylight))

    # At night, darken the entire image toward black
    if daylight <= 0.01:
        img[:] = NIGHT_TOP  # pure black sky & ground

    # Tone mapping / clamp
    # Slight exposure control by daylight: darker at night
    exposure = 0.3 + 0.7 * daylight
    img = img * exposure

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ---------- input handlers ----------
def mouse_callback(event, x, y, flags, param):
    global dragging, last_x, last_y, yaw, pitch, sensitivity
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        last_x, last_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - last_x
        dy = y - last_y
        yaw += dx * sensitivity
        pitch += dy * sensitivity
        pitch = clamp(pitch, -89.0, 89.0)
        last_x, last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

# ---------- run window ----------
if __name__ == "__main__":
    cv2.namedWindow("Unity-like Procedural Sky", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Unity-like Procedural Sky", WIDTH, HEIGHT)
    cv2.setMouseCallback("Unity-like Procedural Sky", mouse_callback)

    print("Controls: drag = look, A/D = sun left/right, W/S = sun up/down, [,] = FOV, +/- = sensitivity, Q/Esc = quit")

    while True:
        frame = generate_sky(WIDTH, HEIGHT, fov_deg, pitch, yaw, sun_yaw, sun_pitch)
        cv2.imshow("Unity-like Procedural Sky", frame)
        key = cv2.waitKey(16) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key == ord('a'):
            sun_yaw -= 2.0
        elif key == ord('d'):
            sun_yaw += 2.0
        elif key == ord('w'):
            sun_pitch = clamp(sun_pitch + 1.0, -89.0, 89.0)
        elif key == ord('s'):
            sun_pitch = clamp(sun_pitch - 1.0, -89.0, 89.0)
        elif key == ord('['):
            fov_deg = clamp(fov_deg - 5.0, 20.0, 160.0)
        elif key == ord(']'):
            fov_deg = clamp(fov_deg + 5.0, 20.0, 160.0)
        elif key == ord('+') or key == ord('='):
            sensitivity = clamp(sensitivity + 0.05, 0.01, 5.0)
            print(f"sensitivity={sensitivity:.2f}")
        elif key == ord('-') or key == ord('_'):
            sensitivity = clamp(sensitivity - 0.05, 0.01, 5.0)
            print(f"sensitivity={sensitivity:.2f}")

    cv2.destroyAllWindows()

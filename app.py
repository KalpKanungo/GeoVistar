import gradio as gr
import cv2
import numpy as np
import torch
import tempfile
import copy
from depth_infer import load_model, infer_depth
from geometry import create_intrinsics, backproject
from plane_fitting import fit_plane, plane_distance

torch.set_grad_enabled(False)

model, transform = load_model()
model.to("cpu")
model.eval()

# ========================
# FLOOR
# ========================
def compute_floor(image):

    image = cv2.resize(image, (512, 384))
    depth = infer_depth(model, transform, image)

    h, w = depth.shape
    fx, fy, cx, cy = create_intrinsics(w, h)
    points = backproject(depth, fx, fy, cx, cy)

    a, b, c, d = fit_plane(points)
    dist = plane_distance(points, a, b, c, d)

    bottom_mask = np.zeros_like(dist, dtype=bool)
    bottom_mask[int(h * 0.5):, :] = True

    floor_mask = (dist < 0.01) & bottom_mask
    floor_mask = floor_mask.astype("uint8") * 255

    highlight = image.copy()
    highlight[floor_mask == 255] = (
        0.7 * highlight[floor_mask == 255] +
        0.3 * np.array([0, 255, 0])
    ).astype(np.uint8)

    return image, highlight, floor_mask


# ========================
# ROTATION
# ========================
def rotate_image(image, angle):

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    return rotated


# ========================
# COLLISION CHECK
# ========================
def is_overlapping(scene, x, y, threshold=35):

    for obj in scene["objects"]:
        if abs(obj["x"] - x) < threshold and abs(obj["y"] - y) < threshold:
            return True

    return False


# ========================
# RENDER
# ========================
def render_scene(scene, clean=False):

    base = scene["base"].copy() if clean else scene["highlight"].copy()

    for i, obj in enumerate(scene["objects"]):

        furniture = obj["image"].copy()

        if obj["flip"]:
            furniture = cv2.flip(furniture, 1)

        if obj["rotation"] != 0:
            furniture = rotate_image(furniture, obj["rotation"])

        fh, fw = furniture.shape[:2]
        new_w = int(fw * obj["scale"])
        new_h = int(fh * obj["scale"])

        if new_w < 10 or new_h < 10:
            continue

        furniture = cv2.resize(furniture, (new_w, new_h))

        x1 = int(obj["x"] - new_w // 2)
        y1 = int(obj["y"] - new_h)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(base.shape[1], x1 + new_w)
        y2 = min(base.shape[0], y1 + new_h)

        fw_crop = x2 - x1
        fh_crop = y2 - y1

        furniture = furniture[:fh_crop, :fw_crop]

        if furniture.shape[2] == 4:
            alpha = furniture[:, :, 3] / 255.0
            for c in range(3):
                base[y1:y2, x1:x2, c] = (
                    alpha[:fh_crop, :fw_crop] * furniture[:, :, c] +
                    (1 - alpha[:fh_crop, :fw_crop]) * base[y1:y2, x1:x2, c]
                )

        if scene["selected"] == i and not clean:
            cv2.rectangle(base, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return base


# ========================
# HISTORY
# ========================
def push_history(scene):
    scene["history"].append(copy.deepcopy(scene["objects"]))
    scene["redo_stack"] = []
    return scene


def undo(scene):
    if scene["history"]:
        scene["redo_stack"].append(copy.deepcopy(scene["objects"]))
        scene["objects"] = scene["history"].pop()
    return scene, render_scene(scene)


def redo(scene):
    if scene["redo_stack"]:
        scene["history"].append(copy.deepcopy(scene["objects"]))
        scene["objects"] = scene["redo_stack"].pop()
    return scene, render_scene(scene)


# ========================
# CLICK
# ========================
def handle_click(scene, size_scale, flip, rotation, evt: gr.SelectData):

    if scene is None:
        return None, None

    x, y = evt.index

    if scene["move_mode"] and scene["selected"] is not None:
        scene["objects"][scene["selected"]]["x"] = x
        scene["objects"][scene["selected"]]["y"] = y
        return scene, render_scene(scene)

    for i, obj in enumerate(scene["objects"]):
        if abs(obj["x"] - x) < 40 and abs(obj["y"] - y) < 40:
            scene["selected"] = i
            return scene, render_scene(scene)

    if scene["active_furniture"] is None:
        return scene, render_scene(scene)

    if scene["floor_mask"][y, x] == 0:
        return scene, render_scene(scene)

    if is_overlapping(scene, x, y):
        return scene, render_scene(scene)

    scene = push_history(scene)

    scene["objects"].append({
        "image": scene["active_furniture"],
        "x": x,
        "y": y,
        "scale": size_scale,
        "flip": flip,
        "rotation": rotation
    })

    scene["selected"] = len(scene["objects"]) - 1

    return scene, render_scene(scene)


# ========================
# DELETE
# ========================
def delete_selected(scene):

    if scene["selected"] is not None:
        scene = push_history(scene)
        scene["objects"].pop(scene["selected"])
        scene["selected"] = None

    return scene, render_scene(scene)


# ========================
# RESET
# ========================
def reset_scene(scene):

    scene["objects"] = []
    scene["selected"] = None
    scene["history"] = []
    scene["redo_stack"] = []

    return scene, scene["highlight"]


# ========================
# MOVE TOGGLE
# ========================
def toggle_move(scene):
    scene["move_mode"] = not scene["move_mode"]
    return scene


# ========================
# DOWNLOAD
# ========================
def download_image(scene):

    final = render_scene(scene, clean=True)
    temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cv2.imwrite(temp.name, cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    return temp.name


# ========================
# UI
# ========================
with gr.Blocks() as demo:

    furniture_state = gr.State([])
    scene_state = gr.State()
    active_furniture_state = gr.State()

    with gr.Tab("Furniture Library"):

        upload_furniture = gr.File(file_count="multiple", file_types=["image"])
        furniture_gallery = gr.Gallery(columns=4)

        def load_furniture(files):
            images = []
            for f in files:
                img = cv2.imread(f.name, cv2.IMREAD_UNCHANGED)
                images.append(img)
            return images, images

        upload_furniture.change(
            load_furniture,
            inputs=upload_furniture,
            outputs=[furniture_gallery, furniture_state]
        )

        def select_furniture(lib, evt: gr.SelectData):
            return lib[evt.index]

        furniture_gallery.select(
            select_furniture,
            inputs=furniture_state,
            outputs=active_furniture_state
        )

    with gr.Tab("Room Designer"):

        room_display = gr.Image(type="numpy", label="Upload Room & Click to Place")

        # ===== Row 1 =====
        with gr.Row():
            size_slider = gr.Slider(0.2, 1.5, value=0.5, step=0.05, label="Size", scale=2)
            rotation_slider = gr.Slider(-180, 180, value=0, step=5, label="Rotation", scale=2)
            flip_checkbox = gr.Checkbox(label="Flip", scale=1)

        # ===== Row 2 =====
        with gr.Row():
            move_btn = gr.Button("Move", size="sm")
            delete_btn = gr.Button("Delete", size="sm")
            reset_btn = gr.Button("Reset", size="sm")

        # ===== Row 3 =====
        with gr.Row():
            undo_btn = gr.Button("Undo", size="sm")
            redo_btn = gr.Button("Redo", size="sm")
            download_btn = gr.Button("Download", size="sm")

        download_file = gr.File()

        def preprocess(image):

            base, highlight, floor_mask = compute_floor(image)

            scene = {
                "base": base,
                "highlight": highlight,
                "floor_mask": floor_mask,
                "objects": [],
                "selected": None,
                "active_furniture": None,
                "move_mode": False,
                "history": [],
                "redo_stack": []
            }

            return highlight, scene

        room_display.upload(
            preprocess,
            inputs=room_display,
            outputs=[room_display, scene_state]
        )

        def set_active(scene, active):
            scene["active_furniture"] = active
            return scene

        active_furniture_state.change(
            set_active,
            inputs=[scene_state, active_furniture_state],
            outputs=scene_state
        )

        room_display.select(
            handle_click,
            inputs=[scene_state, size_slider, flip_checkbox, rotation_slider],
            outputs=[scene_state, room_display]
        )

        move_btn.click(toggle_move, inputs=scene_state, outputs=scene_state)
        delete_btn.click(delete_selected, inputs=scene_state, outputs=[scene_state, room_display])
        reset_btn.click(reset_scene, inputs=scene_state, outputs=[scene_state, room_display])
        undo_btn.click(undo, inputs=scene_state, outputs=[scene_state, room_display])
        redo_btn.click(redo, inputs=scene_state, outputs=[scene_state, room_display])
        download_btn.click(download_image, inputs=scene_state, outputs=download_file)

demo.launch()
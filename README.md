---
title: GeoVistar

emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: gradio
emoji: 🌖
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.7.0
app_file: app.py
pinned: false
---

# 🏠 GeoVistar: AI-Powered Augmented Reality Room Designer

> 📐 Upload a room photo, detect the floor automatically, and place virtual furniture in real-time using depth estimation, geometric plane fitting, and scene segmentation.

---

## 🚀 Live Demo

👉 **Try it here:**
🔗 https://huggingface.co/spaces/kalpkanungo/GeoVistar

---

## 📌 Overview

GeoVistar is an end-to-end AR room design system that lets you **visualize furniture in your actual room** before buying or rearranging.

It combines:
- **Monocular depth estimation** to understand 3D room structure
- **Geometric plane fitting** to detect the floor surface
- **Alpha-blended furniture overlay** for realistic placement
- **Interactive scene editing** with undo/redo, move, rotate, flip, and delete

---

## 🖼️ Try It With Sample Images

The app comes with built-in sample rooms and furniture — no uploads needed to get started!

| Sample Rooms | Sample Furniture |
|---|---|
| 3 ready-to-use room photos | Chair, Sofa, Side Table |

Just open the app, click a sample room in the **Room Designer** tab, then click a sample furniture piece in the **Furniture Library** tab and start placing!

---

## ⚙️ Key Features

### 🔍 Automatic Floor Detection
- Upload any room photo
- Depth map is inferred using a pretrained model (DPT/MiDaS)
- Floor plane is fitted using geometric plane fitting
- Only valid floor pixels accept furniture placement — no floating objects

---

### 🛋️ Interactive Furniture Placement
- Upload your own furniture images (PNG with transparency) or use samples
- Click anywhere on the detected floor to place furniture
- Full control over each object:
  - **Size** slider
  - **Rotation** slider (-180° to 180°)
  - **Flip** toggle
  - **Move** mode for repositioning

---

### 🔄 Scene Management
- **Undo / Redo** full action history
- **Delete** selected objects
- **Reset** scene to original room
- **Download** final render as a clean PNG (no bounding boxes)

---

## 🏗️ System Architecture
```
Room Image Upload
       ↓
Depth Estimation (DPT Model)
       ↓
Camera Intrinsics + Backprojection → 3D Point Cloud
       ↓
Geometric Plane Fitting (Floor Detection)
       ↓
Floor Mask Generation
       ↓
User Clicks on Floor → Furniture Placement
       ↓
Alpha Blending + Scale / Rotation / Flip
       ↓
Rendered Scene (with Undo/Redo History)
       ↓
Clean PNG Download
```

---

## 🛠️ Tech Stack

- **Depth Estimation:** DPT / MiDaS (via Hugging Face Transformers)
- **Geometry:** NumPy, SciPy (plane fitting, backprojection)
- **Scene Rendering:** OpenCV (alpha blending, rotation, flip)
- **Segmentation:** YOLOv8 (Ultralytics)
- **Frontend:** Gradio
- **Deployment:** Hugging Face Spaces

---

## 📈 Key Highlights

- 🎯 Floor-constrained placement — furniture can only be placed on detected floor pixels
- 🔄 Full undo/redo history for non-destructive editing
- 🖼️ Supports transparent PNG furniture with alpha blending
- ⚡ Runs entirely on CPU — no GPU required
- 💾 Export clean renders without UI overlays
- 🧪 Built-in sample rooms and furniture for instant demo

---

## 🚀 Getting Started (Local Setup)
```bash
git clone https://github.com/KalpKanungo/GeoVistar
cd GeoVistar
pip install -r requirements.txt
python app.py
```

---

## 💡 Future Improvements

- Add a built-in furniture library with more pieces
- Support wall detection for hanging decor placement
- Add lighting/shadow simulation for more realistic overlays
- Integrate 3D furniture models with perspective-correct rendering
- Mobile-friendly UI

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!
Feel free to fork the repo and open a pull request.

---

## ⭐ If you found this useful

Give it a ⭐ on GitHub — it helps a lot!
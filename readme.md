# 3D Cell Viewer - PhD Pipeline Instructions
---

## 📋 Complete Setup (First Time)

### 1. Prerequisites

- **Python 3.12** (recommended for package compatibility)
- **NVIDIA GPU** (any CUDA-compatible GPU: RTX 30xx, 40xx, 50xx, etc.)
- **NVIDIA Studio Drivers** (recommended) or Game Ready drivers
- **CUDA Toolkit 12.6+** from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
- **Microsoft Visual C++ Build Tools** from [Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### 2. Create Virtual Environment

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify GPU Detection

```powershell
.\venv\Scripts\python.exe -c "import pyclesperanto_prototype as cle; print(cle.get_device())"
```

Should output your GPU, e.g.: `<NVIDIA GeForce RTX 3060 on Platform: NVIDIA CUDA>`

### 4. Fix CUDA Path Warning (Optional)

If you see a CUDA path warning, set the environment variable:

```powershell
[System.Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6", "User")
# Restart terminal after running this
```

# Run the viewer
.\venv\Scripts\python.exe scripts\run_viewer.py
```

---

## 🖥️ GPU Compatibility

This code works with **any NVIDIA GPU** that supports CUDA. The processing libraries (pyclesperanto, RedLionfish) use OpenCL which auto-detects your GPU.

| GPU | Status | Notes |
|-----|--------|-------|
| RTX 5090 (32GB) | ✅ Fastest | Large images, many iterations |
| RTX 4090 (24GB) | ✅ Excellent | Very fast processing |
| RTX 3060 (12GB) | ✅ Good | May need smaller images for large datasets |
| RTX 2080 (8GB) | ✅ Works | Process one channel at a time |
| GTX 1080 (8GB) | ✅ Works | Slower, but functional |

**Memory Tips for smaller GPUs:**
- Process one channel at a time
- Use fewer deconvolution iterations
- Crop regions of interest before processing
- Close other GPU applications

---

## 🎯 Getting the Best 3D Images

### Step 1: Load Your File

1. Run the script: `.\venv\Scripts\python.exe scripts\run_viewer.py`
2. A file dialog opens → Select your `.czi` file
3. The viewer opens in **3D mode** with all channels loaded

### Step 2: Initial 3D View Optimization

In the napari window:

| Action | How |
|--------|-----|
| **Rotate view** | Click + drag with left mouse button |
| **Zoom** | Scroll wheel |
| **Pan** | Click + drag with middle mouse button |
| **Reset view** | Press `Home` key |

### Step 3: Optimize Rendering Settings

For each channel layer (left panel):

1. **Click the layer name** to select it
2. **Expand layer controls** (click the arrow)
3. Adjust these settings:

| Setting | Recommended Value | Purpose |
|---------|-------------------|---------|
| **Rendering** | `attenuated_mip` | Best for fluorescence |
| **Attenuation** | 0.02 - 0.08 | Lower = more transparent |
| **Contrast limits** | Adjust sliders | Remove background noise |
| **Gamma** | 0.8 - 1.2 | Adjust midtones |

---

## 🎛️ Available Widgets

The viewer includes several processing widgets in the right panel:

### 🔬 GPU Deconvolution

Richardson-Lucy deconvolution for sharpening images. Best for high-NA objectives (40x-100x oil).

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| **Iterations** | 50-150 | More = sharper but slower |
| **Wavelength** | Match your fluorophore | See table below |
| **NA** | Your objective's NA | 1.4 for oil, 0.8 for 20x air |
| **Use GPU** | ✅ Checked | Uses your NVIDIA GPU |

**⚠️ Note:** Deconvolution works best with high-NA objectives. For 20x/0.80 NA with large pixels, you may get artifacts. In that case, skip deconvolution and use GPU Enhance or manual contrast adjustment instead.

### ⚡ GPU Enhance

Quick image enhancement using GPU-accelerated filters:

| Setting | Effect |
|---------|--------|
| **Denoise** ✅ | Removes speckle noise (median filter) |
| **Sharpen** ✅ | Enhances edges (unsharp mask) |
| **Sigma** | 1.5-3.0 for subtle, 4-6 for strong |

### 🎨 Thesis Quality Enhancement

Advanced enhancement for maximum visual impact:

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| **Background Subtract** | Removes haze/fog | ✅ ON, radius 50-80 |
| **Local Contrast** | Makes fine structures pop (CLAHE) | 2.0-3.0 |
| **Gamma** | Brightens dim features | 0.7-0.9 (lower = brighter) |
| **Percentile Low/High** | Contrast stretch | 1.0% / 99.8% |

**💡 Tip:** If your images are already good quality, aggressive enhancement may lose detail. Sometimes minimal processing (just contrast adjustment) is best!

### 📷 Camera Controls

Control 3D viewing angles:

| Feature | Description |
|---------|-------------|
| **Preset Angles** | Top, Front, Side, Isometric views |
| **Save Position** | Save current camera to slots 1-5 |
| **Load Position** | Restore saved camera positions |
| **Auto-Rotate** | 360° rotation animation |
| **Toggle 2D/3D** | Switch viewing modes |

**Camera Presets:**
- **Top (XY)** — Looking straight down
- **Front (XZ)** — Front view
- **Side (YZ)** — Side view
- **Isometric 1-4** — Classic 3D angles from different corners
- **Tilted Top** — Almost top-down with slight tilt
- **Dramatic Low** — Low angle for dramatic effect

### 🎬 Rendering Style

Apply pre-configured rendering looks:

| Style | Effect |
|-------|--------|
| **Soft Glow** | Gentle, dreamy (low attenuation) |
| **High Contrast** | Punchy, sharp |
| **Deep Volume** | See deep into the sample |
| **Maximum Intensity** | Classic MIP, no attenuation |
| **Publication Ready** | Balanced, neutral |
| **Neon** | Bright, glowing effect |

---

## 🎨 Best Practices for PhD-Quality Images

### Recommended Workflow

For most images, minimal processing is best:

```
Raw Image → Adjust Contrast Limits → Choose Colormap → Find Good Angle → Export
```

For images that need enhancement:

```
Raw Image → Thesis Enhancement OR GPU Enhance → Adjust Colormap → Export
```

**⚠️ Don't stack enhancements** — use either GPU Enhance OR Thesis Quality, not both.

### Fluorophore Wavelengths

| Fluorophore | Wavelength (μm) | Colormap |
|-------------|-----------------|----------|
| DAPI (nuclei) | 0.46 | `blue` or `cyan` |
| GFP | 0.51 | `green` |
| **Alexa Fluor 488** | **0.52** | `green` |
| RFP/mCherry | 0.58 | `magenta` |
| Cy5/Far Red | 0.67 | `red` |

### Colormap Tips for Multi-Channel

To avoid color mixing (e.g., green + blue = cyan), use complementary pairs:

| Pair | Overlap becomes |
|------|-----------------|
| **Green + Magenta** | White (recommended!) |
| **Cyan + Red** | White |
| **Yellow + Blue** | White |

### When to Skip Processing

If your images show:
- ✅ Clear structures
- ✅ Good signal-to-noise
- ✅ Dark background

...then you probably don't need enhancement! Just adjust contrast limits and export.

---

## 📸 Exporting High-Quality Renders

### Export Hotkeys

| Hotkey | Resolution | Best for |
|--------|------------|----------|
| **S** | 3840 × 2160 (4K) | Screen presentations |
| **Shift+S** | 7680 × 4320 (8K) | Maximum detail, cropping |
| **P** | 4000 × 3000 | Print quality (~300 DPI A4) |

Files are saved with timestamps to avoid overwriting: `render_4K_20260109_143052.png`

### Export Tips

- **Use 8K (Shift+S)** for thesis figures — you can crop and still have plenty of resolution
- **Position your view first** — use Camera Controls to get the perfect angle
- **Try different colormaps** — `magma`, `viridis`, `turbo` can be more striking than standard colors
- **Black background** works best for fluorescence

---

## ⌨️ Keyboard Shortcuts

### Export & Rendering

| Key | Action |
|-----|--------|
| `S` | Save 4K render (3840×2160) |
| `Shift+S` | Save 8K render (7680×4320) |
| `P` | Save print-quality render (4000×3000) |
| `R` | Toggle auto-rotation |

### View Controls

| Key | Action |
|-----|--------|
| `2` / `3` | Switch 2D/3D view |
| `Home` | Reset camera view |
| `T` | Toggle layer visibility |
| `Space` | Play/pause (if multiple timepoints) |

### Widget Controls

| Key | Action |
|-----|--------|
| `W` | **Show ALL widgets** (recover hidden panels) |
| `F1` | Toggle GPU Deconvolution widget |
| `F2` | Toggle GPU Enhance widget |
| `F3` | Toggle Thesis Quality widget |
| `F4` | Toggle Camera Controls widget |
| `F5` | Toggle Rendering Style widget |

---

## 🔧 Troubleshooting

### "No module named 'napari'"

```powershell
# Make sure venv is activated and use explicit path
.\venv\Scripts\python.exe scripts\run_viewer.py
```

### CUDA warning appears

This is usually harmless — the GPU still works. To suppress:

```powershell
[System.Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6", "User")
# Restart terminal
```

### Deconvolution creates dotted/stippled artifacts

This happens when the PSF is smaller than your pixel size (common with 20x objectives):
- **Solution 1:** Reduce iterations to 10-15
- **Solution 2:** Skip deconvolution entirely — use GPU Enhance instead
- **Solution 3:** Just adjust contrast manually

### Deconvolution is slow

- Reduce iterations for preview (20-50)
- Ensure "Use GPU" is checked
- Check GPU is detected in terminal

### Out of GPU memory

- Process one channel at a time
- Crop to region of interest first
- Close other GPU applications

### Widget disappeared

Press **W** to show all widgets, or use F1-F5 to toggle individual widgets.

---

## 📊 Understanding the Output

After deconvolution (with appropriate settings), compare:

| Metric | Before | After (good deconv) |
|--------|--------|---------------------|
| **Sharpness** | Blurry edges | Crisp boundaries |
| **Resolution** | ~250nm lateral | ~150-180nm effective |
| **SNR** | Lower | Higher |
| **Background** | Diffuse haze | Clean, dark |

---

## 💡 Pro Tips

1. **Less is more** — good images often need only contrast adjustment
2. **Save raw data separately** — processing is destructive
3. **Try different colormaps** — `magma` and `viridis` are very popular for publications
4. **Use camera presets** — find a compelling 3D angle
5. **Export at 8K** — gives you room to crop for thesis figures
6. **Document your parameters** — for reproducibility
7. **Compare settings** — run same image with different parameters to see what works

---

## 📚 References

- [napari documentation](https://napari.org/stable/)
- [RedLionfish paper](https://github.com/rosalindfranklininstitute/RedLionfish)
- [pyclesperanto documentation](https://github.com/clEsperanto/pyclesperanto_prototype)
- [Richardson-Lucy deconvolution theory](https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution)

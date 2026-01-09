import napari
import numpy as np
import tkinter as tk
from tkinter import filedialog
from aicsimageio import AICSImage
from magicgui import magicgui
from napari.types import ImageData
import RedLionfishDeconv as rl
import pyclesperanto_prototype as cle

# Initialize GPU
print(f"🚀 GPU Detected: {cle.get_device()}")

# Global metadata storage (for widget access)
_image_metadata = {"dx": 0.1, "dz": 0.3}

def select_file():
    """Opens a Windows dialog to select the .czi file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Confocal .CZI File",
        filetypes=[("Zeiss CZI", "*.czi"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path


def generate_psf(shape, dx, dz, wavelength=0.5, na=1.4):
    """
    Generate a simple 3D Gaussian PSF approximation.
    For better results, use a measured PSF or PSF generator like 'psf' package.
    
    Parameters:
        shape: (Z, Y, X) shape of the PSF
        dx: XY pixel size in microns
        dz: Z step size in microns  
        wavelength: emission wavelength in microns (default 0.5 = 500nm)
        na: numerical aperture (default 1.4)
    """
    # Theoretical resolution limits (in microns)
    fwhm_xy = 0.51 * wavelength / na  # Rayleigh criterion
    fwhm_z = 0.88 * wavelength / (na ** 2)  # Axial resolution
    
    # Convert to sigma in pixels (FWHM = 2.355 * sigma)
    sigma_xy = (fwhm_xy / 2.355) / dx
    sigma_z = (fwhm_z / 2.355) / dz
    
    # Ensure minimum sigma of 1 pixel for numerical stability
    sigma_xy = max(sigma_xy, 1.0)
    sigma_z = max(sigma_z, 1.0)
    
    print(f"   PSF sigma: XY={sigma_xy:.2f}px, Z={sigma_z:.2f}px")
    
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    z = z - shape[0] // 2
    y = y - shape[1] // 2
    x = x - shape[2] // 2
    
    psf = np.exp(-(x**2 + y**2) / (2 * sigma_xy**2) - z**2 / (2 * sigma_z**2))
    
    # Ensure no zeros and normalize
    psf = np.clip(psf, 1e-10, None)
    psf = psf / psf.sum()
    
    return psf.astype(np.float32)


def gpu_deconvolve(image: np.ndarray, psf: np.ndarray, iterations: int = 50, 
                   use_gpu: bool = True) -> np.ndarray:
    """
    Richardson-Lucy deconvolution using RedLionfish (GPU-accelerated).
    
    Parameters:
        image: 3D numpy array (ZYX)
        psf: 3D PSF array
        iterations: number of RL iterations (more = sharper, but slower)
        use_gpu: use OpenCL GPU acceleration
    """
    # Ensure float32 for GPU processing
    image = image.astype(np.float32)
    psf = psf.astype(np.float32)
    
    # Store original range for rescaling
    orig_min = float(image.min())
    orig_max = float(image.max())
    
    # Normalize to 0-1 range (avoid division by zero)
    if orig_max > orig_min:
        image = (image - orig_min) / (orig_max - orig_min)
    else:
        print("⚠️ Warning: Image has no dynamic range!")
        return image
    
    # Add small offset to avoid zeros (RL needs positive values)
    image = np.clip(image, 1e-6, 1.0)
    
    print(f"🔬 Starting deconvolution: {iterations} iterations, GPU={use_gpu}")
    print(f"   Image shape: {image.shape}, range: [{image.min():.4f}, {image.max():.4f}]")
    print(f"   PSF shape: {psf.shape}, sum: {psf.sum():.4f}")
    
    try:
        result = rl.doRLDeconvolutionFromNpArrays(
            image, 
            psf,
            niter=iterations,
            method='gpu' if use_gpu else 'cpu',
            resAsUint8=False
        )
        
        # Check for NaN values
        if np.isnan(result).any():
            print("⚠️ Warning: GPU deconvolution produced NaN values, trying CPU...")
            result = rl.doRLDeconvolutionFromNpArrays(
                image, 
                psf,
                niter=iterations,
                method='cpu',
                resAsUint8=False
            )
        
        # If still NaN, try scipy fallback
        if np.isnan(result).any():
            print("⚠️ RedLionfish failed, trying scipy Richardson-Lucy...")
            result = _scipy_richardson_lucy(image, psf, iterations)
        
        # If still NaN, return original
        if np.isnan(result).any():
            print("❌ All deconvolution methods failed, returning original")
            return image * (orig_max - orig_min) + orig_min
            
        # Rescale back to original range
        result = result * (orig_max - orig_min) + orig_min
        
        print("✅ Deconvolution complete!")
        return result
        
    except Exception as e:
        print(f"❌ Deconvolution error: {e}")
        print("   Trying scipy fallback...")
        try:
            result = _scipy_richardson_lucy(image, psf, iterations)
            result = result * (orig_max - orig_min) + orig_min
            print("✅ Deconvolution complete (scipy)!")
            return result
        except Exception as e2:
            print(f"❌ Scipy also failed: {e2}")
            return image * (orig_max - orig_min) + orig_min


def _scipy_richardson_lucy(image, psf, iterations):
    """Fallback Richardson-Lucy using scipy (CPU only but robust)."""
    from scipy.signal import fftconvolve
    
    # Ensure positive values
    image = np.clip(image, 1e-10, None)
    psf = np.clip(psf, 1e-10, None)
    psf = psf / psf.sum()
    
    # Flip PSF for correlation
    psf_mirror = psf[::-1, ::-1, ::-1]
    
    # Richardson-Lucy iteration
    estimate = image.copy()
    for i in range(iterations):
        # Convolve estimate with PSF
        conv = fftconvolve(estimate, psf, mode='same')
        conv = np.clip(conv, 1e-10, None)
        
        # Compute ratio
        ratio = image / conv
        
        # Update estimate
        estimate = estimate * fftconvolve(ratio, psf_mirror, mode='same')
        estimate = np.clip(estimate, 1e-10, None)
        
        if (i + 1) % 10 == 0:
            print(f"   Scipy RL iteration {i + 1}/{iterations}")
    
    return estimate

def launch_phd_pipeline():
    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting.")
        return

    # 1. Load with AICSImage (Preserves Zeiss Metadata)
    img = AICSImage(file_path)
    
    # 2. Critical Metadata Extraction
    dx = img.physical_pixel_sizes.X
    dy = img.physical_pixel_sizes.Y
    dz = img.physical_pixel_sizes.Z
    z_aspect = dz / dx 
    
    print(f"--- Metadata for PhD Report ---")
    print(f"Voxel Size: {dx:.4f} x {dy:.4f} x {dz:.4f} μm")
    print(f"Z-Anisotropy Ratio: {z_aspect:.2f}")

    # 3. Initialize High-Performance Viewer
    viewer = napari.Viewer(ndisplay=3, title=f"3D Cell Viewer | {file_path}")

    # Store metadata for deconvolution widget (global)
    _image_metadata["dx"] = dx
    _image_metadata["dz"] = dz
    
    # 4. Add Channels
    colors = ['green', 'blue', 'red', 'magenta']
    for i in range(img.dims.C):
        data = img.get_image_data("ZYX", C=i)
        clim = np.percentile(data, [1, 99.8])
        
        layer = viewer.add_image(
            data,
            name=f"Ch{i}: {img.channel_names[i]}",
            scale=(z_aspect, 1, 1),
            colormap=colors[i % len(colors)],
            blending='additive',
            contrast_limits=clim,
            rendering='attenuated_mip', 
            attenuation=0.04
        )
        layer.interpolation3d = 'linear'
        # Disable bounding box to avoid OpenGL errors on some systems
        # layer.bounding_box.visible = True

    # 5. GPU Deconvolution Widget
    @magicgui(
        call_button="🚀 Run GPU Deconvolution",
        iterations={"widget_type": "SpinBox", "min": 5, "max": 500, "value": 50},
        wavelength={"widget_type": "FloatSpinBox", "min": 0.3, "max": 0.8, "value": 0.5, "step": 0.01},
        na={"widget_type": "FloatSpinBox", "min": 0.5, "max": 1.5, "value": 1.4, "step": 0.1},
        use_gpu={"value": True},
    )
    def deconvolve_widget(
        image_layer: "napari.layers.Image",
        iterations: int = 50,
        wavelength: float = 0.5,
        na: float = 1.4,
        use_gpu: bool = True,
    ) -> ImageData:
        """Apply Richardson-Lucy deconvolution to selected layer."""
        data = image_layer.data.astype(np.float32)
        
        # Generate PSF based on image metadata
        psf_shape = (min(31, data.shape[0]), 31, 31)  # Odd dimensions
        psf = generate_psf(psf_shape, _image_metadata["dx"], _image_metadata["dz"], wavelength, na)
        
        # Run deconvolution
        result = gpu_deconvolve(data, psf, iterations, use_gpu)
        
        # Calculate contrast limits safely (handle NaN and constant values)
        valid_data = result[~np.isnan(result)]
        if len(valid_data) > 0:
            cmin = float(np.percentile(valid_data, 1))
            cmax = float(np.percentile(valid_data, 99.8))
            # Ensure min < max (napari requires strictly increasing)
            if cmax <= cmin:
                cmax = cmin + 1.0
            clim = [cmin, cmax]
        else:
            clim = [0, 1]
        
        # Add result as new layer
        viewer.add_image(
            result,
            name=f"{image_layer.name} [Deconv x{iterations}]",
            scale=image_layer.scale,
            colormap=image_layer.colormap,
            blending='additive',
            contrast_limits=clim,
            rendering='attenuated_mip',
            attenuation=0.04
        )
        print(f"✨ Added deconvolved layer: {image_layer.name} [Deconv x{iterations}]")
    
    # 6. GPU Enhancement Widget (pyclesperanto)
    @magicgui(
        call_button="⚡ GPU Enhance",
        sigma={"widget_type": "FloatSpinBox", "min": 0.5, "max": 10.0, "value": 2.0, "step": 0.5},
    )
    def gpu_enhance_widget(
        image_layer: "napari.layers.Image",
        sigma: float = 2.0,
        denoise: bool = True,
        sharpen: bool = True,
    ) -> ImageData:
        """GPU-accelerated image enhancement using pyclesperanto."""
        data = image_layer.data.astype(np.float32)
        result = cle.push(data)
        
        if denoise:
            # GPU median filter for denoising
            result = cle.median_sphere(result, radius_x=1, radius_y=1, radius_z=1)
            print("  ✓ Applied GPU median filter")
        
        if sharpen:
            # Unsharp mask via difference of Gaussians
            blurred = cle.gaussian_blur(result, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma/2)
            result = cle.subtract_images(result, blurred)
            result = cle.add_images(cle.push(data), result)  # Add back original
            print("  ✓ Applied GPU unsharp mask")
        
        result = cle.pull(result)
        
        # Calculate contrast limits safely (handle NaN and constant values)
        valid_data = result[~np.isnan(result)]
        if len(valid_data) > 0:
            cmin = float(np.percentile(valid_data, 1))
            cmax = float(np.percentile(valid_data, 99.8))
            # Ensure min < max (napari requires strictly increasing)
            if cmax <= cmin:
                cmax = cmin + 1.0
            clim = [cmin, cmax]
        else:
            clim = [0, 1]
        
        viewer.add_image(
            result,
            name=f"{image_layer.name} [Enhanced]",
            scale=image_layer.scale,
            colormap=image_layer.colormap,
            blending='additive',
            contrast_limits=clim,
            rendering='attenuated_mip',
            attenuation=0.04
        )
        print(f"✨ Added enhanced layer: {image_layer.name} [Enhanced]")
    
    # 7. THESIS-QUALITY Enhancement Widget
    @magicgui(
        call_button="🎨 Thesis-Quality Enhancement",
        background_radius={"widget_type": "SpinBox", "min": 10, "max": 200, "value": 50, 
                          "tooltip": "Rolling ball radius for background subtraction (larger = removes more haze)"},
        local_contrast={"widget_type": "FloatSpinBox", "min": 1.0, "max": 5.0, "value": 2.0, "step": 0.5,
                       "tooltip": "Local contrast boost factor (higher = more dramatic)"},
        gamma={"widget_type": "FloatSpinBox", "min": 0.3, "max": 2.0, "value": 0.8, "step": 0.1,
              "tooltip": "Gamma correction (<1 brightens dim features, >1 darkens)"},
        percentile_low={"widget_type": "FloatSpinBox", "min": 0.0, "max": 10.0, "value": 0.5, "step": 0.5,
                       "tooltip": "Lower percentile for contrast stretch"},
        percentile_high={"widget_type": "FloatSpinBox", "min": 90.0, "max": 100.0, "value": 99.9, "step": 0.1,
                        "tooltip": "Upper percentile for contrast stretch"},
    )
    def thesis_enhance_widget(
        image_layer: "napari.layers.Image",
        background_subtract: bool = True,
        background_radius: int = 50,
        boost_local_contrast: bool = True,
        local_contrast: float = 2.0,
        apply_gamma: bool = True,
        gamma: float = 0.8,
        percentile_low: float = 0.5,
        percentile_high: float = 99.9,
    ) -> ImageData:
        """
        Advanced enhancement for thesis-quality images.
        Combines background subtraction, local contrast enhancement, and gamma correction.
        """
        from scipy import ndimage
        from skimage import exposure
        
        data = image_layer.data.astype(np.float32)
        result = data.copy()
        
        print("🎨 Starting thesis-quality enhancement...")
        
        # 1. Background subtraction (rolling ball approximation)
        if background_subtract:
            print(f"  → Subtracting background (radius={background_radius})...")
            # Use GPU for large gaussian as background estimate
            gpu_data = cle.push(result)
            # Large gaussian blur approximates rolling ball background
            background = cle.gaussian_blur(gpu_data, 
                                           sigma_x=background_radius, 
                                           sigma_y=background_radius, 
                                           sigma_z=background_radius/4)
            # Subtract background, keep positive values
            result = cle.pull(cle.subtract_images(gpu_data, background))
            result = np.clip(result, 0, None)
            print("  ✓ Background removed - structures should pop more")
        
        # 2. Local contrast enhancement (CLAHE-like effect)
        if boost_local_contrast:
            print(f"  → Boosting local contrast (factor={local_contrast})...")
            # Process slice by slice for 3D CLAHE-like effect
            for z in range(result.shape[0]):
                slice_data = result[z]
                # Normalize slice
                slice_min, slice_max = slice_data.min(), slice_data.max()
                if slice_max > slice_min:
                    slice_norm = (slice_data - slice_min) / (slice_max - slice_min)
                    # Apply adaptive histogram equalization
                    slice_enhanced = exposure.equalize_adapthist(
                        slice_norm, 
                        kernel_size=64,  # Local region size
                        clip_limit=0.03 * local_contrast  # Controls enhancement strength
                    )
                    result[z] = slice_enhanced * (slice_max - slice_min) + slice_min
            print("  ✓ Local contrast boosted - fine details enhanced")
        
        # 3. Gamma correction
        if apply_gamma:
            print(f"  → Applying gamma correction ({gamma})...")
            # Normalize to 0-1 for gamma
            r_min, r_max = result.min(), result.max()
            if r_max > r_min:
                result = (result - r_min) / (r_max - r_min)
                result = np.power(result, gamma)
                result = result * (r_max - r_min) + r_min
            print("  ✓ Gamma applied - tonal range adjusted")
        
        # 4. Final contrast stretch with custom percentiles
        print(f"  → Stretching contrast ({percentile_low}% - {percentile_high}%)...")
        p_low = np.percentile(result, percentile_low)
        p_high = np.percentile(result, percentile_high)
        result = np.clip(result, p_low, p_high)
        result = (result - p_low) / (p_high - p_low) * 65535  # Scale to 16-bit range
        
        # Calculate display limits
        valid_data = result[~np.isnan(result)]
        if len(valid_data) > 0:
            cmin = float(np.percentile(valid_data, 0.1))
            cmax = float(np.percentile(valid_data, 99.95))
            if cmax <= cmin:
                cmax = cmin + 1.0
            clim = [cmin, cmax]
        else:
            clim = [0, 65535]
        
        viewer.add_image(
            result,
            name=f"{image_layer.name} [Thesis]",
            scale=image_layer.scale,
            colormap=image_layer.colormap,
            blending='additive',
            contrast_limits=clim,
            rendering='attenuated_mip',
            attenuation=0.04
        )
        print(f"✨ Added thesis-quality layer: {image_layer.name} [Thesis]")
        print("   💡 Tip: Try different colormaps like 'magma', 'viridis', or 'turbo' for extra visual impact!")
    
    # 8. Camera & View Controls Widget
    # Storage for saved camera positions
    _saved_cameras = {}
    
    @magicgui(
        call_button=False,
        auto_call=False,
        preset={"widget_type": "ComboBox", 
                "choices": ["Top (XY)", "Front (XZ)", "Side (YZ)", 
                           "Isometric 1", "Isometric 2", "Isometric 3", "Isometric 4",
                           "Tilted Top", "Dramatic Low"],
                "value": "Isometric 1"},
        slot_name={"widget_type": "ComboBox",
                   "choices": ["Slot 1", "Slot 2", "Slot 3", "Slot 4", "Slot 5"],
                   "value": "Slot 1"},
    )
    def camera_widget(
        preset: str = "Isometric 1",
        slot_name: str = "Slot 1",
    ):
        """Camera and view controls for 3D visualization."""
        pass  # Buttons handle the actions
    
    # Define preset camera angles (rotation around X, Y, Z in degrees)
    CAMERA_PRESETS = {
        "Top (XY)": (0, 0, 90),           # Looking down from above
        "Front (XZ)": (0, 0, 0),          # Looking from front
        "Side (YZ)": (0, 90, 0),          # Looking from side
        "Isometric 1": (-30, 30, -45),    # Classic isometric view
        "Isometric 2": (-30, -30, 45),    # Opposite corner
        "Isometric 3": (-45, 45, -30),    # Higher angle
        "Isometric 4": (-20, 20, -60),    # Lower, more dramatic
        "Tilted Top": (-15, 15, -80),     # Almost top but slightly tilted
        "Dramatic Low": (-60, 30, -30),   # Low dramatic angle
    }
    
    def apply_preset():
        """Apply the selected camera preset."""
        preset_name = camera_widget.preset.value
        if preset_name in CAMERA_PRESETS:
            angles = CAMERA_PRESETS[preset_name]
            viewer.camera.angles = angles
            print(f"📷 Camera set to: {preset_name} {angles}")
    
    def save_camera():
        """Save current camera position to selected slot."""
        slot = camera_widget.slot_name.value
        _saved_cameras[slot] = {
            'angles': tuple(viewer.camera.angles),
            'zoom': viewer.camera.zoom,
            'center': tuple(viewer.camera.center),
        }
        print(f"💾 Camera saved to {slot}: angles={viewer.camera.angles}, zoom={viewer.camera.zoom:.2f}")
    
    def load_camera():
        """Load camera position from selected slot."""
        slot = camera_widget.slot_name.value
        if slot in _saved_cameras:
            cam = _saved_cameras[slot]
            viewer.camera.angles = cam['angles']
            viewer.camera.zoom = cam['zoom']
            viewer.camera.center = cam['center']
            print(f"📷 Camera loaded from {slot}")
        else:
            print(f"⚠️ No camera saved in {slot}")
    
    def reset_view():
        """Reset camera to default view."""
        viewer.reset_view()
        if viewer.dims.ndisplay == 3:
            viewer.camera.angles = (-30, 30, -45)  # Nice isometric default
        print("🔄 View reset")
    
    def toggle_2d_3d():
        """Toggle between 2D and 3D view."""
        if viewer.dims.ndisplay == 2:
            viewer.dims.ndisplay = 3
            viewer.camera.angles = (-30, 30, -45)
            print("🎲 Switched to 3D view")
        else:
            viewer.dims.ndisplay = 2
            print("📋 Switched to 2D view")
    
    # Auto-rotation using Qt timer (runs on main thread)
    from qtpy.QtCore import QTimer
    
    rotation_timer = QTimer()
    rotation_state = {"active": False, "start_angles": None, "step": 0}
    
    def rotation_step():
        """Single rotation step - called by timer."""
        if not rotation_state["active"]:
            rotation_timer.stop()
            return
        
        if rotation_state["step"] >= 360:
            rotation_state["step"] = 0  # Loop continuously
        
        start = rotation_state["start_angles"]
        new_angles = (start[0], start[1], start[2] + rotation_state["step"])
        viewer.camera.angles = new_angles
        rotation_state["step"] += 2  # Speed: 2 degrees per frame
    
    rotation_timer.timeout.connect(rotation_step)
    
    def start_rotation():
        """Start auto-rotation animation (useful for presentations)."""
        if rotation_state["active"]:
            return  # Already rotating
        
        rotation_state["active"] = True
        rotation_state["start_angles"] = tuple(viewer.camera.angles)
        rotation_state["step"] = 0
        rotation_timer.start(50)  # 50ms = 20 FPS
        print("🔄 Auto-rotation started (press 'R' to stop)")
    
    def stop_rotation():
        """Stop auto-rotation."""
        rotation_state["active"] = False
        rotation_timer.stop()
        print("🔄 Auto-rotation stopped")
    
    # Add buttons to the widget
    from magicgui.widgets import PushButton, Container
    
    btn_apply = PushButton(text="📷 Apply Preset")
    btn_apply.clicked.connect(apply_preset)
    
    btn_save = PushButton(text="💾 Save Position")
    btn_save.clicked.connect(save_camera)
    
    btn_load = PushButton(text="📂 Load Position")
    btn_load.clicked.connect(load_camera)
    
    btn_reset = PushButton(text="🔄 Reset View")
    btn_reset.clicked.connect(reset_view)
    
    btn_toggle = PushButton(text="🎲 Toggle 2D/3D")
    btn_toggle.clicked.connect(toggle_2d_3d)
    
    def toggle_rotation_button():
        """Toggle rotation from button click."""
        if rotation_state["active"]:
            stop_rotation()
            btn_rotate.text = "🎬 Auto-Rotate"
        else:
            start_rotation()
            btn_rotate.text = "⏹️ Stop Rotation"
    
    btn_rotate = PushButton(text="🎬 Auto-Rotate")
    btn_rotate.clicked.connect(toggle_rotation_button)
    
    # Create container with all controls
    camera_container = Container(widgets=[
        camera_widget.preset,
        btn_apply,
        camera_widget.slot_name,
        btn_save,
        btn_load,
        btn_reset,
        btn_toggle,
        btn_rotate,
    ])
    
    # 9. Rendering Presets Widget
    @magicgui(
        call_button="🎨 Apply Rendering Style",
        style={"widget_type": "ComboBox",
               "choices": ["Soft Glow", "High Contrast", "Deep Volume", 
                          "Maximum Intensity", "Publication Ready", "Neon"],
               "value": "Soft Glow"},
    )
    def rendering_widget(
        image_layer: "napari.layers.Image",
        style: str = "Soft Glow",
    ):
        """Apply rendering presets to selected layer."""
        
        styles = {
            "Soft Glow": {
                "rendering": "attenuated_mip",
                "attenuation": 0.03,
                "interpolation3d": "linear",
                "gamma": 0.9,
            },
            "High Contrast": {
                "rendering": "attenuated_mip", 
                "attenuation": 0.06,
                "interpolation3d": "linear",
                "gamma": 1.2,
            },
            "Deep Volume": {
                "rendering": "attenuated_mip",
                "attenuation": 0.015,
                "interpolation3d": "linear",
                "gamma": 0.7,
            },
            "Maximum Intensity": {
                "rendering": "mip",
                "attenuation": 0.0,
                "interpolation3d": "linear",
                "gamma": 1.0,
            },
            "Publication Ready": {
                "rendering": "attenuated_mip",
                "attenuation": 0.04,
                "interpolation3d": "linear",
                "gamma": 1.0,
            },
            "Neon": {
                "rendering": "attenuated_mip",
                "attenuation": 0.02,
                "interpolation3d": "linear",
                "gamma": 0.6,
            },
        }
        
        if style in styles:
            s = styles[style]
            image_layer.rendering = s["rendering"]
            image_layer.attenuation = s["attenuation"]
            image_layer.interpolation3d = s["interpolation3d"]
            image_layer.gamma = s["gamma"]
            print(f"🎨 Applied '{style}' rendering: attenuation={s['attenuation']}, gamma={s['gamma']}")
    
    # Keyboard shortcut to toggle rotation
    @viewer.bind_key('r')
    def toggle_rotation(viewer):
        if rotation_state["active"]:
            stop_rotation()
        else:
            start_rotation()
    
    # Add widgets to viewer and store references for show/hide
    dock_widgets = {}
    dock_widgets['deconv'] = viewer.window.add_dock_widget(deconvolve_widget, name="🔬 GPU Deconvolution")
    dock_widgets['enhance'] = viewer.window.add_dock_widget(gpu_enhance_widget, name="⚡ GPU Enhance")
    dock_widgets['thesis'] = viewer.window.add_dock_widget(thesis_enhance_widget, name="🎨 Thesis Quality")
    dock_widgets['camera'] = viewer.window.add_dock_widget(camera_container, name="📷 Camera Controls")
    dock_widgets['render'] = viewer.window.add_dock_widget(rendering_widget, name="🎬 Rendering Style")
    
    # Function to show all widgets
    def show_all_widgets():
        """Show all dock widgets (handles deleted widgets gracefully)."""
        restored = 0
        deleted = []
        
        for name, dock in dock_widgets.items():
            try:
                # Try to show - will raise RuntimeError if deleted
                dock.show()
                restored += 1
            except RuntimeError:
                # Widget was deleted/closed permanently
                deleted.append(name)
            except Exception as e:
                deleted.append(name)
        
        if restored > 0:
            print(f"📋 Restored {restored} widget(s)!")
        
        if deleted:
            print(f"⚠️ These widgets were deleted and need a restart: {', '.join(deleted)}")
    
    # Keyboard shortcut to show all widgets
    @viewer.bind_key('w')
    def show_widgets_hotkey(viewer):
        show_all_widgets()
    
    # Keyboard shortcuts to toggle individual widgets
    def safe_toggle_widget(name, label):
        """Safely toggle a widget, handling deleted widgets."""
        try:
            dock = dock_widgets[name]
            dock.setVisible(not dock.isVisible())
            print(f"{label}: {'shown' if dock.isVisible() else 'hidden'}")
        except RuntimeError:
            print(f"⚠️ {label} was deleted. Restart viewer to restore.")
    
    @viewer.bind_key('F1')
    def toggle_deconv(viewer):
        safe_toggle_widget('deconv', '🔬 GPU Deconvolution')
    
    @viewer.bind_key('F2')
    def toggle_enhance(viewer):
        safe_toggle_widget('enhance', '⚡ GPU Enhance')
    
    @viewer.bind_key('F3')
    def toggle_thesis(viewer):
        safe_toggle_widget('thesis', '🎨 Thesis Quality')
    
    @viewer.bind_key('F4')
    def toggle_camera(viewer):
        safe_toggle_widget('camera', '📷 Camera Controls')
    
    @viewer.bind_key('F5')
    def toggle_render(viewer):
        safe_toggle_widget('render', '🎬 Rendering Style')

    # 10. The "PhD Report" High-Res Export Hotkeys
    # Get the script directory (repo root) and create output folder
    import os
    from datetime import datetime
    
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Renders will be saved to: {output_dir}")
    
    @viewer.bind_key('s')
    def save_4k_render(viewer):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"render_4K_{timestamp}.png")
        viewer.screenshot(out_path, canvas_only=True, size=(3840, 2160))
        print(f"📸 4K Render (3840×2160) saved to: {out_path}")
    
    @viewer.bind_key('Shift-s')
    def save_8k_render(viewer):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"render_8K_{timestamp}.png")
        viewer.screenshot(out_path, canvas_only=True, size=(7680, 4320))
        print(f"📸 8K Render (7680×4320) saved to: {out_path}")
    
    @viewer.bind_key('p')
    def save_print_render(viewer):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 300 DPI at A4 size ≈ 3508 × 2480 pixels
        out_path = os.path.join(output_dir, f"render_PRINT_{timestamp}.png")
        viewer.screenshot(out_path, canvas_only=True, size=(4000, 3000))
        print(f"📸 Print-Quality Render (4000×3000, ~300DPI A4) saved to: {out_path}")

    # Scale bar can cause OpenGL errors on some systems
    # Uncomment if your system handles it well:
    # viewer.scale_bar.visible = True
    # viewer.scale_bar.unit = "um"
    
    print("\n📋 Keyboard Shortcuts:")
    print("  [S]       - Save 4K render (3840×2160)")
    print("  [Shift+S] - Save 8K render (7680×4320) - for maximum detail")
    print("  [P]       - Save print-quality render (4000×3000, ~300DPI)")
    print("  [R]       - Toggle auto-rotation (for presentations)")
    print("  [2] / [3] - Switch between 2D/3D view")
    print("  [Home]    - Reset camera view")
    print("")
    print("  [W]       - Show ALL widgets (recover hidden panels)")
    print("  [F1]      - Toggle GPU Deconvolution widget")
    print("  [F2]      - Toggle GPU Enhance widget")
    print("  [F3]      - Toggle Thesis Quality widget")
    print("  [F4]      - Toggle Camera Controls widget")
    print("  [F5]      - Toggle Rendering Style widget")
    print("\n🎛️  Use the dock widgets on the right for GPU processing!")

    napari.run()

if __name__ == "__main__":
    launch_phd_pipeline()
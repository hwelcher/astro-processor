#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Subframe Selector UI (FITS)

Key points in this version:
- Preview fix: render from a contiguous NumPy array as 24-bit RGB (no palette), set
  SmoothPixmapTransform on the view and SmoothTransformation on the pixmap item,
  and use FullViewportUpdate + DeviceCoordinateCache to avoid moiré/checkerboard.
- Histogram dialog y-axis now reads "Pixels per bin".
- All previous features retained: details table (sortable), preview window with
  zoom-to-fit/100%/histogram, robust sigma bands, per-metric bin-edge caching,
  stable right histogram, filters enabled by default (but inclusive), export kept
  with progress, cancel-safe processing, tooltips, pixel-scale helper, etc.

Dependencies:
  pip install PySide6 numpy scipy astropy matplotlib
  # Optional: faster morphology/labeling if available:
  pip install opencv-python
"""

import sys
import os
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
import numpy.random as random

# Optional fast paths
try:
    import cv2
    HAS_CV2 = True
    try:
        HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        HAS_CUDA = False
except Exception:
    HAS_CV2 = False
    HAS_CUDA = False

# Optional CuPy (GPU) path for blur/morphology
try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.ndimage import gaussian_filter as cpx_gaussian_filter  # type: ignore
    from cupyx.scipy.ndimage import binary_dilation as cpx_binary_dilation  # type: ignore
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False

from scipy.ndimage import gaussian_filter as ndi_gaussian_filter
from scipy.ndimage import binary_dilation as ndi_binary_dilation
from scipy.ndimage import label as ndi_label
from scipy.ndimage import find_objects as ndi_find_objects

from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject, QSize, QEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFormLayout, QSpinBox, QDoubleSpinBox,
    QGroupBox, QProgressBar, QMessageBox, QCheckBox, QComboBox, QAbstractItemView,
    QTextEdit, QDialog, QDialogButtonBox, QLineEdit, QSplitter, QGridLayout, QSlider,
    QProgressDialog, QTableWidget, QTableWidgetItem, QSizePolicy, QGraphicsView,
    QGraphicsScene, QToolBar
)
from PySide6.QtGui import (
    QPalette, QColor, QCloseEvent, QBrush, QImage, QPixmap, QWheelEvent, QPainter, QAction
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

FWHM_FACTOR = 2.354820045  # 2 * sqrt(2 * ln(2))


# =========================== Analysis / Metrics ===========================

def load_fits(
    path: str,
    hdu_index: Optional[int] = None,
    plane_index: Optional[int] = None
) -> Tuple[np.ndarray, Any]:
    """Load FITS -> (2D float32 array, FITS header). (memmap=False for BSCALE/BZERO/BLANK)"""
    with fits.open(path, memmap=False) as hdul:
        if hdu_index is not None:
            if hdu_index < 0 or hdu_index >= len(hdul):
                raise ValueError(f"HDU index {hdu_index} out of range for {path}")
            hdu = hdul[hdu_index]
        else:
            hdu = None
            for h in hdul:
                if getattr(h, "data", None) is not None:
                    hdu = h
                    break
        if hdu is None or hdu.data is None:
            raise ValueError(f"No image HDU found in {path}")

        header = hdu.header
        arr = np.array(hdu.data, dtype=np.float32)

    if arr.ndim == 3:
        idx = 0 if plane_index is None else plane_index
        if idx < 0 or idx >= arr.shape[0]:
            raise ValueError(f"plane_index {idx} out of range (0..{arr.shape[0]-1}) for {path}")
        arr = arr[idx, :, :]

    if arr.ndim != 2:
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Unsupported FITS dimensionality {arr.shape} in {path}")

    med = float(np.nanmedian(arr))
    arr = np.nan_to_num(arr, nan=med, posinf=med, neginf=med)
    return arr, header


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = img.shape
    nh, nw = h // factor, w // factor
    img = img[:nh * factor, :nw * factor]
    return img.reshape(nh, factor, nw, factor).mean(axis=(1, 3))  # block average


def robust_bg_noise(img: np.ndarray, fast: bool = False, sample_px: int = 2_000_000) -> Tuple[float, float]:
    """Robust background and noise sigma via median/MAD.
    If fast=True and the image is large, compute on a random subset to reduce cost.
    """
    x = img
    if fast:
        n = img.size
        if n > sample_px:
            rng = np.random.default_rng(12345)
            idx = rng.choice(n, size=sample_px, replace=False)
            x = img.reshape(-1)[idx]
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = 1.4826 * mad if mad > 0 else float(np.std(img))
    return med, sigma


def _blur_gaussian(img: np.ndarray, sigma: float, use_gpu: bool = False) -> np.ndarray:
    if use_gpu and HAS_CUPY:
        g = cp.asarray(img, dtype=cp.float32)
        out = cpx_gaussian_filter(g, sigma=sigma, mode="nearest")
        return cp.asnumpy(out).astype(np.float32, copy=False)
    if HAS_CV2 and use_gpu and HAS_CUDA:
        src32 = img.astype(np.float32, copy=False)
        gsrc = cv2.cuda_GpuMat()
        gsrc.upload(src32)
        k = max(3, int(6.0 * sigma + 1) | 1)
        gf = cv2.cuda.createGaussianFilter(cv2.CV_32F, cv2.CV_32F, (k, k), sigma, sigma, borderMode=cv2.BORDER_REPLICATE)
        gdst = gf.apply(gsrc)
        out = gdst.download()
        return out
    if HAS_CV2:
        src = img.astype(np.float32, copy=False)
        return cv2.GaussianBlur(src, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    else:
        return ndi_gaussian_filter(img, sigma=sigma).astype(np.float32, copy=False)


def _dilate(binary: np.ndarray, iterations: int, use_gpu: bool = False) -> np.ndarray:
    if iterations <= 0:
        return binary
    if use_gpu and HAS_CUPY:
        b = cp.asarray(binary)
        out = cpx_binary_dilation(b, iterations=iterations)
        return cp.asnumpy(out)
    if HAS_CV2 and use_gpu and HAS_CUDA:
        kernel = np.ones((3, 3), np.uint8)
        src = (binary.astype(np.uint8) * 255)
        gsrc = cv2.cuda_GpuMat(); gsrc.upload(src)
        mf = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, gsrc.type(), kernel, iterations=iterations)
        gdst = mf.apply(gsrc)
        dst = gdst.download()
        return (dst > 0)
    if HAS_CV2:
        kernel = np.ones((3, 3), np.uint8)
        src = (binary.astype(np.uint8) * 255)
        dil = cv2.dilate(src, kernel, iterations=iterations, borderType=cv2.BORDER_CONSTANT)
        return (dil > 0)
    else:
        return ndi_binary_dilation(binary, iterations=iterations)


def _label(binary: np.ndarray):
    if HAS_CV2:
        src = binary.astype(np.uint8)
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        areas = stats[:, cv2.CC_STAT_AREA] if stats is not None else None
        return labels, (nlabels - 1), areas
    else:
        labels, n = ndi_label(binary)
        return labels, n, None


def detect_stars(
    img: np.ndarray,
    k_sigma: float = 4.5,
    min_area: int = 12,
    max_area: int = 2000,
    dilation_iter: int = 1,
    max_stars: int = 3000,
    fast_stats: bool = False,
    use_gpu: bool = False
) -> Tuple[np.ndarray, int]:
    bg, noise = robust_bg_noise(img, fast=fast_stats)
    blurred = _blur_gaussian(img, sigma=1, use_gpu=use_gpu)
    det = blurred - bg
    thresh = det > (k_sigma * noise)

    if dilation_iter > 0:
        thresh = _dilate(thresh, iterations=dilation_iter, use_gpu=use_gpu)

    labels, n, areas = _label(thresh)
    if n == 0:
        return labels, 0

    keep_mask = np.zeros((n + 1,), dtype=bool)
    keep_mask[0] = False

    if areas is not None and HAS_CV2:
        areas_arr = np.asarray(areas, dtype=np.int64)
        keep_mask[1:] = (areas_arr[1:] >= min_area) & (areas_arr[1:] <= max_area)
    else:
        areas_arr = np.bincount(labels.ravel(), minlength=n + 1)
        keep_mask[1:] = (areas_arr[1:] >= min_area) & (areas_arr[1:] <= max_area)

    keep_ids = np.where(keep_mask)[0]
    if keep_ids.size == 0:
        return np.zeros_like(labels), 0

    if keep_ids.size > max_stars:
        keep_ids = keep_ids[np.argsort(areas_arr[keep_ids])[::-1][:max_stars]]

    # Vectorized remap using a lookup table
    lut = np.zeros((int(keep_mask.size)), dtype=np.int32)
    lut[keep_ids] = np.arange(1, keep_ids.size + 1, dtype=np.int32)
    new_labels = lut[labels]

    return new_labels, int(new_labels.max())


# --- OPTIMIZED FUNCTION ---
def star_shape_metrics(
    img: np.ndarray,
    labels: np.ndarray,
    pad: int = 2,
    max_stars_to_fit: int = 500  # New parameter to limit expensive fits
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate star shape metrics using a faster Astropy 2D Gaussian fitter
    on a random subset of stars to improve performance.
    """
    fwhm_vals: List[float] = []
    hfr_vals: List[float] = []
    ecc_vals: List[float] = []
    peak_minus_bg_vals: List[float] = []

    max_id = int(labels.max())
    if max_id <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    slices = ndi_find_objects(labels)
    if not slices:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # --- Optimization 1: Select a subset of stars for fitting ---
    all_star_labels = [lb for lb, sl in enumerate(slices, 1) if sl is not None]
    if not all_star_labels:
        return np.array([]), np.array([]), np.array([]), np.array([])

    if len(all_star_labels) > max_stars_to_fit:
        # Use a fixed seed for reproducibility if needed, or remove for true randomness
        rng = random.default_rng()
        fit_star_labels = set(rng.choice(all_star_labels, size=max_stars_to_fit, replace=False))
    else:
        fit_star_labels = set(all_star_labels)

    # --- Optimization 2: Use Astropy's faster fitter ---
    # Create the fitter instance *once* outside the loop
    fitter = fitting.LevMarLSQFitter()

    # Loop through all stars to calculate inexpensive metrics (HFR, Peak)
    for lb in all_star_labels:
        sl = slices[lb - 1]
        ysl, xsl = sl
        y0e = max(0, ysl.start - pad); y1e = min(img.shape[0], ysl.stop + pad)
        x0e = max(0, xsl.start - pad); x1e = min(img.shape[1], xsl.stop + pad)

        stamp = img[y0e:y1e, x0e:x1e]
        lab_roi = labels[y0e:y1e, x0e:x1e]
        msub = (lab_roi == lb)

        if not np.any(msub):
            continue

        bg_local = np.median(stamp[~msub]) if np.any(~msub) else np.median(stamp)
        flux = stamp - bg_local
        flux[flux < 0] = 0.0

        yy, xx = np.indices(stamp.shape)
        w = flux * msub
        total = w.sum()
        if total <= 0:
            continue

        # Moment analysis is fast and needed for HFR and initial guesses
        cy = (w * yy).sum() / total
        cx = (w * xx).sum() / total
        dy, dx = yy - cy, xx - cx
        
        # --- HFR (calculated for all stars) ---
        r = np.sqrt(dx*dx + dy*dy)
        r_flat = r[msub].ravel(); f_flat = w[msub].ravel()
        order = np.argsort(r_flat)
        csum = np.cumsum(f_flat[order])
        target = 0.5 * csum[-1]
        idx = np.searchsorted(csum, target)
        if idx >= len(r_flat): idx = len(r_flat) - 1
        hfr_vals.append(float(r_flat[order][idx]))
        peak_minus_bg_vals.append(float(np.max(stamp[msub]) - bg_local))

        # --- Expensive Fitting (only for the subset) ---
        if lb in fit_star_labels:
            Iyy = (w * dy * dy).sum() / total
            Ixx = (w * dx * dx).sum() / total
            Ixy = (w * dx * dy).sum() / total

            try:
                # Initial guesses from moments
                amplitude_guess = float(np.max(stamp[msub]) - bg_local)
                cov = np.array([[Ixx, Ixy], [Ixy, Iyy]], dtype=np.float64)
                evals, evecs = np.linalg.eigh(cov)
                sig_major = np.sqrt(max(1e-9, evals[1]))
                sig_minor = np.sqrt(max(1e-9, evals[0]))
                theta_guess = np.arctan2(evecs[1, 1], evecs[0, 1])

                # Astropy model
                model_guess = models.Gaussian2D(
                    amplitude=amplitude_guess,
                    y_mean=cy, x_mean=cx,
                    y_stddev=sig_major, x_stddev=sig_minor,
                    theta=theta_guess
                )
                
                # Run the fit
                y_coords, x_coords = np.where(msub)
                z_data = stamp[y_coords, x_coords] - bg_local
                fitted_model = fitter(model_guess, x_coords, y_coords, z_data)

                # Extract params from the fitted Astropy model
                sig_y, sig_x = fitted_model.y_stddev.value, fitted_model.x_stddev.value
                sig_a = max(sig_x, sig_y)
                sig_b = min(sig_x, sig_y)

                # Eccentricity from fit
                ecc = math.sqrt(1.0 - (sig_b**2) / (sig_a**2)) if sig_a > 1e-6 else 0.0
                ecc_vals.append(ecc)

                # FWHM from fit
                fwhm_fit = FWHM_FACTOR * math.sqrt(abs(sig_a * sig_b))
                fwhm_vals.append(fwhm_fit)

            except Exception:
                # Skip star if fit fails, or add fallback moment calc if desired
                continue

    return (
        np.array(fwhm_vals, dtype=np.float64),
        np.array(hfr_vals, dtype=np.float64),
        np.array(ecc_vals, dtype=np.float64),
        np.array(peak_minus_bg_vals, dtype=np.float64),
    )


def analyze_image(
    path: str,
    downsample_factor: int,
    k_sigma: float,
    min_area: int,
    max_area: int,
    dilation_iter: int,
    max_stars: int,
    arcsec_per_pixel: Optional[float] = None,
    fast_stats: bool = False,
    use_gpu: bool = False,
) -> Dict[str, Any]:
    img, header = load_fits(path, hdu_index=None, plane_index=None)
    h, w = img.shape
    work = downsample(img, downsample_factor)

    # Extract observation date, fall back to None if not found
    date_obs = header.get("DATE-OBS")

    mean = float(np.mean(work, dtype=np.float64))
    median = float(np.median(work))
    std = float(np.std(work, dtype=np.float64))
    bg, noise = robust_bg_noise(work, fast=fast_stats)

    labels, nstars = detect_stars(
        work, k_sigma=k_sigma, min_area=min_area, max_area=max_area,
        dilation_iter=dilation_iter, max_stars=max_stars,
        fast_stats=fast_stats, use_gpu=use_gpu
    )

    fwhm_med_ds = float("nan")
    hfr_med_ds = float("nan")
    ecc_mean = float("nan")
    snr_star_median = float("nan")

    if nstars > 0:
        fwhm_arr, hfr_arr, ecc_arr, peak_minus_bg = star_shape_metrics(work, labels)
        if fwhm_arr.size:
            fwhm_med_ds = float(np.median(fwhm_arr))
        if hfr_arr.size:
            hfr_med_ds = float(np.median(hfr_arr))
        if ecc_arr.size:
            ecc_mean = float(np.mean(ecc_arr))
        if peak_minus_bg.size and noise > 0:
            snr_star_median = float(np.median(peak_minus_bg / noise))

    fwhm_med_orig = fwhm_med_ds * downsample_factor if np.isfinite(fwhm_med_ds) else float("nan")
    hfr_med_orig  = hfr_med_ds  * downsample_factor if np.isfinite(hfr_med_ds)  else float("nan")

    fwhm_arcsec = hfr_arcsec = None
    if arcsec_per_pixel is not None and np.isfinite(fwhm_med_orig):
        fwhm_arcsec = fwhm_med_orig * arcsec_per_pixel
        if np.isfinite(hfr_med_orig):
            hfr_arcsec = hfr_med_orig * arcsec_per_pixel

    res: Dict[str, Any] = {
        "path": path,
        "date_obs": date_obs,
        "width": int(w),
        "height": int(h),
        "downsample": int(downsample_factor),
        "mean": mean,
        "median": median,
        "stddev": std,
        "background": float(bg),
        "noise_sigma": float(noise),
        "stars": int(nstars),
        "fwhm_px": fwhm_med_orig,
        "hfr_px":  hfr_med_orig,
        "fwhm_px_downsampled": fwhm_med_ds if downsample_factor > 1 else None,
        "hfr_px_downsampled":  hfr_med_ds  if downsample_factor > 1 else None,
        "eccentricity": ecc_mean,
        "snr_star_median": snr_star_median,
    }
    if fwhm_arcsec is not None:
        res["fwhm_arcsec"] = float(fwhm_arcsec)
    if hfr_arcsec is not None:
        res["hfr_arcsec"] = float(hfr_arcsec)
    return res


# =========================== Worker (parallel + cancel) ===========================

class StatsWorker(QThread):
    progressed = Signal(int, int)      # processed, total
    finished_ok = Signal(list)         # list[Dict[str, Any]]
    failed = Signal(str)
    cancelled = Signal(list)           # partial results

    def __init__(self, files: List[str], params: Dict[str, Any], workers: int, use_processes: bool, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.files = files
        self.params = params
        self.workers = max(1, int(workers))
        self.use_processes = bool(use_processes)
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def _submit(self, ex, path):
        return ex.submit(
            analyze_image,
            path,
            self.params["downsample"],
            self.params["k_sigma"],
            self.params["min_area"],
            self.params["max_area"],
            self.params["dilation"],
            self.params["max_stars"],
            self.params["arcsec_per_pixel"],
        )

    def run(self):
        try:
            results: List[Dict[str, Any]] = []
            total = len(self.files)
            if total == 0:
                self.finished_ok.emit([])
                return

            Executor = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

            inflight = {}
            next_idx = 0
            done_count = 0

            with Executor(max_workers=self.workers) as ex:
                while next_idx < total and len(inflight) < self.workers:
                    fut = self._submit(ex, self.files[next_idx])
                    inflight[fut] = next_idx
                    next_idx += 1

                while inflight:
                    if self._cancel:
                        for fut in inflight.keys():
                            fut.cancel()
                        self.cancelled.emit(results)
                        return

                    for fut in list(as_completed(inflight.keys(), timeout=None)):
                        inflight.pop(fut, None)
                        if fut.cancelled():
                            self.cancelled.emit(results)
                            return
                        res = fut.result()
                        results.append(res)
                        done_count += 1
                        self.progressed.emit(done_count, total)

                        if next_idx < total and not self._cancel:
                            nfut = self._submit(ex, self.files[next_idx])
                            inflight[nfut] = next_idx
                            next_idx += 1
                        break

                self.finished_ok.emit(results)

        except Exception as e:
            self.failed.emit(str(e))


# =========================== FITS Preview (zoom/pan + toolbar + histogram) ===========================

def _autoscale_stretch(img: np.ndarray, p_lo=0.25, p_hi=99.75, asinh_g=5.0) -> np.ndarray:
    """Return 8-bit grayscale view with percentile clip + gentle asinh."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    lo = np.percentile(finite, p_lo)
    hi = np.percentile(finite, p_hi)
    if not np.isfinite(hi) or not np.isfinite(lo) or hi <= lo:
        hi = float(np.max(finite))
        lo = float(np.min(finite))
        if hi <= lo:
            hi = lo + 1.0
    norm = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    if asinh_g and asinh_g > 0:
        norm = np.arcsinh(asinh_g * norm) / np.arcsinh(asinh_g)
    view = (norm * 255.0).astype(np.uint8)
    return view


class _PreviewGraphicsView(QGraphicsView):
    """GraphicsView with smooth zoom (wheel) and pan (drag), emits scale changes."""
    scaleChanged = Signal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force smooth resampling (prevents cross-hatched/moire)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.LosslessImageRendering)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(Qt.black)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._auto_fit = True
        self._last_scale = 1.0

    def _emit_scale_if_changed(self):
        try:
            s = float(self.transform().m11())
            if not np.isfinite(s):
                s = 1.0
        except Exception:
            s = 1.0
        if abs(s - self._last_scale) > 1e-3:
            self._last_scale = s
            self.scaleChanged.emit(s)

    def wheelEvent(self, event: QWheelEvent) -> None:
        angle = event.angleDelta().y()
        if angle == 0:
            return super().wheelEvent(event)
        factor = 1.25 if angle > 0 else 1/1.25
        self._auto_fit = False
        self.scale(factor, factor)
        self._emit_scale_if_changed()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._auto_fit and self.scene() is not None:
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
        self._emit_scale_if_changed()


class _HistDialog(QDialog):
    def __init__(self, view8: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Histogram")
        self.resize(600, 400)
        layout = QVBoxLayout(self)
        self.fig = Figure(figsize=(6, 4), layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        ax = self.fig.add_subplot(111)
        vals = view8.astype(np.uint8).ravel()
        counts, bins = np.histogram(vals, bins=256, range=(0, 255))
        ax.bar((bins[:-1]+bins[1:])*0.5, counts, width=(bins[1]-bins[0]), align="center")
        ax.set_xlabel("8-bit value")
        ax.set_ylabel("Pixels per bin")  # <— clarified label
        ax.grid(True, axis="y", alpha=0.3)
        self.canvas.draw_idle()
        btns = QDialogButtonBox(QDialogButtonBox.Close, self)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class FitsPreviewWindow(QMainWindow):
    """High-quality preview with zoom-to-fit/100% and histogram."""
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Preview: {Path(path).name}")
        self.resize(1100, 850)

        # Load & stretch
        data, _ = load_fits(path)
        self.view8 = _autoscale_stretch(data)  # 8-bit grayscale view for display & histogram

        # Graphics view for smooth zoom/pan
        self.scene = QGraphicsScene(self)
        self.view = _PreviewGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)

        # Build a 24-bit RGB base image and precompute mipmaps for anti-aliased zoom-out.
        # Ensure contiguous buffer, then detach via copy() so Qt owns memory.
        h, w = self.view8.shape
        self._orig_h, self._orig_w = h, w
        rgb0 = np.repeat(self.view8[:, :, None], 3, axis=2).copy(order="C")  # (H,W,3) uint8

        def _qimg_from_rgb(arr3):
            hh, ww = arr3.shape[:2]
            qi = QImage(arr3.data, ww, hh, 3 * ww, QImage.Format_RGB888).copy()
            try:
                qi.setDevicePixelRatio(1.0)
            except Exception:
                pass
            return qi

        # Base pixmap
        base_qimg = _qimg_from_rgb(rgb0)
        self.pixmap = QPixmap.fromImage(base_qimg)

        # Create the graphics item and keep item scaled so logical size equals original image size
        self.pix_item = self.scene.addPixmap(self.pixmap)
        self.scene.setSceneRect(0, 0, w, h)

        # Precompute mipmaps (1/2, 1/4, ..., down to min dimension ~64)
        self._mips: List[Tuple[int, int, QPixmap]] = [(w, h, self.pixmap)]
        prev = rgb0
        min_dim = 64
        while prev.shape[0] >= 2 * min_dim and prev.shape[1] >= 2 * min_dim:
            ph, pw = prev.shape[:2]
            nh, nw = ph // 2, pw // 2
            if HAS_CV2:
                nxt = cv2.resize(prev, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                nxt = prev[:nh * 2, :nw * 2].reshape(nh, 2, nw, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
            qimg_l = _qimg_from_rgb(nxt)
            pm_l = QPixmap.fromImage(qimg_l)
            self._mips.append((nw, nh, pm_l))
            prev = nxt

        # Keep track of current mip width to avoid redundant swaps
        self._current_mip_w = w
        # Critical: smooth resampling & device cache to keep results consistent across zoom levels
        try:
            self.pix_item.setTransformationMode(Qt.SmoothTransformation)
            self.pix_item.setCacheMode(self.pix_item.NoCache)
        except Exception:
            pass  # Older bindings may not expose these; the view's smoothing still applies.

        # React to scale changes to choose the nearest mip level
        self.view.scaleChanged.connect(self._update_mip)

    def _update_mip(self, *_):
        # Determine the current view scale (scene -> view transform)
        try:
            s = float(self.view.transform().m11())
            if not np.isfinite(s):
                s = 1.0
        except Exception:
            s = 1.0
        # Desired on-screen width in pixels of the image
        target_w = max(1, int(self._orig_w * s))
        # Select the mip whose width is closest to target
        best = min(self._mips, key=lambda t: abs(t[0] - target_w))
        bw, bh, bpm = best
        if bw != self._current_mip_w:
            self._current_mip_w = bw
            self.pix_item.setPixmap(bpm)
            # Scale item so its logical size remains (orig_w, orig_h)
            sx = self._orig_w / bw
            self.pix_item.setScale(sx)
            try:
                self.pix_item.setTransformationMode(Qt.SmoothTransformation)
                self.pix_item.setCacheMode(self.pix_item.NoCache)
            except Exception:
                pass

        # Toolbar (Fit, 100%, Histogram) — ensure single instance, avoid duplicate buttons
        existing = getattr(self, "_toolbar", None)
        if existing is None:
            for old_tb in self.findChildren(QToolBar):
                try:
                    self.removeToolBar(old_tb)
                except Exception:
                    pass
                old_tb.deleteLater()
            tb = QToolBar("Preview Tools", self)
            tb.setObjectName("previewToolbar")
            tb.setIconSize(QSize(22, 22))
            self.addToolBar(tb)
            self._toolbar = tb
        else:
            self._toolbar.clear()

        act_fit = QAction("Zoom to fit", self)
        act_fit.setStatusTip("Fit image to window")
        act_fit.triggered.connect(self.zoom_to_fit)
        self._toolbar.addAction(act_fit)

        act_100 = QAction("Zoom 100%", self)
        act_100.setStatusTip("Display at 1:1 pixel scale")
        act_100.triggered.connect(self.zoom_100)
        self._toolbar.addAction(act_100)

        self._toolbar.addSeparator()
        act_hist = QAction("Histogram", self)
        act_hist.setStatusTip("Show histogram of the stretched preview")
        act_hist.triggered.connect(self.show_histogram)
        self._toolbar.addAction(act_hist)

    def showEvent(self, ev):
        super().showEvent(ev)
        self.view._auto_fit = True
        self.zoom_to_fit()  # fill window initially
        self._update_mip()

    @Slot()
    def zoom_to_fit(self):
        self.view._auto_fit = True
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._update_mip()

    @Slot()
    def zoom_100(self):
        self.view._auto_fit = False
        self.view.resetTransform()  # 1:1 pixel display
        self._update_mip()

    @Slot()
    def show_histogram(self):
        dlg = _HistDialog(self.view8, parent=self)
        dlg.exec()


# =========================== Graph Window (+ Filters/Export/Table) ===========================

METRIC_FIELDS = [
    ("fwhm_px",       "FWHM (px)"),
    ("hfr_px",        "HFR (px)"),
    ("eccentricity",  "Eccentricity"),
    ("stars",         "Stars detected"),
]

TABLE_COLUMNS = [
    ("index", "#"),
    ("kept", "Kept"),
    ("filename", "Filename"),
    ("fwhm_px", "FWHM (px)"),
    ("hfr_px", "HFR (px)"),
    ("eccentricity", "Eccentricity"),
    ("stars", "Stars"),
    ("noise_sigma", "Noise σ (MAD)"),
    ("median", "Median"),
    ("mean", "Mean"),
    ("stddev", "StdDev"),
    ("snr_star_median", "SNR (star median)"),
    ("width", "W"),
    ("height", "H"),
    ("downsample", "Down"),
]

METRIC_HELP: Dict[str, str] = {
    "fwhm_px": "<b>FWHM (px)</b>: Full Width at Half Maximum of star profiles. Smaller = sharper.",
    "hfr_px": "<b>HFR (px)</b>: Radius containing half the star flux. Smaller = sharper.",
    "eccentricity": "<b>Eccentricity</b>: 0 is round, 1 is a line. High values indicate elongation.",
    "stars": "<b>Stars detected</b>: Count of accepted sources. Low count can mean clouds/poor SNR.",
    "noise_sigma": "<b>Noise σ (MAD)</b>: Robust background noise. Lower = cleaner background.",
    "median": "<b>Median</b>: Image median level (sky background proxy).",
    "mean": "<b>Mean</b>: Average pixel value (sensitive to outliers).",
    "stddev": "<b>StdDev</b>: Spread of pixel values (not robust to outliers).",
    "snr_star_median": "<b>SNR (star median)</b>: Median peak-minus-background per star divided by noise σ.",
    "_bands": "<b>Grey bands</b>: Robust sigma bands about the series median (±1σ/±2σ where σ=1.4826×MAD).",
}

class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subframe Metrics & Bands")
        self.resize(640, 520)
        layout = QVBoxLayout(self)
        text = QTextEdit(self)
        text.setReadOnly(True)
        html = ["<h3>Metrics</h3><ul>"]
        for key, label in METRIC_FIELDS:
            html.append(f"<li><b>{label}</b> — {METRIC_HELP.get(key, '')}</li>")
        html.append("</ul>")
        html.append("<h3>Grey Regions</h3>")
        html.append(METRIC_HELP["_bands"])
        html.append("<p><i>Tip:</i> Use FWHM/HFR for sharpness, Eccentricity for tracking/tilt, Stars/SNR for transparency.</p>")
        text.setHtml("\n".join(html))
        layout.addWidget(text)
        btns = QDialogButtonBox(QDialogButtonBox.Ok, self)
        btns.accepted.connect(self.accept)
        layout.addWidget(btns)


class CopyWorker(QThread):
    progressed = Signal(int, int)    # done, total
    failed = Signal(str)
    finished_ok = Signal(int)        # copied count
    cancelled = Signal(int)          # copied count so far

    def __init__(self, src_paths: List[str], out_dir: str, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.src_paths = src_paths
        self.out_dir = out_dir
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def _safe_copy(self, src: Path, dest_dir: Path) -> Path:
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
            return dest
        stem = dest.stem
        suf = dest.suffix
        i = 1
        while True:
            candidate = dest_dir / f"{stem}_{i}{suf}"
            if not candidate.exists():
                shutil.copy2(src, candidate)
                return candidate
            i += 1

    def run(self):
        try:
            dest_dir = Path(self.out_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            total = len(self.src_paths)
            done = 0
            for p in self.src_paths:
                if self._cancel:
                    self.cancelled.emit(done)
                    return
                sp = Path(p)
                self._safe_copy(sp, dest_dir)
                done += 1
                self.progressed.emit(done, total)
            self.finished_ok.emit(done)
        except Exception as e:
            self.failed.emit(str(e))


class GraphWindow(QMainWindow):
    def __init__(self, results: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Subframe Selector – Graphs")
        self.results = results[:]
        self.metric_key = METRIC_FIELDS[0][0]

        # Cache metric arrays
        self.metric_arrays: Dict[str, np.ndarray] = {}
        for key, _ in METRIC_FIELDS:
            arr = np.array([self._safe_num(r.get(key)) for r in self.results], dtype=float)
            self.metric_arrays[key] = arr

        # Compute robust deviations (σ-units from series median) for FWHM/Ecc/Stars
        self.dev_arrays: Dict[str, np.ndarray] = {}
        for base_key, dev_key in [("fwhm_px", "fwhm_dev"), ("eccentricity", "ecc_dev"), ("stars", "stars_dev")]:
            arr = self.metric_arrays.get(base_key)
            if arr is None:
                continue
            med, sigma = self._robust_center_sigma(arr)
            if med is None or sigma is None or not np.isfinite(sigma) or sigma == 0:
                self.dev_arrays[dev_key] = np.full_like(arr, np.nan, dtype=float)
            else:
                self.dev_arrays[dev_key] = (arr - med) / sigma

        # Per-metric histogram bin edge cache
        self.hist_bins_cache: Dict[str, np.ndarray] = {}

        n = len(self.results)
        self.keep_mask = np.ones(n, dtype=bool)
        self.manual_excluded = np.zeros(n, dtype=bool)
        # Low-res preview cache: path -> (mtime, QPixmap). Uses ~20% linear size.
        self._preview_cache: Dict[str, Tuple[float, QPixmap]] = {}
        self._preview_downsample_factor: int = 5

        # --------- Top/bottom splitter: table on top (taller), plots+filters bottom
        main_splitter = QSplitter(Qt.Vertical, self)
        main_splitter.setChildrenCollapsible(False)

        # --- Top: details table
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in TABLE_COLUMNS])
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        # Update headers for deviation columns and hide unused ones
        try:
            headers = [c[0] for c in TABLE_COLUMNS]
            if "median" in headers:
                self.table.setHorizontalHeaderItem(headers.index("median"), QTableWidgetItem("FWHM Dev (σ)"))
            if "mean" in headers:
                self.table.setHorizontalHeaderItem(headers.index("mean"), QTableWidgetItem("Ecc Dev (σ)"))
            if "stddev" in headers:
                self.table.setHorizontalHeaderItem(headers.index("stddev"), QTableWidgetItem("Stars Dev (σ)"))
            for name in ("noise_sigma", "snr_star_median"):
                if name in headers:
                    self.table.setColumnHidden(headers.index(name), True)
        except Exception:
            pass

        self._populate_table()
        try:
            self._fill_deviation_columns()
        except Exception:
            pass
        self.table.doubleClicked.connect(self._on_row_double_clicked)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.table.installEventFilter(self)

        top_row = QWidget(self)
        row_layout = QHBoxLayout(top_row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(self.table, 3)

        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumWidth(240)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        pal = self.preview_label.palette()
        pal.setColor(QPalette.Window, QColor(20, 20, 20))
        self.preview_label.setAutoFillBackground(True)
        self.preview_label.setPalette(pal)
        row_layout.addWidget(self.preview_label, 2)

        main_splitter.addWidget(top_row)

        # --- Bottom: left plots (2/3) + right filters (1/3)
        bottom_split = QSplitter(Qt.Horizontal, self)
        bottom_split.setChildrenCollapsible(False)

        left = QWidget(self)
        left_v = QVBoxLayout(left)
        top = QHBoxLayout()
        self.lbl_metric = QLabel("Metric:", self)
        self.lbl_metric.setToolTip("Choose which metric to plot. Grey bands show robust median ±1σ/±2σ (σ from MAD).")
        top.addWidget(self.lbl_metric)

        self.metric_combo = QComboBox(self)
        for k, label in METRIC_FIELDS:
            self.metric_combo.addItem(label, userData=k)
        for i, (k, _label) in enumerate(METRIC_FIELDS):
            self.metric_combo.setItemData(i, METRIC_HELP.get(k, ""), Qt.ToolTipRole)
        self.metric_combo.setToolTip("Select a metric. Hover options for definitions.")
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        top.addWidget(self.metric_combo)

        self.chk_lock_bins = QCheckBox("Lock histogram bins", self)
        self.chk_lock_bins.setChecked(True)
        self.chk_lock_bins.setToolTip("If checked, histogram bin edges are computed once per metric and reused. Uncheck to recompute bins on each redraw.")
        self.chk_lock_bins.toggled.connect(self._on_lock_bins_toggled)
        top.addWidget(self.chk_lock_bins)

        top.addStretch()
        self.btn_help = QPushButton("Help", self)
        self.btn_help.setToolTip("Open a detailed explanation of metrics and grey sigma bands.")
        self.btn_help.clicked.connect(self._show_help)
        top.addWidget(self.btn_help)
        left_v.addLayout(top)

        # Wider left plot area: 2:1
        self.fig = Figure(figsize=(10, 5), layout="constrained")
        self.canvas = FigureCanvas(self.fig)
        left_v.addWidget(self.canvas)
        bottom_split.addWidget(left)

        # Right pane: Filters + Export
        right = QWidget(self)
        right_v = QVBoxLayout(right)

        master_row = QHBoxLayout()
        self.chk_enable_filters = QCheckBox("Enable filters", self)
        self.chk_enable_filters.setChecked(True)  # enabled by default
        self.chk_enable_filters.setToolTip("When off, all frames are kept and filter controls are disabled. Turn on to restrict ranges.")
        master_row.addWidget(self.chk_enable_filters)
        master_row.addStretch()
        right_v.addLayout(master_row)

        filters_group = QGroupBox("Filters (keep frames within ranges)", self)
        filters_layout = QGridLayout(filters_group)

        self.filter_controls: Dict[str, Dict[str, Any]] = {}
        row = 0
        for key, label in METRIC_FIELDS:
            arr = self.metric_arrays[key]
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                continue
            lo_obs = float(np.nanmin(finite))
            hi_obs = float(np.nanmax(finite))
            if not np.isfinite(lo_obs) or not np.isfinite(hi_obs):
                continue
            if math.isclose(lo_obs, hi_obs):
                lo_obs, hi_obs = lo_obs - 1.0, hi_obs + 1.0

            span = hi_obs - lo_obs
            pad = max(1e-9, 0.02 * abs(span))
            lo = math.floor((lo_obs - pad) * 100.0) / 100.0
            hi = math.ceil((hi_obs + pad) * 100.0) / 100.0

            lbl = QLabel(label, self)
            lbl.setToolTip(f"Filter range for {label}. Non-finite values (NaN/Inf) are ignored by this filter.")
            filters_layout.addWidget(lbl, row, 0, 1, 3)

            slider_min = QSlider(Qt.Horizontal, self); slider_min.setRange(0, 1000); slider_min.setValue(0)
            slider_max = QSlider(Qt.Horizontal, self); slider_max.setRange(0, 1000); slider_max.setValue(1000)
            slider_min.setToolTip(f"Minimum {label}")
            slider_max.setToolTip(f"Maximum {label}")
            filters_layout.addWidget(slider_min, row+1, 0, 1, 2)
            filters_layout.addWidget(slider_max, row+2, 0, 1, 2)

            read_min = QDoubleSpinBox(self); read_min.setDecimals(2); read_min.setRange(lo, hi); read_min.setValue(lo)
            read_max = QDoubleSpinBox(self); read_max.setDecimals(2); read_max.setRange(lo, hi); read_max.setValue(hi)
            read_min.setToolTip(f"Minimum value for {label}")
            read_max.setToolTip(f"Maximum value for {label}")
            filters_layout.addWidget(read_min, row+1, 2)
            filters_layout.addWidget(read_max, row+2, 2)

            def slider_to_val(pos: int, lo=lo, hi=hi) -> float:
                return lo + (hi - lo) * (pos / 1000.0)

            def val_to_slider(val: float, lo=lo, hi=hi) -> int:
                if hi == lo:
                    return 0
                t = (val - lo) / (hi - lo)
                return int(round(np.clip(t, 0.0, 1.0) * 1000))

            self.filter_controls[key] = {
                "label": lbl,
                "lo_default": lo, "hi_default": hi,
                "lo_obs": lo_obs, "hi_obs": hi_obs,
                "slider_min": slider_min, "slider_max": slider_max,
                "read_min": read_min, "read_max": read_max,
                "val_to_slider": val_to_slider, "slider_to_val": slider_to_val,
            }

            def on_min_slider(pos, k=key):
                c = self.filter_controls[k]
                v = c["slider_to_val"](pos)
                vmax = c["read_max"].value()
                if v > vmax:
                    v = vmax
                    pos = c["val_to_slider"](v)
                    c["slider_min"].blockSignals(True); c["slider_min"].setValue(pos); c["slider_min"].blockSignals(False)
                c["read_min"].blockSignals(True); c["read_min"].setValue(v); c["read_min"].blockSignals(False)
                self._apply_filters()

            def on_max_slider(pos, k=key):
                c = self.filter_controls[k]
                v = c["slider_to_val"](pos)
                vmin = c["read_min"].value()
                if v < vmin:
                    v = vmin
                    pos = c["val_to_slider"](v)
                    c["slider_max"].blockSignals(True); c["slider_max"].setValue(pos); c["slider_max"].blockSignals(False)
                c["read_max"].blockSignals(True); c["read_max"].setValue(v); c["read_max"].blockSignals(False)
                self._apply_filters()

            def on_min_spin(val, k=key):
                c = self.filter_controls[k]
                vmax = c["read_max"].value()
                if val > vmax:
                    val = vmax
                    c["read_min"].blockSignals(True); c["read_min"].setValue(val); c["read_min"].blockSignals(False)
                c["slider_min"].blockSignals(True); c["slider_min"].setValue(c["val_to_slider"](val)); c["slider_min"].blockSignals(False)
                self._apply_filters()

            def on_max_spin(val, k=key):
                c = self.filter_controls[k]
                vmin = c["read_min"].value()
                if val < vmin:
                    val = vmin
                    c["read_max"].blockSignals(True); c["read_max"].setValue(val); c["read_max"].blockSignals(False)
                c["slider_max"].blockSignals(True); c["slider_max"].setValue(c["val_to_slider"](val)); c["slider_max"].blockSignals(False)
                self._apply_filters()

            slider_min.valueChanged.connect(on_min_slider)
            slider_max.valueChanged.connect(on_max_slider)
            read_min.valueChanged.connect(on_min_spin)
            read_max.valueChanged.connect(on_max_spin)

            row += 3

        btn_row = QHBoxLayout()
        self.btn_reset = QPushButton("Reset Filters", self)
        self.btn_reset.setToolTip("Restore padded default ranges for all filters (rounded to hundredths).")
        self.btn_reset.clicked.connect(self._reset_filters)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        if row == 0:
            filters_layout.addLayout(btn_row, 0, 0, 1, 3)
        else:
            filters_layout.addLayout(btn_row, row, 0, 1, 3)
            row += 1

        out_group = QGroupBox("Export kept frames", self)
        out_layout = QHBoxLayout(out_group)
        self.le_outdir = QLineEdit(self); self.le_outdir.setPlaceholderText("Select output directory...")
        self.le_outdir.setToolTip("Destination folder for copied kept files.")
        self.btn_browse_out = QPushButton("Browse…", self); self.btn_browse_out.setToolTip("Select output directory")
        self.btn_export = QPushButton("Export Kept", self); self.btn_export.setToolTip("Copy kept files to the output directory.")
        self.btn_export.setEnabled(False)
        out_layout.addWidget(self.le_outdir, 1)
        out_layout.addWidget(self.btn_browse_out)
        out_layout.addWidget(self.btn_export)

        right_v.addWidget(filters_group)
        right_v.addWidget(out_group)
        right_v.addStretch()

        bottom_split.addWidget(left)
        bottom_split.addWidget(right)
        bottom_split.setStretchFactor(0, 2)  # left 2/3
        bottom_split.setStretchFactor(1, 1)  # right 1/3

        main_splitter.addWidget(bottom_split)
        main_splitter.setStretchFactor(0, 3)  # table
        main_splitter.setStretchFactor(1, 5)  # plots
        main_splitter.setSizes([420, 600])

        self.setCentralWidget(main_splitter)
        self.resize(1280, 780)

        # Wire export + master toggle
        self.btn_browse_out.clicked.connect(self._choose_outdir)
        self.btn_export.clicked.connect(self._export_kept)
        self.chk_enable_filters.toggled.connect(self._on_enable_filters_toggled)

        self._set_filters_enabled(True)
        self._reset_filters()
        self._init_plot()
        self.plot_metric()

        self._preview_cacher: Optional[PreviewCacheWorker] = None
        self._start_preview_caching()

    def closeEvent(self, e: QCloseEvent) -> None:
        if self._preview_cacher and self._preview_cacher.isRunning():
            self._preview_cacher.request_cancel()
            self._preview_cacher.wait(2000)  # Wait up to 2s
        super().closeEvent(e)

    # ---------- Caching

    @Slot(str, float, QPixmap)
    def _on_preview_ready(self, path: str, mtime: float, pixmap: QPixmap):
        """Slot to receive a pre-cached preview pixmap from the background worker."""
        if self._preview_cache is not None:
            self._preview_cache[path] = (mtime, pixmap)

    def _start_preview_caching(self):
        """Kicks off the background thread to pre-cache all preview images."""
        if self._preview_cacher and self._preview_cacher.isRunning():
            self._preview_cacher.request_cancel()
            self._preview_cacher.wait()

        paths = [r["path"] for r in self.results]
        self._preview_cacher = PreviewCacheWorker(paths, self._preview_downsample_factor, self)
        self._preview_cacher.preview_ready.connect(self._on_preview_ready)
        self._preview_cacher.start()

    # ---------- Utils

    @staticmethod
    def _safe_num(v):
        try:
            return float(v)
        except Exception:
            return np.nan

    @staticmethod
    def _label_for(key: str) -> str:
        for k, label in METRIC_FIELDS:
            if k == key:
                return label
        return key

    @staticmethod
    def _robust_center_sigma(data: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return None, None
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.std(finite))
        return med, sigma

    # ---------- Table

    def _populate_table(self):
        self.table.setRowCount(len(self.results))
        for i, r in enumerate(self.results):
            def item(val):
                if isinstance(val, (int, np.integer)):
                    it = QTableWidgetItem(str(val)); it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter); return it
                if isinstance(val, float) or isinstance(val, np.floating):
                    it = QTableWidgetItem(f"{val:.6g}"); it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter); return it
                return QTableWidgetItem(str(val))

            filename = Path(r["path"]).name
            values = [
                i+1, "✓", filename,
                r.get("fwhm_px"), r.get("hfr_px"), r.get("eccentricity"),
                r.get("stars"), r.get("noise_sigma"),
                r.get("median"), r.get("mean"), r.get("stddev"),
                r.get("snr_star_median"),
                r.get("width"), r.get("height"), r.get("downsample"),
            ]
            for col, (_key, _label) in enumerate(TABLE_COLUMNS):
                self.table.setItem(i, col, item(values[col]))

        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def _fill_deviation_columns(self):
        headers = [c[0] for c in TABLE_COLUMNS]
        col_med = headers.index("median") if "median" in headers else None
        col_mean = headers.index("mean") if "mean" in headers else None
        col_std = headers.index("stddev") if "stddev" in headers else None
        fdev = getattr(self, "dev_arrays", {}).get("fwhm_dev")
        edev = getattr(self, "dev_arrays", {}).get("ecc_dev")
        sdev = getattr(self, "dev_arrays", {}).get("stars_dev")
        n = len(self.results)
        def mk(val):
            it = QTableWidgetItem("" if val is None or not np.isfinite(val) else f"{val:.3f}")
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return it
        for i in range(n):
            if col_med is not None and isinstance(fdev, np.ndarray) and i < fdev.size:
                self.table.setItem(i, col_med, mk(float(fdev[i])))
            if col_mean is not None and isinstance(edev, np.ndarray) and i < edev.size:
                self.table.setItem(i, col_mean, mk(float(edev[i])))
            if col_std is not None and isinstance(sdev, np.ndarray) and i < sdev.size:
                self.table.setItem(i, col_std, mk(float(sdev[i])))

    def _update_table_kept_marks(self):
        red = QBrush(QColor(220, 90, 90))
        normal = QBrush(Qt.white)
        for i in range(len(self.results)):
            kept = bool(self.keep_mask[i])
            kept_item = self.table.item(i, 1)
            if kept_item:
                kept_item.setText("✓" if kept else "×")
                kept_item.setForeground(normal if kept else red)
            for c in range(self.table.columnCount()):
                it = self.table.item(i, c)
                if not it:
                    continue
                it.setForeground(normal if kept else red)

    def _on_row_double_clicked(self, model_index):
        row = model_index.row()
        fname_item = self.table.item(row, 2)  # Filename column
        if not fname_item:
            return
        fname = fname_item.text()
        for r in self.results:
            if Path(r["path"]).name == fname:
                self._open_preview(r["path"])
                break

    def eventFilter(self, obj, event):
        if obj is self.table and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Delete:
                self._exclude_selected_rows()
                return True
        return super().eventFilter(obj, event)

    def _exclude_selected_rows(self):
        if not hasattr(self, "manual_excluded"):
            return
        rows = [idx.row() for idx in self.table.selectionModel().selectedRows()]
        if not rows:
            return
        for r in rows:
            if 0 <= r < len(self.manual_excluded):
                # Toggle: if currently excluded, include; otherwise exclude
                self.manual_excluded[r] = not bool(self.manual_excluded[r])
        self._apply_filters()

    def _on_table_selection_changed(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        self._load_preview_for_row(rows[0].row())

    def _load_preview_for_row(self, row: int):
        try:
            path = self.results[row]["path"]
            self._preview_base_pix = self._get_preview_pixmap(path)
            self._rescale_preview()
        except Exception as e:
            self.preview_label.setText(str(e))

    def _rescale_preview(self):
        pix = getattr(self, "_preview_base_pix", None)
        if pix is None:
            return
        scaled = pix.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

    def _get_preview_pixmap(self, path: str) -> QPixmap:
        try:
            mtime = os.path.getmtime(path)
        except Exception:
            mtime = 0.0
        cached = self._preview_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        img, _ = load_fits(path)
        f = max(1, int(self._preview_downsample_factor))
        low = downsample(img, f)
        view8 = _autoscale_stretch(low)
        h, w = view8.shape
        rgb = np.repeat(view8[:, :, None], 3, axis=2).copy(order="C")
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self._preview_cache[path] = (mtime, pix)
        return pix

    def _open_preview(self, path: str):
        self.preview = FitsPreviewWindow(path, parent=self)
        self.preview.showMaximized()

    # ---------- Filters

    def _on_enable_filters_toggled(self, checked: bool):
        self._set_filters_enabled(checked)
        self._apply_filters()

    def _set_filters_enabled(self, enabled: bool):
        for ctrl in self.filter_controls.values():
            for key in ("slider_min", "slider_max", "read_min", "read_max", "label"):
                ctrl[key].setEnabled(enabled)

    def _reset_filters(self):
        for key, ctrl in self.filter_controls.items():
            lo, hi = ctrl["lo_default"], ctrl["hi_default"]
            lo = max(min(lo, hi), lo); hi = max(hi, lo)
            ctrl["read_min"].blockSignals(True); ctrl["read_min"].setRange(lo, hi); ctrl["read_min"].setValue(lo); ctrl["read_min"].blockSignals(False)
            ctrl["read_max"].blockSignals(True); ctrl["read_max"].setRange(lo, hi); ctrl["read_max"].setValue(hi); ctrl["read_max"].blockSignals(False)
            ctrl["slider_min"].blockSignals(True); ctrl["slider_min"].setValue(0); ctrl["slider_min"].blockSignals(False)
            ctrl["slider_max"].blockSignals(True); ctrl["slider_max"].setValue(1000); ctrl["slider_max"].blockSignals(False)
        self._apply_filters()

    def _apply_filters(self):
        n = len(self.results)
        if not self.chk_enable_filters.isChecked():
            keep = np.ones(n, dtype=bool)
        else:
            keep = np.ones(n, dtype=bool)
            for key, ctrl in self.filter_controls.items():
                v = self.metric_arrays[key]
                lo = ctrl["read_min"].value()
                hi = ctrl["read_max"].value()
                finite = np.isfinite(v)
                in_range = (~finite) | ((v >= lo) & (v <= hi))
                keep &= in_range
        # Apply manual exclusions
        if hasattr(self, "manual_excluded"):
            keep &= (~self.manual_excluded)
        self.keep_mask = keep

        self._update_table_kept_marks()
        self.plot_metric()
        self._update_export_enabled()

    # ---------- Plotting

    def _show_help(self):
        dlg = HelpDialog(self)
        dlg.exec()

    def _on_metric_changed(self, _idx):
        self.metric_key = self.metric_combo.currentData()
        self.plot_metric()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        try:
            self._rescale_preview()
        except Exception:
            pass

    def _on_lock_bins_toggled(self, _checked: bool):
        if not self.chk_lock_bins.isChecked():
            self.hist_bins_cache.clear()
        self.plot_metric()

    def _init_plot(self):
        self.canvas.mpl_connect("pick_event", self._on_pick)

    def _on_pick(self, event):
        artist = event.artist
        if artist is getattr(self, "_scatter_kept", None):
            idxs = self._kept_indices[event.ind]
        elif artist is getattr(self, "_scatter_excl", None):
            idxs = self._excl_indices[event.ind]
        else:
            return
        if len(idxs) == 0:
            return
        row = int(np.atleast_1d(idxs)[0])
        filename = Path(self.results[row]["path"]).name
        items = self.table.findItems(filename, Qt.MatchExactly)
        if items:
            self.table.selectRow(items[0].row())
            self.table.scrollToItem(items[0], QAbstractItemView.PositionAtCenter)

    def plot_metric(self):
        self.fig.clear()
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax_line = self.fig.add_subplot(gs[0, 0])
        ax_hist = self.fig.add_subplot(gs[0, 1])

        y = self.metric_arrays[self.metric_key]
        x = np.arange(1, len(y) + 1)

        med, sigma = self._robust_center_sigma(y)
        if med is not None and sigma is not None and np.isfinite(sigma):
            ax_line.axhspan(med - 2*sigma, med + 2*sigma, color="gray", alpha=0.18, label="±2σ (robust)")
            ax_line.axhspan(med - 1*sigma, med + 1*sigma, color="gray", alpha=0.30, label="±1σ (robust)")
            ax_line.axhline(med, color="k", linewidth=1.2, label="Median")

        finite = np.isfinite(y)
        kept_mask = (self.keep_mask & finite)
        excl_mask = ((~self.keep_mask) & finite)

        kept_x = x[kept_mask]; kept_y = y[kept_mask]
        excl_x = x[excl_mask]; excl_y = y[excl_mask]

        self._kept_indices = np.flatnonzero(kept_mask)
        self._excl_indices = np.flatnonzero(excl_mask)

        self._scatter_kept = ax_line.scatter(kept_x, kept_y, s=28, picker=True, label="Kept")
        self._scatter_excl = ax_line.scatter(excl_x, excl_y, s=28, marker="x", c="red", picker=True, label="Excluded")

        ax_line.set_xlabel("Frame index")
        ax_line.set_ylabel(self._label_for(self.metric_key))
        ax_line.grid(True, alpha=0.3)
        ax_line.legend(loc="best", fontsize=9)

        # Right: STABLE histogram + CDF using ALL finite values (ignores filters)
        all_finite = y[np.isfinite(y)]
        if all_finite.size > 0:
            if self.chk_lock_bins.isChecked() and self.metric_key in self.hist_bins_cache:
                bin_edges = self.hist_bins_cache[self.metric_key]
            else:
                bin_edges = np.histogram_bin_edges(all_finite, bins="auto")
                if self.chk_lock_bins.isChecked():
                    self.hist_bins_cache[self.metric_key] = bin_edges

            counts, bin_edges = np.histogram(all_finite, bins=bin_edges)
            ax_hist.bar((bin_edges[:-1] + bin_edges[1:]) * 0.5, counts,
                        width=(bin_edges[1:] - bin_edges[:-1]),
                        alpha=0.7, align="center", label="Count")
            ax_hist.set_ylabel("Pixels per bin")  # clarified
            ax_hist.set_xlabel(self._label_for(self.metric_key))
            ax_hist.grid(True, axis="y", alpha=0.3)

            ax_cdf = ax_hist.twinx()
            cumsum = np.cumsum(counts, dtype=float)
            cdf = cumsum / cumsum[-1] if cumsum[-1] > 0 else np.zeros_like(cumsum, dtype=float)
            try:
                ax_cdf.stairs(cdf, bin_edges, color="black", linewidth=1.2)
            except AttributeError:
                ax_cdf.step(bin_edges[1:], cdf, where="post", color="black", linewidth=1.2)
            ax_cdf.set_ylim(0.0, 1.0)
            ax_cdf.set_ylabel("Probability")
        else:
            ax_hist.set_xlabel(self._label_for(self.metric_key))
            ax_hist.set_ylabel("Pixels per bin")
            ax_hist.grid(True, alpha=0.3)

        if med is not None and sigma is not None:
            ax_line.text(0.01, 0.98, f"Median: {med:.4g}\nσ (robust): {sigma:.4g}",
                         transform=ax_line.transAxes, va="top", ha="left", fontsize=9,
                         bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=4))

        self.canvas.draw_idle()

    # ---------- Export kept files

    def _update_export_enabled(self):
        any_kept = bool(np.any(self.keep_mask))
        out_ok = bool(self.le_outdir.text().strip())
        self.btn_export.setEnabled(any_kept and out_ok)

    def _choose_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", "")
        if d:
            self.le_outdir.setText(d)
        self._update_export_enabled()

    @Slot()
    def _export_kept(self):
        outdir = self.le_outdir.text().strip()
        if not outdir:
            QMessageBox.warning(self, "Output directory", "Please choose an output directory.")
            return
        kept_paths = [self.results[i]["path"] for i, k in enumerate(self.keep_mask) if k]
        if not kept_paths:
            QMessageBox.information(self, "Nothing to export", "No kept files under current filters.")
            return

        self.copy_worker = CopyWorker(kept_paths, outdir, parent=self)
        dlg = QProgressDialog("Copying files…", "Cancel", 0, len(kept_paths), self)
        dlg.setWindowTitle("Export Kept")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumWidth(400)

        def on_prog(done, total):
            dlg.setMaximum(total)
            dlg.setValue(done)
            dlg.setLabelText(f"Copying files… {done}/{total}")

        def on_cancel_clicked():
            if self.copy_worker and self.copy_worker.isRunning():
                self.copy_worker.request_cancel()

        def on_finished_ok(copied):
            dlg.setValue(dlg.maximum())
            dlg.setLabelText(f"Export complete: {copied} file(s) copied.")
            QMessageBox.information(self, "Export complete", f"Copied {copied} file(s) to:\n{outdir}")
            dlg.close()
            self.copy_worker.deleteLater()
            self.copy_worker = None

        def on_cancelled(copied):
            QMessageBox.information(self, "Export canceled", f"Copied {copied} file(s) before cancel.")
            dlg.close()
            self.copy_worker.deleteLater()
            self.copy_worker = None

        def on_failed(err):
            QMessageBox.critical(self, "Export failed", err)
            dlg.close()
            self.copy_worker.deleteLater()
            self.copy_worker = None

        dlg.canceled.connect(on_cancel_clicked)
        self.copy_worker.progressed.connect(on_prog)
        self.copy_worker.finished_ok.connect(on_finished_ok)
        self.copy_worker.cancelled.connect(on_cancelled)
        self.copy_worker.failed.connect(on_failed)
        dlg.show()
        self.copy_worker.start()


class PreviewCacheWorker(QThread):
    """Pre-caches low-res preview pixmaps in the background."""
    preview_ready = Signal(str, float, QPixmap)  # path, mtime, pixmap

    def __init__(self, paths: List[str], downsample_factor: int, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.paths = paths
        self.downsample_factor = downsample_factor
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        for path in self.paths:
            if self._cancel:
                return
            try:
                mtime = os.path.getmtime(path)
                img, _ = load_fits(path)
                f = max(1, int(self.downsample_factor))
                low = downsample(img, f)
                view8 = _autoscale_stretch(low)
                h, w = view8.shape
                rgb = np.repeat(view8[:, :, None], 3, axis=2).copy(order="C")
                qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
                pix = QPixmap.fromImage(qimg)
                self.preview_ready.emit(path, mtime, pix)
            except Exception:
                # Silently ignore fails; the on-demand loader will catch it.
                continue


# =========================== Main Window ===========================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subframe Selector")

        central = QWidget(self)
        main = QHBoxLayout(central)

        # Left: file list + buttons
        left_box = QVBoxLayout()
        self.lbl_files = QLabel("Files:", self)
        self.lbl_files.setToolTip("List of FITS files to analyze.")
        self.file_list = QListWidget(self)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setToolTip("List of FITS files to analyze. Use 'Add Files…' to populate.")
        left_box.addWidget(self.lbl_files)
        left_box.addWidget(self.file_list, 1)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add Files…", self)
        self.btn_add.setToolTip("Select one or more FITS files to add to the list.")
        self.btn_clear = QPushButton("Clear", self)
        self.btn_clear.setToolTip("Remove all files from the list.")
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_clear)
        left_box.addLayout(btn_row)

        gen_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Statistics", self)
        self.btn_generate.setEnabled(False)
        self.btn_generate.setToolTip("Run the statistics pipeline on the files in the list.")
        self.btn_cancel = QPushButton("Cancel", self)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setToolTip("Abort processing. Partial results may be available.")
        gen_row.addWidget(self.btn_generate, 1)
        gen_row.addWidget(self.btn_cancel)
        left_box.addLayout(gen_row)

        prog_row = QHBoxLayout()
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(True)
        self.progress.setFormat("Idle")
        self.progress.setToolTip("Shows processing progress and status messages.")
        prog_row.addWidget(self.progress, 1)
        self.lbl_workers = QLabel("Workers:", self)
        prog_row.addWidget(self.lbl_workers)
        self.sp_workers = QSpinBox(self)
        self.sp_workers.setRange(1, max(1, os.cpu_count() or 4))
        self.sp_workers.setValue(max(1, os.cpu_count() or 4))
        self.sp_workers.setToolTip("Parallelism level. Threads by default; increase to use more CPU cores.")
        self.lbl_workers.setToolTip(self.sp_workers.toolTip())
        prog_row.addWidget(self.sp_workers)
        self.chk_proc = QCheckBox("Use process pool", self)
        self.chk_proc.setToolTip("Use processes instead of threads. Threads are often faster for NumPy/OpenCV; processes provide isolation.")
        prog_row.addWidget(self.chk_proc)
        # GPU + Fast stats toggles
        # self.chk_gpu = QCheckBox("Use GPU", self)
        # gpu_available = HAS_CUDA or HAS_CUPY
        # self.chk_gpu.setChecked(gpu_available)
        # self.chk_gpu.setEnabled(gpu_available)
        # tip = "Use GPU acceleration (CuPy or OpenCV CUDA). Falls back to CPU if unavailable."
        # if not gpu_available:
        #     tip += " Install CuPy (matching your CUDA) for GPU support."
        # self.chk_gpu.setToolTip(tip)
        # prog_row.addWidget(self.chk_gpu)
        self.chk_fast = QCheckBox("Fast background/noise", self)
        self.chk_fast.setChecked(True)
        self.chk_fast.setToolTip("Approximate median/MAD on a subset for speed on large images.")
        prog_row.addWidget(self.chk_fast)
        left_box.addLayout(prog_row)

        main.addLayout(left_box, 2)

        # Right: parameters
        right_panel = QVBoxLayout()
        param_group = QGroupBox("Detection Settings", self)
        form = QFormLayout(param_group)

        self.sp_down = QSpinBox(self); self.sp_down.setMinimum(1); self.sp_down.setMaximum(16); self.sp_down.setValue(2)
        self.sp_down.setToolTip("Downsample factor (block-average). Speeds up detection; FWHM/HFR are scaled back to original pixels.")

        self.sp_ksig = QDoubleSpinBox(self); self.sp_ksig.setDecimals(1); self.sp_ksig.setRange(0.5, 20.0); self.sp_ksig.setValue(4.5); self.sp_ksig.setSingleStep(0.5)
        self.sp_ksig.setToolTip("Detection threshold in σ above local background (after mild blur). Higher = fewer, stronger detections.")

        self.sp_min_area = QSpinBox(self); self.sp_min_area.setRange(1, 100000); self.sp_min_area.setValue(12)
        self.sp_min_area.setToolTip("Minimum connected-component size (pixels) to accept as a star. Raise to suppress noise/speckles.")

        self.sp_max_area = QSpinBox(self); self.sp_max_area.setRange(10, 1000000); self.sp_max_area.setValue(2000)
        self.sp_max_area.setToolTip("Maximum connected-component size (pixels). Caps large blobs (nebulae, hot regions, saturated cores).")

        self.sp_dilate = QSpinBox(self); self.sp_dilate.setRange(0, 10); self.sp_dilate.setValue(1)
        self.sp_dilate.setToolTip("Binary dilation iterations before labeling. Helps merge split star cores; too high can merge neighbors.")

        self.sp_max_stars = QSpinBox(self); self.sp_max_stars.setRange(10, 200000); self.sp_max_stars.setValue(3000)
        self.sp_max_stars.setToolTip("Maximum stars to keep (largest by area). Limits cost on dense fields.")

        form.addRow("Downsample (×):", self.sp_down); form.labelForField(self.sp_down).setToolTip(self.sp_down.toolTip())
        form.addRow("k-sigma:", self.sp_ksig);       form.labelForField(self.sp_ksig).setToolTip(self.sp_ksig.toolTip())
        form.addRow("Min area (px):", self.sp_min_area); form.labelForField(self.sp_min_area).setToolTip(self.sp_min_area.toolTip())
        form.addRow("Max area (px):", self.sp_max_area); form.labelForField(self.sp_max_area).setToolTip(self.sp_max_area.toolTip())
        form.addRow("Dilation (iter):", self.sp_dilate); form.labelForField(self.sp_dilate).setToolTip(self.sp_dilate.toolTip())
        form.addRow("Max stars:", self.sp_max_stars);   form.labelForField(self.sp_max_stars).setToolTip(self.sp_max_stars.toolTip())

        right_panel.addWidget(param_group)

        # # Pixel Scale helper
        # px_group = QGroupBox("Pixel Scale (arcsec/px helper)", self)
        # px_form = QFormLayout(px_group)

        # self.sp_px_um = QDoubleSpinBox(self); self.sp_px_um.setDecimals(3); self.sp_px_um.setRange(0.1, 50.0); self.sp_px_um.setValue(3.760)
        # self.sp_px_um.setToolTip("Camera pixel size in microns (µm).")

        # self.sp_flen = QDoubleSpinBox(self); self.sp_flen.setDecimals(1); self.sp_flen.setRange(50.0, 10000.0); self.sp_flen.setValue(530.0)
        # self.sp_flen.setToolTip("Effective focal length in millimeters (mm). Include reducers/barlows if known.")

        # self.sp_binning = QSpinBox(self); self.sp_binning.setRange(1, 8); self.sp_binning.setValue(1)
        # self.sp_binning.setToolTip("Sensor binning factor (1 = no binning). Multiplies effective pixel size.")

        # self.sp_reducer = QDoubleSpinBox(self); self.sp_reducer.setDecimals(3); self.sp_reducer.setRange(0.2, 5.0); self.sp_reducer.setValue(1.000)
        # self.sp_reducer.setToolTip("Optics multiplier: 0.8 for a 0.8× reducer, 2.0 for a 2× barlow. Multiplies focal length.")

        # self.btn_calc_scale = QPushButton("Calculate", self)
        # self.btn_calc_scale.setToolTip("Compute arcsec/pixel = 206.265 * (pixel_size_µm * binning) / (focal_length_mm * reducer). Fills the field above.")

        # px_form.addRow("Pixel size (µm):", self.sp_px_um); px_form.labelForField(self.sp_px_um).setToolTip(self.sp_px_um.toolTip())
        # px_form.addRow("Focal length (mm):", self.sp_flen); px_form.labelForField(self.sp_flen).setToolTip(self.sp_flen.toolTip())
        # px_form.addRow("Binning (×):", self.sp_binning);    px_form.labelForField(self.sp_binning).setToolTip(self.sp_binning.toolTip())
        # px_form.addRow("Reducer/Barlow (×):", self.sp_reducer); px_form.labelForField(self.sp_reducer).setToolTip(self.sp_reducer.toolTip())
        # px_form.addRow(self.btn_calc_scale)

        # right_panel.addWidget(px_group)

        hint = QLabel("Graphs use robust sigma bands like PixInsight: median ±1σ/±2σ (σ from MAD).", self)
        hint.setWordWrap(True)
        right_panel.addWidget(hint)
        right_panel.addStretch()

        main.addLayout(right_panel, 1)

        self.setCentralWidget(central)
        self.resize(1200, 720)

        # Signals
        self.btn_add.clicked.connect(self.on_add_files)
        self.btn_clear.clicked.connect(self.on_clear)
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_cancel.clicked.connect(self.on_cancel)
        self.file_list.model().rowsInserted.connect(self._update_buttons)
        self.file_list.model().rowsRemoved.connect(self._update_buttons)
        QApplication.instance().aboutToQuit.connect(self._shutdown)

        # Styling
        QApplication.setStyle("Fusion")
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(33, 37, 43))
        pal.setColor(QPalette.WindowText, Qt.white)
        pal.setColor(QPalette.Base, QColor(25, 28, 33))
        pal.setColor(QPalette.AlternateBase, QColor(40, 44, 52))
        pal.setColor(QPalette.ToolTipBase, Qt.white)
        pal.setColor(QPalette.ToolTipText, Qt.white)
        pal.setColor(QPalette.Text, Qt.white)
        pal.setColor(QPalette.Button, QColor(45, 49, 56))
        pal.setColor(QPalette.ButtonText, Qt.white)
        pal.setColor(QPalette.Highlight, QColor(82, 143, 235))
        pal.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.instance().setPalette(pal)

        self.worker: Optional[StatsWorker] = None
        self.results: Optional[List[Dict[str, Any]]] = None

    # ---------- Safe teardown helpers ----------

    def _cleanup_worker(self):
        if self.worker is None:
            return
        try:
            if self.worker.isRunning():
                self.worker.request_cancel()
                self.worker.wait(30000)
            else:
                self.worker.wait(5000)
        finally:
            self.worker.deleteLater()
            self.worker = None

    def _shutdown(self):
        self._cleanup_worker()

    def closeEvent(self, e: QCloseEvent) -> None:
        if self.worker and self.worker.isRunning():
            self.on_cancel()
            self.worker.wait(30000)
        return super().closeEvent(e)

    # ---------- Slots ----------

    @Slot()
    def on_add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select FITS files", "", "FITS files (*.fits *.fit *.fts);;All files (*.*)"
        )
        for f in files:
            if f and os.path.isfile(f):
                self.file_list.addItem(QListWidgetItem(f))
        self._update_buttons()

    @Slot()
    def on_clear(self):
        self.file_list.clear()
        self._update_buttons()

    def _update_buttons(self):
        self.btn_generate.setEnabled(self.file_list.count() > 0 and not self._is_processing())
        self.btn_add.setEnabled(not self._is_processing())
        self.btn_clear.setEnabled(not self._is_processing())
        self.btn_cancel.setEnabled(self._is_processing())
        self.sp_workers.setEnabled(not self._is_processing())
        self.chk_proc.setEnabled(not self._is_processing())

    def _is_processing(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _gather_params(self) -> Dict[str, Any]:
        arc = None
        return {
            "downsample": int(self.sp_down.value()),
            "k_sigma": float(self.sp_ksig.value()),
            "min_area": int(self.sp_min_area.value()),
            "max_area": int(self.sp_max_area.value()),
            "dilation": int(self.sp_dilate.value()),
            "max_stars": int(self.sp_max_stars.value()),
            "arcsec_per_pixel": arc,
            "use_gpu": False,
            # "use_gpu": bool(self.chk_gpu.isChecked() and HAS_CUDA),
            "fast_stats": bool(self.chk_fast.isChecked()),
        }

    @Slot()
    def on_generate(self):
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            return

        params = self._gather_params()

        self.progress.setRange(0, 0)
        self.progress.setFormat("Processing…")
        self.btn_generate.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.sp_workers.setEnabled(False)
        self.chk_proc.setEnabled(False)
        if hasattr(self, "chk_gpu"):
            self.chk_gpu.setEnabled(False)
        if hasattr(self, "chk_fast"):
            self.chk_fast.setEnabled(False)

        self.worker = StatsWorker(
            files, params, workers=int(self.sp_workers.value()),
            use_processes=self.chk_proc.isChecked(), parent=self
        )
        self.worker.progressed.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.start()

    @Slot()
    def on_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_cancel()
            self.btn_cancel.setEnabled(False)
            self.progress.setFormat("Canceling…")

    @Slot(int, int)
    def on_progress(self, done: int, total: int):
        if self.progress.maximum() <= 1:
            self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"Processed {done}/{total}")

    def _reset_idle(self):
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Idle")
        self.sp_workers.setEnabled(True)
        self.chk_proc.setEnabled(True)
        if hasattr(self, "chk_gpu"):
            self.chk_gpu.setEnabled(True)
        if hasattr(self, "chk_fast"):
            self.chk_fast.setEnabled(True)
        self._update_buttons()

    @Slot(list)
    def on_finished(self, results: List[Dict[str, Any]]):
        self.worker.wait()
        self._reset_idle()
        self.worker.deleteLater()
        self.worker = None
        if not results:
            QMessageBox.information(self, "No results", "No statistics were produced.")
            return

        # Sort results: chronological (if DATE-OBS exists), then by filename.
        # Items without DATE-OBS are grouped after those with dates.
        results.sort(key=lambda r: (r.get("date_obs") is None, r.get("date_obs", ""), Path(r.get("path", "")).name))

        self.results = results
        self.graph_window = GraphWindow(results, parent=self)
        self.graph_window.showMaximized()

    @Slot(list)
    def on_cancelled(self, partial: List[Dict[str, Any]]):
        self.worker.wait()
        self._reset_idle()
        self.worker.deleteLater()
        self.worker = None
        if not partial:
            QMessageBox.information(self, "Canceled", "Processing canceled. No results to display.")
            return
        resp = QMessageBox.question(
            self, "Processing canceled",
            f"Processing was canceled after {len(partial)} file(s).\n"
            "Do you want to view the partial results?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        if resp == QMessageBox.Yes:
            self.results = partial
            self.graph_window = GraphWindow(partial, parent=self)
            self.graph_window.showMaximized()

    @Slot(str)
    def on_failed(self, err: str):
        if self.worker:
            self.worker.wait()
            self.worker.deleteLater()
            self.worker = None
        self._reset_idle()
        QMessageBox.critical(self, "Processing failed", err)


# =========================== Entrypoint ===========================

def main():
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

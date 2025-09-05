# Astro Processor

Astro Processor is a growing collection of desktop tools for deep-sky image QA and processing. The first tool is a Subframe Selector that inspects FITS light frames, measures quality metrics, and helps you keep the best frames for stacking.

This README focuses on the Subframe Selector while keeping room for future tools.

## Features

- Subframe Selector (FITS)
  - Batch analysis of FITS frames (single files or multi-selection)
  - Metrics per frame: FWHM (px), HFR (px), Eccentricity, Star count, background/noise
  - Interactive graphs with robust bands (median +/-1 sigma and +/-2 sigma from MAD)
  - Details table (sortable) with a Kept column
  - Filters panel to keep frames within value ranges
  - Preview window (zoom-to-fit/100% + histogram) and a quick inline preview pane
  - Export kept frames with progress and cancel
  - Keyboard: press Delete in the table to toggle kept/excluded for selected rows

## Requirements

- Python 3.9+
- Packages: PySide6, numpy, scipy, astropy, matplotlib (optional: opencv-python)

Example setup (PowerShell):

```
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install --upgrade pip
pip install PySide6 numpy scipy astropy matplotlib opencv-python
```

## Run (GUI)

```
python subframe_selector_ui.py
```

- Click "Add Files..." and select FITS frames
- Choose worker count and backend (threads or processes)
- Adjust detection settings (k-sigma, min/max star area, dilation) if needed
- Click "Generate Statistics"

## Workflow

1. Inspect graphs (FWHM, HFR, Eccentricity, Stars) and the grey median +/- sigma bands
2. Use the Filters panel to keep frames within sensible ranges
3. Click a row to see an embedded preview; double-click for a full preview window
4. Press Delete to toggle include/exclude for selected rows
5. Export kept frames to a destination folder

## Interpreting metrics

- FWHM (px): smaller is better; tracks focus/seeing; compare within a session
- HFR (px): smaller is better; half-flux radius-based size
- Eccentricity: smaller is better; large values may indicate guiding/wind/tilt
- Stars: larger is often better (conditions/exposure), but depends on field content

## Tips and performance

- Downsample 2x or 3x for faster detection; metrics are scaled to original pixels
- Set workers near your CPU core count; try threads first, processes for isolation
- Use graphs to spot trends (e.g., FWHM rising over time) and tighten filters
- The embedded preview pane uses cached, low-resolution images for responsiveness

## Project layout

```
subframe_selector_ui.py        # GUI app (Subframe Selector)
subframe_selector_ui copy.py   # Reference working copy
output/                        # Example output folder (optional)
```

## Roadmap

- Optional arcsec/px reporting behind a toggle
- Deeper GPU acceleration where available
- Additional tools (calibration helpers, registration/inspection, PSF visualization)
- Preset filters (Strict / Balanced / Loose)
- CSV/Parquet export of metrics

## Contributing

- Open issues for bugs and feature requests
- Include sample FITS or synthetic data when reporting analysis problems
- Keep PRs focused and incremental

## License

This project currently does not include a license file.

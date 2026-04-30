# SmokeSight
 
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
 
Radiometrically-calibrated plume measurement from EO/IR surveillance video.

In Military and defence research, detection is a solved problem. SmokeSight does the part after that, which is per-pixel optical depth `tau(x,y,t)`, wavelength-dependent transmittance, line-of-sight column density, and dispersion coefficients, all with documented uncertainty bounds. 

---
 
## Install
 
```bash
pip install smokesight
# or, with full atmospheric calibration support
pip install "smokesight[calibrate]"
```
 
---
 
## Usage
 
```python
import smokesight as ss
 
cal    = ss.calibrate("plume.mp4", config="cal.yaml")
bg     = ss.background(cal, n_frames=100)
result = ss.retrieve(cal, bg)
 
# result.tau        -- optical depth tau(x, y, t)
# result.sigma_tau  -- per-pixel uncertainty
# result.T_lambda   -- transmittance cube (multi-band input)
 
dyn = ss.dynamics(result)
print(dyn.sigma_y_coeffs)  # Pasquill-Gifford fit
print(dyn.rise_velocity)   # m/s
 
result.to_netcdf("output.nc")
```
 
Output is CF-compliant NetCDF4 and opens directly in xarray. No conversion step needed.
 
---
 
## Pipeline
 
| Module | Input | Output |
|---|---|---|
| `calibrate` | raw video + cal metadata | radiance cube `L(x,y,t,lambda)` |
| `background` | radiance cube | background plate `L0` + confidence map |
| `retrieve` | `L`, `L0` | `tau(x,y,t)` + `sigma_tau` |
| `dynamics` | `tau` | rise velocity, `sigma_y`, `sigma_z` |
| `io` | any result | NetCDF4/CF + xarray API |
 
---
 
## Why this exists? And why did we build this?
 
Three things that do not exist together in any open package:
 
- **Imagery-to-radiometry.** Turning DN values into calibrated radiance, accounting for sensor response, atmospheric path, and background. Closed military tools do this. Open tools do not.
- **Uncertainty-propagated inversion.** Per-pixel `sigma_tau` from the Beer-Lambert inversion makes outputs usable as scientific measurements, not just visualizations.
- **Atmospheric science output formats.** CF NetCDF plus xarray means dispersion modelers, STE solvers, and sensor evaluation researchers can consume the data directly.
---
 
## Contributing
 
Open an issue before submitting large changes. Tests must pass and new measurement outputs must include uncertainty propagation, if it cannot be documented, it does not ship.
 
```bash
git clone https://github.com/TasumLuke/Smoke-Sight
pip install -e ".[dev]"
pre-commit install
pytest
```
 
---
 
## Citation
 
```bibtex
@software{smokesight,
  title = {SmokeSight: Radiometric plume measurement from EO/IR video},
  year  = {2025},
  url   = {https://github.com/TasumLuke/smokesight}
}
```

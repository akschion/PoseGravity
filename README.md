# PoseGravity

💥 Code for [PoseGravity: Pose Estimation from Points and Lines with Axis Prior](https://doi.org/10.48550/arXiv.2405.12646) 📷 📐 ⬇️ 🌔

### Overview

Solves the absolute camera pose estimation problem with additional partial knowledge of the camera's orientation. This partial knowledge (axis prior) usually comes from a sensor measurement (e.g. IMU accelerometer), a detected vanishing point in the image, or some domain knowledge assumption (e.g. camera is fixed upright on a tripod). The solution can be used by itself or as a quick starting point for general pose estimation.

**Summary of Features:**
- 🔥 Efficient O(n) runtime with closed-form solution. Typically runs orders of magnitude faster than general purpose solvers
- 🔥 Handles combinations of both point and line features in minimal and overconstrained configurations
- 🔥 Faster and more accurate solutions available for minimal and planar configurations (see paper)
- 🔥 Algorithm solutions have reduced search space to converge to general solution fairly quickly, even with moderate noise in measurement or detections

The algorithm may yield up to two solutions (which can be disambiguated by cheirality or domain knowledge), but will usually only yield one in most cases.

### Running the Code

`PoseGravity.hpp` is a header-only C++ file containing all the code for the algorithm without any dependencies. `PoseGravity.cpp` provides code for Python bindings that can be compiled using [PyBind11](https://github.com/pybind/pybind11). 

`estimatePoseWithGravity()` is the main function implementing the PoseGravity algorithm. Optionally, you can call `refinePose()` (experimental) afterwards on the result to do iterative optimization towards general solution. This approach is often faster than using general purpose solvers.

For sample code, see `test.cpp`. For other details and tips, see function headers for `estimatePoseWithGravity()` and `refinePose()`. Tested with C++14.

### Acknowledgements

Thanks to [BallerTV](www.baller.tv) for their continued support of baller research 💯 🏀

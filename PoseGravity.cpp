/*************************************************
 * @author Akshay Chandrasekhar
 * @brief Python bindings for PoseGravity algorithm using Pybind11. Requires
 * C++14 for overloaded function bindings.
 *
 * @copyright Copyright (c) 2022 Baller, Inc
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ************************************************/

#include "PoseGravity.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

//function to load data from 2D numpy arrays into vectors of arrays
template <class T>
void loadNumpyValues(const pybind11::detail::unchecked_reference<T, 2> &r, std::vector<std::array<T, 3>> &np) {
    int num_elements = r.shape(0);
    int num_cols = r.shape(1);
    if (num_cols == 2) {
        for (int i=0; i<num_elements; i++) {
            std::array<T, 3> row = {r(i, 0), r(i, 1), 1.0};
            np.push_back(row);
        }
    } else { //num_cols=3
        for (int i=0; i<num_elements; i++) {
            std::array<T, 3> row = {r(i, 0), r(i, 1), r(i, 2)};
            np.push_back(row);
        }
    }
}

//Python wrapper to call PoseGravity function from numpy inputs
//returns whether estimation was successful. Stores results in R_mat and T_array placeholders
//refine=true will run optimization to yield general pose estimate.
//See estimatePoseWithGravity and refinePose for other inputs
int estimatePoseWithGravityPybind(const py::array_t<double, py::array::c_style | py::array::forcecast> &pts2D_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &pts3D_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &lines2D_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &lines3D_v_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &lines3D_p_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &gravity_array,
                                py::array_t<double, py::array::c_style | py::array::forcecast> &R1_mat,
                                py::array_t<double, py::array::c_style | py::array::forcecast> &T1_array,
                                py::array_t<double, py::array::c_style | py::array::forcecast> &R2_mat,
                                py::array_t<double, py::array::c_style | py::array::forcecast> &T2_array,
                                double v_scale=-1, bool refine=false) {

    //load point correspondences
    std::vector<std::array<double, 3>> pts2D;
    std::vector<std::array<double, 3>> pts3D;
    auto p1 = pts2D_array.unchecked<2>();
    auto p2 = pts3D_array.unchecked<2>();
    if (p2.shape(1) != 3) throw std::runtime_error("Pts3D must be Nx3 array");
    if (p1.shape(1) != 2 && p1.shape(1) != 3) throw std::runtime_error("Pts2D must be Nx2 (image points) or Nx3 (vectors) array");
    pts2D.reserve(p1.shape(0));
    pts3D.reserve(p2.shape(0));
    loadNumpyValues(p1, pts2D);
    loadNumpyValues(p2, pts3D);

    //load line correspondences
    std::vector<std::array<double, 3>> lines2D;
    std::vector<std::array<double, 3>> lines3D_v;
    std::vector<std::array<double, 3>> lines3D_p;
    auto l1 = lines2D_array.unchecked<2>();
    auto l2 = lines3D_v_array.unchecked<2>();
    auto l3 = lines3D_p_array.unchecked<2>();
    if (l1.shape(1) != 3) throw std::runtime_error("Lines2D must be Nx3 array");
    if (l2.shape(1) != 3) throw std::runtime_error("Lines3D direction must be Nx3");
    if (l3.shape(1) != 3) throw std::runtime_error("Lines3D point must be Nx3");
    lines2D.reserve(l1.shape(0));
    lines3D_v.reserve(l2.shape(0));
    lines3D_p.reserve(l3.shape(0));
    loadNumpyValues(l1, lines2D);
    loadNumpyValues(l2, lines3D_v);
    loadNumpyValues(l3, lines3D_p);

    //load gravity
    auto g = gravity_array.unchecked<1>();
    if (g.shape(0) != 3) throw std::runtime_error("Gravity vector must have length of 3");
    std::array<double, 3> gravity = {g(0), g(1), g(2)};

    //placeholders
    std::array<double, 9> R1{}, R2{};
    std::array<double, 3> T1{}, T2{};
    //run algorithm with checks, not storing objective function value
    double cost_val = refine ? 0. : -1.;
    int num_sol = PoseGravity::estimatePoseWithGravity<double, true>(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, gravity, R1, T1, R2, T2, cost_val, v_scale);

    if (num_sol) {
        //refine pose for general problem solution, not available for minimal configurations
        if (refine && pts2D.size() + lines2D.size() >= 3) {
            bool success = false;
            if (num_sol == 1)
                success = PoseGravity::refinePose(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, R1, T1, cost_val, v_scale);
            else //planar pose has two solutions in overconstrained cases
                success = PoseGravity::refinePosePlanar(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, R1, T1, R2, T2,cost_val, v_scale);
        }

        //store solution
        auto r = R1_mat.mutable_unchecked<2>();
        auto t = T1_array.mutable_unchecked<1>();
        if (r.shape(0) != 3 || r.shape(1) != 3) throw std::runtime_error("Rotation matrix R must be 3x3");
        if (t.shape(0) != 3) throw std::runtime_error("Translation vector must have length of 3");

        t(0) = T1[0], t(1) = T1[1], t(2) = T1[2];
        r(0, 0) = R1[0], r(0, 1) = R1[1], r(0, 2) = R1[2];
        r(1, 0) = R1[3], r(1, 1) = R1[4], r(1, 2) = R1[5];
        r(2, 0) = R1[6], r(2, 1) = R1[7], r(2, 2) = R1[8];

        if (num_sol > 1) {
            auto r2 = R2_mat.mutable_unchecked<2>();
            auto t2 = T2_array.mutable_unchecked<1>();
            if (r2.shape(0) != 3 || r2.shape(1) != 3) throw std::runtime_error("Rotation matrix R must be 3x3, method with additional R, T solution placeholders must be called");
            if (t2.shape(0) != 3) throw std::runtime_error("Translation vector must have length of 3, method with additional R, T solution placeholders must be called");

            t2(0) = T2[0], t2(1) = T2[1], t2(2) = T2[2];
            r2(0, 0) = R2[0], r2(0, 1) = R2[1], r2(0, 2) = R2[2];
            r2(1, 0) = R2[3], r2(1, 1) = R2[4], r2(1, 2) = R2[5];
            r2(2, 0) = R2[6], r2(2, 1) = R2[7], r2(2, 2) = R2[8];
        }
    }

    return num_sol;
}

//overloaded function for non-minimal and non-planar pose problems which provide at most 1 solution
int estimatePoseWithGravityPybind(const py::array_t<double, py::array::c_style | py::array::forcecast> &pts2D_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &pts3D_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &lines2D_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &lines3D_v_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &lines3D_p_array,
                                const py::array_t<double, py::array::c_style | py::array::forcecast> &gravity_array,
                                py::array_t<double, py::array::c_style | py::array::forcecast> &R1_mat,
                                py::array_t<double, py::array::c_style | py::array::forcecast> &T1_array,
                                double v_scale=-1, bool refine=false) {

    auto p1 = pts2D_array.unchecked<2>();
    auto l1 = lines2D_array.unchecked<2>();
    if (p1.shape(0) + l1.shape(0) == 2)
        throw std::runtime_error("Minimal configuration detected. Please call method with additional R and T placeholders as minimal configurations can have up to 2 solutions");

    py::array_t<double, py::array::c_style | py::array::forcecast> R2_mat;
    py::array_t<double, py::array::c_style | py::array::forcecast> T2_array;
    int num_sol = estimatePoseWithGravityPybind(pts2D_array, pts3D_array, lines2D_array, lines3D_v_array, lines3D_p_array, gravity_array, R1_mat, T1_array, R2_mat, T2_array, v_scale);

    if (num_sol == 2)
        throw std::runtime_error("Configuration resulted in two solutions. Please call method with additional R and T placeholders to return both solutions.");
    return num_sol;
}

//bind
PYBIND11_MODULE(PoseGravity, handle){
    handle.doc() = "PoseGravity: Pose Estimation from Points and Lines with Axis Prior by Akshay Chandrasekhar";
    handle.def("estimatePoseWithGravity", py::overload_cast<const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            py::array_t<double, py::array::c_style | py::array::forcecast>&,
            py::array_t<double, py::array::c_style | py::array::forcecast>&,
            double, bool>(&estimatePoseWithGravityPybind));
    handle.def("estimatePoseWithGravity", py::overload_cast<const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            const py::array_t<double, py::array::c_style | py::array::forcecast>&,
            py::array_t<double, py::array::c_style | py::array::forcecast>&,
            py::array_t<double, py::array::c_style | py::array::forcecast>&,
            py::array_t<double, py::array::c_style | py::array::forcecast>&,
            py::array_t<double, py::array::c_style | py::array::forcecast>&,
            double, bool>(&estimatePoseWithGravityPybind));
}
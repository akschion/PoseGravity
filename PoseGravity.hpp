/*************************************************
 * @author Akshay Chandrasekhar
 * @brief PoseGravity algorithm. Introduced in paper "PoseGravity: Pose
 * Estimation From Points and Lines with Axis Prior"
 * https://doi.org/10.48550/arXiv.2405.12646
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

#ifndef POSEGRAVITY_POSEGRAVITY_HPP
#define POSEGRAVITY_POSEGRAVITY_HPP
#define _USE_MATH_DEFINES
#include <array>
#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>

//tolerances for zero checking
const double POSE_GRAVITY_DBL_TOL = 1e-12;
const float POSE_GRAVITY_FLT_TOL = 1e-7f;

/////////////////Matrix and Vector convenience class and operators
/** NOTE: most operations here are specific to PoseGravity algorithm and are not intended for use as general matrix/vector operations */

//convenience wrapper handle 3x3 matrix operations
namespace PoseGravityUtils {

const double DBL_TOL = POSE_GRAVITY_DBL_TOL;
const float FLT_TOL = POSE_GRAVITY_FLT_TOL;

#if FMA
namespace Matrix3x3 {

    //multiply a 3x3 matrix by a 3x3 matrix, mul_transposed transposes first matrix
    template <class T>
    std::array<T, 9> multiplyMatrix3x3(std::array<T, 9> &data, std::array<T, 9> &m, bool mul_transposed=false) {
        std::array<T, 9> tmp;
        if (mul_transposed) {
            tmp = {std::fma(data[0], m[0], std::fma(data[3], m[3], data[6] * m[6])),
                   std::fma(data[0], m[1], std::fma(data[3], m[4], data[6] * m[7])),
                   std::fma(data[0], m[2], std::fma(data[3], m[5], data[6] * m[8])),
                   std::fma(data[1], m[0], std::fma(data[4], m[3], data[7] * m[6])),
                   std::fma(data[1], m[1], std::fma(data[4], m[4], data[7] * m[7])),
                   std::fma(data[1], m[2], std::fma(data[4], m[5], data[7] * m[8])),
                   std::fma(data[2], m[0], std::fma(data[5], m[3], data[8] * m[6])),
                   std::fma(data[2], m[1], std::fma(data[5], m[4], data[8] * m[7])),
                   std::fma(data[2], m[2], std::fma(data[5], m[5], data[8] * m[8]))};
        } else {
            tmp = {std::fma(data[0], m[0], std::fma(data[1], m[3], data[2] * m[6])),
                   std::fma(data[0], m[1], std::fma(data[1], m[4], data[2] * m[7])),
                   std::fma(data[0], m[2], std::fma(data[1], m[5], data[2] * m[8])),
                   std::fma(data[3], m[0], std::fma(data[4], m[3], data[5] * m[6])),
                   std::fma(data[3], m[1], std::fma(data[4], m[4], data[5] * m[7])),
                   std::fma(data[3], m[2], std::fma(data[4], m[5], data[5] * m[8])),
                   std::fma(data[6], m[0], std::fma(data[7], m[3], data[8] * m[6])),
                   std::fma(data[6], m[1], std::fma(data[7], m[4], data[8] * m[7])),
                   std::fma(data[6], m[2], std::fma(data[7], m[5], data[8] * m[8]))};
        }
        return tmp;
    }

    //multiply a 3x3 matrix by a vector inplace, mul_transposed transposes the matrix
    template <class T>
    void multiplyVectorInplace(std::array<T, 9> &data, std::array<T, 3> &vec, bool mul_transposed=false) {
        T &v0 = vec[0], &v1 = vec[1], &v2 = vec[2];
        if (mul_transposed) {
            vec = {fma(data[0], v0, fma(data[3], v1, data[6] * v2)),
                   fma(data[1], v0, fma(data[4], v1, data[7] * v2)),
                   fma(data[2], v0, fma(data[5], v1, data[8] * v2))};
        } else {
            vec = {fma(data[0], v0, fma(data[1], v1, data[2] * v2)),
                   fma(data[3], v0, fma(data[4], v1, data[5] * v2)),
                   fma(data[6], v0, fma(data[7], v1, data[8] * v2))};
        }
    }

    //multiply a 3x3 matrix by a vector
    template <class T>
    std::array<T, 3> multiplyVector(std::array<T, 9> &data, std::array<T, 3> &vec, bool mul_transposed=false) {
        T &v0 = vec[0], &v1 = vec[1], &v2 = vec[2];
        std::array<T, 3> tmp;
        if (mul_transposed) {
            tmp = {fma(data[0], v0, fma(data[3], v1, data[6] * v2)),
                   fma(data[1], v0, fma(data[4], v1, data[7] * v2)),
                   fma(data[2], v0, fma(data[5], v1, data[8] * v2))};
        } else {
            tmp = {fma(data[0], v0, fma(data[1], v1, data[2] * v2)),
                   fma(data[3], v0, fma(data[4], v1, data[5] * v2)),
                   fma(data[6], v0, fma(data[7], v1, data[8] * v2))};
        }
        return tmp;
    }

    //invert a 3x3 symmetric matrix inplace, multiplied by -1
    template <class T>
    void negativeInverse(std::array<T, 9> &data, bool perform_checks=false) {
        T &a = data[0], &b = data[1], &c = data[2], &e = data[4], &f = data[5], &i = data[8];
        T sum1 = fma(e, i, -f * f), sum2 = fma(c, f, -b * i), sum3 = fma(b, f, -c * e);
        T det = a * sum1 + b * sum2 + c * sum3;

        //catch non-invertible matrices
        if (perform_checks && fabs(det) < (std::is_same<T, double>::value ? DBL_TOL : FLT_TOL))
            throw std::runtime_error("Matrix is not invertible! Degenerate configuration. If using single precision, try double precision");

        det = -1 / det;
        sum2 *= det, sum3 *= det;
        T sum4 = det * fma(c, b, -a * f);

        data = {det * sum1, sum2, sum3, sum2, det * fma(a, i, -c * c), sum4, sum3, sum4, det * fma(a, e, -b * b)};
    }

    //calculate 3x3 symmetric matrix adjoint
    template <class T>
    std::array<T, 9> adjoint(std::array<T, 9> &data) {
        T &a = data[0], &b = data[1], &c = data[2], &e = data[4], &f = data[5], &i = data[8];
        T sum1 = fma(c, f, -b * i), sum2 = fma(b, f, -c * e), sum3 = fma(c, b, -a * f);
        std::array<T, 9> tmp = {fma(e, i, -f * f), sum1, sum2, sum1, fma(a, i, -c * c), sum3, sum2, sum3, fma(a, e, -b * b)};
        return tmp;
    }
};
#else
namespace Matrix3x3 {

    //multiply a 3x3 matrix by a 3x3 matrix, mul_transposed transposes first matrix
    template <class T>
    std::array<T, 9> multiplyMatrix3x3(std::array<T, 9> &data, std::array<T, 9> &m, bool mul_transposed=false) {
        std::array<T, 9> tmp;
        if (mul_transposed) {
            tmp = {data[0] * m[0] + data[3] * m[3] + data[6] * m[6],
                   data[0] * m[1] + data[3] * m[4] + data[6] * m[7],
                   data[0] * m[2] + data[3] * m[5] + data[6] * m[8],
                   data[1] * m[0] + data[4] * m[3] + data[7] * m[6],
                   data[1] * m[1] + data[4] * m[4] + data[7] * m[7],
                   data[1] * m[2] + data[4] * m[5] + data[7] * m[8],
                   data[2] * m[0] + data[5] * m[3] + data[8] * m[6],
                   data[2] * m[1] + data[5] * m[4] + data[8] * m[7],
                   data[2] * m[2] + data[5] * m[5] + data[8] * m[8]};
        } else {
            tmp = {data[0] * m[0] + data[1] * m[3] + data[2] * m[6],
                   data[0] * m[1] + data[1] * m[4] + data[2] * m[7],
                   data[0] * m[2] + data[1] * m[5] + data[2] * m[8],
                   data[3] * m[0] + data[4] * m[3] + data[5] * m[6],
                   data[3] * m[1] + data[4] * m[4] + data[5] * m[7],
                   data[3] * m[2] + data[4] * m[5] + data[5] * m[8],
                   data[6] * m[0] + data[7] * m[3] + data[8] * m[6],
                   data[6] * m[1] + data[7] * m[4] + data[8] * m[7],
                   data[6] * m[2] + data[7] * m[5] + data[8] * m[8]};
        }
        return tmp;
    }

    //multiply a 3x3 matrix by a vector inplace, mul_transposed transposes the matrix
    template <class T>
    void multiplyVectorInplace(std::array<T, 9> &data, std::array<T, 3> &vec, bool mul_transposed=false) {
        T &v0 = vec[0], &v1 = vec[1], &v2 = vec[2];
        if (mul_transposed) {
            vec = {data[0] * v0 + data[3] * v1 + data[6] * v2,
                   data[1] * v0 + data[4] * v1 + data[7] * v2,
                   data[2] * v0 + data[5] * v1 + data[8] * v2};
        } else {
            vec = {data[0] * v0 + data[1] * v1 + data[2] * v2,
                   data[3] * v0 + data[4] * v1 + data[5] * v2,
                   data[6] * v0 + data[7] * v1 + data[8] * v2};
        }
    }

    //multiply a 3x3 matrix by a vector
    template <class T>
    std::array<T, 3> multiplyVector(std::array<T, 9> &data, std::array<T, 3> &vec, bool mul_transposed=false) {
        T &v0 = vec[0], &v1 = vec[1], &v2 = vec[2];
        std::array<T, 3> tmp;
        if (mul_transposed) {
            tmp = {data[0] * v0 + data[3] * v1 + data[6] * v2,
                   data[1] * v0 + data[4] * v1 + data[7] * v2,
                   data[2] * v0 + data[5] * v1 + data[8] * v2};
        } else {
            tmp = {data[0] * v0 + data[1] * v1 + data[2] * v2,
                   data[3] * v0 + data[4] * v1 + data[5] * v2,
                   data[6] * v0 + data[7] * v1 + data[8] * v2};
        }
        return tmp;
    }

    //invert a 3x3 symmetric matrix inplace, multiplied by -1
    template <class T>
    void negativeInverse(std::array<T, 9> &data, bool perform_checks=false) {
        T &a = data[0], &b = data[1], &c = data[2], &e = data[4], &f = data[5], &i = data[8];
        T sum1 = (e * i) - (f * f), sum2 = (c * f) - (b * i), sum3 = (b * f) - (c * e);
        T det = (a * sum1) + (b * sum2) + (c * sum3);

        //catch non-invertible matrices
        if (fabs(det) < (std::is_same<T, double>::value ? DBL_TOL : FLT_TOL))
            throw std::runtime_error("Matrix is not invertible! Degenerate configuration. If using single precision, try double precision");

        det = -1 / det;
        sum2 *= det, sum3 *= det;
        T sum4 = det * ((c * b) - (a * f));

        data = {det * sum1, sum2, sum3, sum2, det * ((a * i) - (c * c)), sum4, sum3, sum4, det * ((a * e) - (b * b))};
    }

    //calculate 3x3 symmetric matrix adjoint
    template <class T>
    std::array<T, 9> adjoint(std::array<T, 9> &data) {
        T &a = data[0], &b = data[1], &c = data[2], &e = data[4], &f = data[5], &i = data[8];
        T sum1 = (c * f) - (b * i), sum2 = (b * f) - (c * e), sum3 = (c * b) - (a * f);
        std::array<T, 9> tmp = {(e * i) - (f * f), sum1, sum2, sum1, (a * i) - (c * c), sum3, sum2, sum3, (a * e) - (b * b)};
        return tmp;
    }
};

#endif

//convenience methods to operate on vectors
namespace Vector {
    //converts vector to unit norm, operates in place
    template <class T>
    void normalize(std::array<T, 3> &v, bool perform_checks=false) {
        T &v0 = v[0], &v1 = v[1], &v2 = v[2];
#if FMA
        T vnorm = fma(v0, v0, fma(v1, v1, v2 * v2));
#else
        T vnorm = v0 * v0 + v1 * v1 + v2 * v2;
#endif
        //catch 0 vectors
        if (perform_checks && fabs(vnorm) < (std::is_same<T, double>::value ? DBL_TOL : FLT_TOL))
            throw std::runtime_error("Attempted divide by 0 during normalization");

        vnorm = T(1) / sqrt(vnorm);
        v0 *= vnorm, v1 *= vnorm, v2 *= vnorm;
    }

    //checks if two vector rays are incident
    template <class T>
    bool equals(std::array<T, 3> v1, std::array<T, 3> v2) {
        Vector::normalize(v1, true), Vector::normalize(v2, true);
        T tol = std::is_same<T, double>::value ? DBL_TOL : FLT_TOL;
        std::array<T, 3> vec = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
        return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]) < tol;
    }
}
}

namespace PoseGravity {

const double DBL_TOL = POSE_GRAVITY_DBL_TOL;
const float FLT_TOL = POSE_GRAVITY_FLT_TOL;

/////////////////additional math functions
//solves depressed cubic x^3 + a3*x + a4 = 0 for a4 !=0, returns first real root found
template <class T>
T solveDepressedCubic(T &a3, T &a4) {
    constexpr T third = 1.0 / 3.0;
    T tol = (std::is_same<T, double>::value ? DBL_TOL : FLT_TOL);
    //already checked fabs(a4) > tol before calling method
    if (fabs(a3) < tol) return cbrt(-a4);
    T p = third * a3;
    T q = T(0.5) * a4;
    T p3 = p * p * p;
    T q2 = q * q;
    T D = q2 + p3; //discriminant
    if (D >= T(0)) { //one real root (D>0)
        D = cbrt(sqrt(D) - q);
        D -= p / D;
        return D;
    } else { //three real roots
        D = sqrt(-p);
        q /= p * D;
        q = q < T(-1) ? T(M_PI) : (q > T(1) ? T(0) : acos(q));
        D *= T(2) * cos(third * q);
        return D;
    }
}

//given projective line, find intersection with unit circle, numerically stable. solutions stored in roots
template <class T>
int intersectLineCircle(T &l0, T &l1, T &l2, std::array<std::array<T, 2>, 4> &roots, int &offset) {
    T tol = std::is_same<T, double>::value ? DBL_TOL : FLT_TOL;
#if FMA
    T vnorm = fma(l0, l0, l1 * l1);
#else
    T vnorm = l0 * l0 + l1 * l1;
#endif
    if (fabs(l0) < tol && fabs(l1) < tol) return 0;
    if (vnorm - 1 > 0 && vnorm - 1 < 10 * tol) { //recover close misses
        vnorm = 1 / sqrt(vnorm);
        roots[offset] = {l0 * vnorm, l1 * vnorm};
        return 1;
    }
    vnorm = 1 / vnorm;
#if FMA
    T d = fma(-vnorm, l2 * l2, 1) * vnorm;
#else
    T d = (T(1) -vnorm * l2 * l2) * vnorm;
#endif
    if (d < T(0)) return T(0);

    vnorm *= -l2;
    d = sqrt(d);
    T x = l0 * vnorm, y = l1 * vnorm;
    if (d <= tol) {
        roots[offset] = {x, y};
        return 1;
    }

    T l0d = l0 * d, l1d = l1 * d;
    roots[offset] = {x + l1d, y - l0d};
    roots[offset+1] = {x - l1d, y + l0d};
    return 2;
}

//transforms 3D point and checks if it is in front of camera's optical center to validate pose
template <class T>
bool cheiralityCheck(std::array<T, 3> &pt2D, std::array<T, 3> pt3D, std::array<T, 9> &R, std::array<T, 3> &Tr) {
    //transform pose
    PoseGravityUtils::Matrix3x3::multiplyVectorInplace(R, pt3D);
    T &p30 = pt3D[0], &p31 = pt3D[1], &p32 = pt3D[2];
    p30 += Tr[0], p31 += Tr[1], p32 += Tr[2];

    //dot product with 2D pt
#if FMA
    T dot_prod = fma(p30, pt2D[0], fma(p31, pt2D[1], p32 * pt2D[2]));
#else
    T dot_prod = p30 * pt2D[0] + p31 * pt2D[1] + p32 * pt2D[2];
#endif

    return dot_prod > 0;
}

//find reference rotation so gravity vector is aligned with camera y-axis
template <class T>
std::array<T, 9> alignGravity(std::array<T, 3> &gravity) {
//    VectorUtils::normalize(gravity);
    //cross-product of gravity and {0, 1, 0} = {-gravity[2], 0, gravity[0]}
    T &g0 = gravity[0], &g1 = gravity[1], &g2 = gravity[2];
    if (g1 + 1 < (std::is_same<T, double>::value ? DBL_TOL : FLT_TOL)) {
        std::array<T, 9> rot180 = {-1, 0, 0, 0, -1, 0, 0, 0, 1};
        return rot180;
    } else {
        //Rotation from axis-angle, R = cI + s[v]x + (1-c)[v^T*v]
        T outer_factor = T(1) / (g1 + T(1));
#if FMA
        std::array<T, 9> R_array = {fma(g2 * g2, outer_factor, g1), -g0, -g2 * g0 * outer_factor,
                                    g0, g1, g2,
                                    -g2 * g0 * outer_factor, -g2, fma(g0 * g0, outer_factor, g1)};
#else
        std::array<T, 9> R_array = {g2 * g2 * outer_factor + g1, -g0, -g2 * g0 * outer_factor,
                                    g0, g1, g2,
                                    -g2 * g0 * outer_factor, -g2, g0 * g0 * outer_factor + g1};
#endif
        return R_array;
    }
}

/////////////////Pose estimation
//recover full rotation matrix and translation from root, rotate back to original camera frame
template <class T>
void recoverPose(std::array<T, 2> &rot_root, std::array<T, 9> &P, std::array<T, 9> &Rg, std::array<T, 9> &R, std::array<T, 3> &Tr) {
    //recover translation
    T &x = rot_root[0], &y = rot_root[1];
    T &t0 = Tr[0], t1 = Tr[1], t2 = Tr[2];
#if FMA
    t0 = fma(P[0], x, fma(P[1], y, P[2]));
    t1 = fma(P[3], x, fma(P[4], y, P[5]));
    t2 = fma(P[6], x, fma(P[7], y, P[8]));

    //rotate pose back to original reference frame
    T &Rg0 = Rg[0], &Rg1 = Rg[1], &Rg2 = Rg[2], &Rg3 = Rg[3], &Rg4 = Rg[4], &Rg5 = Rg[5], &Rg6 = Rg[6], &Rg7 = Rg[7], &Rg8 = Rg[8];
    Tr = {fma(Rg0, t0, fma(Rg3, t1, Rg6 * t2)), fma(Rg1, t0, fma(Rg4, t1, Rg7 * t2)), fma(Rg2, t0, fma(Rg5, t1, Rg8 * t2))};

    //R = R_gravity_alignment^T * R_est
    R = {fma(Rg0, x, -Rg6 * y), Rg3, fma(Rg0, y, Rg6 * x),
         fma(Rg1, x, -Rg7 * y), Rg4, fma(Rg1, y, Rg7 * x),
         fma(Rg2, x, -Rg8 * y), Rg5, fma(Rg2, y, Rg8 * x)};
#else
    t0 = P[0] * x + P[1] * y + P[2];
    t1 = P[3] * x + P[4] * y + P[5];
    t2 = P[6] * x + P[7] * y + P[8];

    //rotate pose back to original reference frame
    T &Rg0 = Rg[0], &Rg1 = Rg[1], &Rg2 = Rg[2], &Rg3 = Rg[3], &Rg4 = Rg[4], &Rg5 = Rg[5], &Rg6 = Rg[6], &Rg7 = Rg[7], &Rg8 = Rg[8];
    Tr = {Rg0 * t0 + Rg3 * t1 + Rg6 * t2, Rg1 * t0 + Rg4 * t1 + Rg7 * t2, Rg2 * t0 + Rg5 * t1 + Rg8 * t2};

    //R = R_gravity_alignment^T * R_est
    R = {Rg0 * x - Rg6 * y, Rg3, Rg0 * y + Rg6 * x,
         Rg1 * x - Rg7 * y, Rg4, Rg1 * y + Rg7 * x,
         Rg2 * x - Rg8 * y, Rg5, Rg2 * y + Rg8 * x};
#endif
}

// Calculate loss conic Omega matrix for optimization, P matrix for recovering translation (relabeled S in paper)
template <class T>
void generateMatrixSums(std::vector<std::array<T, 3>> &pts2D, std::vector<std::array<T, 3>> &pts3D,
                            std::vector<std::array<T, 3>> &lines2D, std::vector<std::array<T, 3>> &lines3D_v,
                            std::vector<std::array<T, 3>> &lines3D_p, std::array<T, 9> &Rg, std::array<T, 9> &P,
                            std::array<T, 6> &Omega, T v_scale, bool perform_checks) {
    int num_pts = pts2D.size(), num_lines = lines2D.size();
    T P0 = 0, P1 = 0, P2 = 0, P3, P4 = 0, P5 = 0, P6, P7, P8 = 0; //QSum
    T QASum0 = 0, QASum1 = 0, QASum2 = 0, QASum3 = 0, QASum4 = 0, QASum5 = 0, QASum6 = 0, QASum7 = 0, QASum8 = 0;
    T Omega0 = 0, Omega1 = 0, Omega2 = 0, Omega4 = 0, Omega5 = 0, Omega8 = 0;

    //rotation to new gravity aligned reference frame
    T &Rg0 = Rg[0], &Rg1 = Rg[1], &Rg2 = Rg[2], &Rg3 = Rg[3], &Rg4 = Rg[4], &Rg5 = Rg[5], &Rg6 = Rg[6], &Rg7 = Rg[7], &Rg8 = Rg[8];

    //value that improves estimation for lines by balancing line direction loss V^T * Q * V
    //if not provided, a tuned one is used instead
    v_scale = v_scale > 0 ? v_scale : 1e2;

#if FMA
    //iterate over points
    for (int i=0; i<num_pts; i++) {
        //create matrix Q
        T &v0 = pts2D[i][0], &v1 = pts2D[i][1], &v2 = pts2D[i][2];
        T a = fma(Rg0, v0, fma(Rg1, v1, Rg2 * v2)), b = fma(Rg3, v0, fma(Rg4, v1, Rg5 * v2)), c = fma(Rg6, v0, fma(Rg7, v1, Rg8 * v2));
        T a2 = a*a, b2 = b*b, c2 = c*c;
        T Q0 = b2 + c2, Q1 = -a*b, Q2 = -a*c, Q4 = a2 + c2, Q5 = -b*c, Q8 = a2 + b2; //negative skew-symmetric matrix squared
        P0 += Q0, P1 += Q1, P2 += Q2, P4 += Q4, P5 += Q5, P8 += Q8;

        //calculate Q * A and sum with QASum
        T &p = pts3D[i][0], &q = pts3D[i][1], &r = pts3D[i][2];
        T QA0 = fma(p, Q0, r * Q2), QA1 = fma(r, Q0, -p * Q2), QA2 = q * Q1;
        T QA3 = fma(p, Q1, r * Q5), QA4 = fma(r, Q1, -p * Q5), QA5 = q * Q4;
        T QA6 = fma(p, Q2, r * Q8), QA7 = fma(r, Q2, -p * Q8), QA8 = q * Q5;
        QASum0 += QA0, QASum1 += QA1, QASum2 += QA2;
        QASum3 += QA3, QASum4 += QA4, QASum5 += QA5;
        QASum6 += QA6, QASum7 += QA7, QASum8 += QA8;

        //calculate first term (A^T * Omega * A) of Omega symmetric matrix loss. lower triangle is cheaper to calculate but store in upper triangle for consistency
        Omega0 += fma(p, QA0, r * QA6), Omega1 += fma(r, QA0, -p * QA6), Omega2 += q * QA3;
        Omega4 += fma(r, QA1, -p * QA7), Omega5 += q * QA4;
        Omega8 += q * QA5;
    }

    //iterate over lines
    for (int j=0; j<num_lines; j++) {
        //create matrix Q
        T &v0 = lines2D[j][0], &v1 = lines2D[j][1], &v2 = lines2D[j][2];
        T a = fma(Rg0, v0, fma(Rg1, v1, Rg2 * v2)), b = fma(Rg3, v0, fma(Rg4, v1, Rg5 * v2)), c = fma(Rg6, v0, fma(Rg7, v1, Rg8 * v2));
        T Q0 = a*a, Q1 = a*b, Q2 = a*c, Q4 = b*b, Q5 = b*c, Q8 = c*c; //self outer matrix
        P0 += Q0, P1 += Q1, P2 += Q2, P4 += Q4, P5 += Q5, P8 += Q8;

        //calculate Q * A and sum with QASum
        T t = lines3D_v[j][0], u = lines3D_v[j][1], v = lines3D_v[j][2], &p = lines3D_p[j][0], &q = lines3D_p[j][1], &r = lines3D_p[j][2];
        t *= v_scale, u *= v_scale, v *= v_scale;
        T QA0 = fma(p, Q0, r * Q2), QA1 = fma(r, Q0, -p * Q2), QA2 = q * Q1;
        T QA3 = fma(p, Q1, r * Q5), QA4 = fma(r, Q1, -p * Q5), QA5 = q * Q4;
        T QA6 = fma(p, Q2, r * Q8), QA7 = fma(r, Q2, -p * Q8), QA8 = q * Q5;
        QASum0 += QA0, QASum1 += QA1, QASum2 += QA2;
        QASum3 += QA3, QASum4 += QA4, QASum5 += QA5;
        QASum6 += QA6, QASum7 += QA7, QASum8 += QA8;

        //calculate first terms (A^T * Omega * A + V^T * Omega * V) of Omega symmetric matrix loss. lower triangle is cheaper to calculate but store in upper triangle for consistency
        //first term is not strictly needed, but provides more numerical stability for little compute
        T t2 = t * t, v_2 = v * v, tu = t * u, uv = u * v, tv = t * v, Q2tv_2 = 2 * Q2 * tv;
        Omega0 += fma(p, QA0, fma(r, QA6, fma(Q0, t2, fma(Q8, v_2, Q2tv_2))));
        Omega1 += fma(r, QA0, fma(-p, QA6, fma((Q0 - Q8), tv, (v_2 - t2) * Q2)));
        Omega2 += fma(q, QA3, fma(Q1, tu, Q5 * uv));
        Omega4 += fma(r, QA1, fma(-p, QA7, fma(Q0, v_2, fma(Q8, t2, -Q2tv_2))));
        Omega5 += fma(q, QA4, fma(Q1, uv, -Q5 * tu));
        Omega8 += fma(q, QA5, Q4 * u * u);
    }

    //calculate matrix P from total Q sums and Q*A sums
    // P = -(sum(Qp) + sum(Ql))^-1 * (sum(Qp*Ap) + sum(Ql*Al))
    P = {P0, P1, P2, 0, P4, P5, 0, 0, P8};
    PoseGravityUtils::Matrix3x3::negativeInverse(P, perform_checks);
    P0 = P[0], P1 = P[1], P2 = P[2], P3 = P[3], P4 = P[4], P5 = P[5], P6 = P[6], P7 = P[7], P8 = P[8];
    P = {fma(P0, QASum0, fma(P1, QASum3, P2 * QASum6)),
         fma(P0, QASum1, fma(P1, QASum4, P2 * QASum7)),
         fma(P0, QASum2, fma(P1, QASum5, P2 * QASum8)),
         fma(P3, QASum0, fma(P4, QASum3, P5 * QASum6)),
         fma(P3, QASum1, fma(P4, QASum4, P5 * QASum7)),
         fma(P3, QASum2, fma(P4, QASum5, P5 * QASum8)),
         fma(P6, QASum0, fma(P7, QASum3, P8 * QASum6)),
         fma(P6, QASum1, fma(P7, QASum4, P8 * QASum7)),
         fma(P6, QASum2, fma(P7, QASum5, P8 * QASum8))};
    P1 = P[1], P2 = P[2], P4 = P[4], P5 = P[5], P7 = P[7], P8 = P[8];

    //finish calculating second term of Omega
    //omega = sum((Ap + P)^T * Qp * (Ap + P)) = A^T * Omega * A + QASum^T * P when simplified
    Omega0 += fma(QASum0, P[0], fma(QASum3, P[3], QASum6 * P[6]));
    Omega1 += fma(QASum0, P1, fma(QASum3, P4, QASum6 * P7));
    Omega2 += fma(QASum0, P2, fma(QASum3, P5, QASum6 * P8));
    Omega4 += fma(QASum1, P1, fma(QASum4, P4, QASum7 * P7));
    Omega5 += fma(QASum1, P2, fma(QASum4, P5, QASum7 * P8));
    Omega8 += fma(QASum2, P2, fma(QASum5, P5, QASum8 * P8));
#else
    //iterate over points
    for (int i=0; i<num_pts; i++) {
        //create matrix Q
        T &v0 = pts2D[i][0], &v1 = pts2D[i][1], &v2 = pts2D[i][2];
        T a = Rg0 * v0 + Rg1 * v1 + Rg2 * v2, b = Rg3 * v0 + Rg4 * v1 + Rg5 * v2, c = Rg6 * v0 + Rg7 * v1 + Rg8 * v2;
        T a2 = a*a, b2 = b*b, c2 = c*c;
        T Q0 = b2 + c2, Q1 = -a*b, Q2 = -a*c, Q4 = a2 + c2, Q5 = -b*c, Q8 = a2 + b2; //negative skew-symmetric matrix squared
        P0 += Q0, P1 += Q1, P2 += Q2, P4 += Q4, P5 += Q5, P8 += Q8;

        //calculate Q * A and sum with QASum
        T &p = pts3D[i][0], &q = pts3D[i][1], &r = pts3D[i][2];
        T QA0 = (p * Q0) + (r * Q2), QA1 = (r * Q0) - (p * Q2), QA2 = q * Q1;
        T QA3 = (p * Q1) + (r * Q5), QA4 = (r * Q1) - (p * Q5), QA5 = q * Q4;
        T QA6 = (p * Q2) + (r * Q8), QA7 = (r * Q2) - (p * Q8), QA8 = q * Q5;
        QASum0 += QA0, QASum1 += QA1, QASum2 += QA2;
        QASum3 += QA3, QASum4 += QA4, QASum5 += QA5;
        QASum6 += QA6, QASum7 += QA7, QASum8 += QA8;

        //calculate first term (A^T * Omega * A) of Omega symmetric matrix loss. lower triangle is cheaper to calculate but store in upper triangle for consistency
        Omega0 += (p * QA0) + (r * QA6), Omega1 += (r * QA0) - (p * QA6), Omega2 += q * QA3;
        Omega4 += (r * QA1) - (p * QA7), Omega5 += q * QA4;
        Omega8 += q * QA5;
    }

    //iterate over lines
    for (int j=0; j<num_lines; j++) {
        //create matrix Q
        T &v0 = lines2D[j][0], &v1 = lines2D[j][1], &v2 = lines2D[j][2];
        T a = Rg0 * v0 + Rg1 * v1 + Rg2 * v2, b = Rg3 * v0 + Rg4 * v1 + Rg5 * v2, c = Rg6 * v0 + Rg7 * v1 + Rg8 * v2;
        T Q0 = a*a, Q1 = a*b, Q2 = a*c, Q4 = b*b, Q5 = b*c, Q8 = c*c; //self outer matrix
        P0 += Q0, P1 += Q1, P2 += Q2, P4 += Q4, P5 += Q5, P8 += Q8;

        //calculate Q * A and sum with QASum
        T t = lines3D_v[j][0], u = lines3D_v[j][1], v = lines3D_v[j][2], &p = lines3D_p[j][0], &q = lines3D_p[j][1], &r = lines3D_p[j][2];
        t *= v_scale, u *= v_scale, v *= v_scale;
        T QA0 = (p * Q0) + (r * Q2), QA1 = (r * Q0) - (p * Q2), QA2 = q * Q1;
        T QA3 = (p * Q1) + (r * Q5), QA4 = (r * Q1) - (p * Q5), QA5 = q * Q4;
        T QA6 = (p * Q2) + (r * Q8), QA7 = (r * Q2) - (p * Q8), QA8 = q * Q5;
        QASum0 += QA0, QASum1 += QA1, QASum2 += QA2;
        QASum3 += QA3, QASum4 += QA4, QASum5 += QA5;
        QASum6 += QA6, QASum7 += QA7, QASum8 += QA8;

        //calculate first terms (A^T * Omega * A + V^T * Omega * V) of Omega symmetric matrix loss. lower triangle is cheaper to calculate but store in upper triangle for consistency
        //first half (A^T * Omega *A) is not strictly needed, but provides more numerical stability for little compute
        T t2 = t * t, v_2 = v * v, tu = t * u, uv = u * v, tv = t * v, Q2tv_2 = 2 * Q2 * tv;
        Omega0 += (p * QA0) + (r * QA6) + (Q0 * t2) + (Q8 * v_2) + Q2tv_2;
        Omega1 += (r * QA0) - (p * QA6) + ((Q0 - Q8) * tv) + ((v_2 - t2) * Q2);
        Omega2 += (q * QA3) + (Q1 * tu) + (Q5 * uv);
        Omega4 += (r * QA1) - (p * QA7) + (Q0 * v_2) + (Q8 * t2) - Q2tv_2;
        Omega5 += (q * QA4) + (Q1 * uv) - (Q5 * tu);
        Omega8 += (q * QA5) + (Q4 * u * u);
    }

    //calculate matrix P from total Q sums and Q*A sums
    // P = -(sum(Qp) + sum(Ql))^-1 * (sum(Qp*Ap) + sum(Ql*Al))
    P = {P0, P1, P2, 0, P4, P5, 0, 0, P8};
    PoseGravityUtils::Matrix3x3::negativeInverse(P);
    P0 = P[0], P1 = P[1], P2 = P[2], P3 = P[3], P4 = P[4], P5 = P[5], P6 = P[6], P7 = P[7], P8 = P[8];
    P = {(P0 * QASum0) + (P1 * QASum3) + (P2 * QASum6),
         (P0 * QASum1) + (P1 * QASum4) + (P2 * QASum7),
         (P0 * QASum2) + (P1 * QASum5) + (P2 * QASum8),
         (P3 * QASum0) + (P4 * QASum3) + (P5 * QASum6),
         (P3 * QASum1) + (P4 * QASum4) + (P5 * QASum7),
         (P3 * QASum2) + (P4 * QASum5) + (P5 * QASum8),
         (P6 * QASum0) + (P7 * QASum3) + (P8 * QASum6),
         (P6 * QASum1) + (P7 * QASum4) + (P8 * QASum7),
         (P6 * QASum2) + (P7 * QASum5) + (P8 * QASum8)};
    P1 = P[1], P2 = P[2], P4 = P[4], P5 = P[5], P7 = P[7], P8 = P[8];

    //finish calculating second term of Omega
    //omega = sum((Ap + P)^T * Qp * (Ap + P)) = A^T * Omega * A + QASum^T * P when simplified
    Omega0 += (QASum0 * P[0]) + (QASum3 * P[3]) + (QASum6 * P[6]);
    Omega1 += (QASum0 * P1) + (QASum3 * P4) + (QASum6 * P7);
    Omega4 += (QASum1 * P1) + (QASum4 * P4) + (QASum7 * P7);
    Omega2 += (QASum0 * P2) + (QASum3 * P5) + (QASum6 * P8);
    Omega5 += (QASum1 * P2) + (QASum4 * P5) + (QASum7 * P8);
    Omega8 += (QASum2 * P2) + (QASum5 * P5) + (QASum8 * P8);
#endif
    T scale = T(1.0) / T(num_pts + num_lines); //scale for better stability
    //avoiding multiplying off diagonals by 2 here since factored into math later
    Omega = {Omega0 * scale, Omega1 * scale, Omega4 * scale, Omega2 * scale, Omega5 * scale, Omega8 * scale};
}

/**
 * @brief Estimates camera pose with gravity prior
 *
 * Given a gravity vector measurement in camera frame corresponding to the world y-axis and 2D-3D feature correspondences,
 * this method estimates the best camera pose (rotation and translation) relative to the 3D world scene. The given axis
 * prior is assumed ground truth and can be collected from several sources (e.g. image vanishing point, device IMU reading,
 * etc.). If the prior corresponds to an axis other than the world y-axis, one can rotate the frame and features to align
 * that axis with world y-axis and transform the solution(s) back to the original space after.
 *
 * The method accepts both point and line features for detections, although points on average lead to better estimates by
 * this method. At minimum, the inputs features must have at least either two points, one point and one line, or three
 * lines. The first two cases are minimal configurations. In those cases, a faster closed-form solution exists, and the
 * algorithm would generally yield two solutions (it may yield less though). Another special case is when all 3D objects
 * (pts3D, lines3D_v, lines3D_p) have their y-coordinate as 0, i.e. the 3D features lie in a plane orthogonal to the
 * world prior axis. This case is a planar configuration, and a special closed-form solution exists in this case as well
 * always yielding two solutions when a solution is possible. This planar solution is particularly fast and accurate and
 * is recommended if your problem can be phrased in this form. All other scenarios are general configurations and would
 * typically yield a single solution.
 *
 * The function is valid for float and double computation (class T). All inputs must have same type. Double is recommended.
 * Flag perform_checks will check small inputs for degeneracy and matrix inversion for full-rank (linear independence
 * of features). Perform_checks will also enable cheirality checking on all point inputs, failing any solution that
 * does not pass the cheirality test on at least half of all points.
 *
 * For more details, see paper "PoseGravity: Pose Estimation from Points and Lines with Axis Prior"
 * https://doi.org/10.48550/arXiv.2405.12646
 *
 * Both 2D and 3D point inputs must be of dimension 3.
 * If the 2D points are points on an image (x, y), the input can be given as (x, y, 1) although normalized points are
 * recommended. 2D line inputs are projective lines l such that l \dot (x, y, 1) = 0 if a point (x, y) lies on the line.
 * 3D line inputs have two parts
 *
 * @param pts2D         Image point features. Must be of dimension 3. If the points are in 2D as (x, y), the input can
 *                      be given as (x, y, 1) although normalized points are recommended.
 *
 * @param pts3D         3D point features corresponding to pts2D.
 *
 * @param lines2D       Image line features. Must be of dimension 3. Defined as projective line l such that a point (x, y)
 *                      lies on the line if l \dot (x, y, 1) = 0. Often the cross product of two 2D points.
 *
 * @param lines3D_v     First part of 3D line features corresponding to lines2D that gives the 3D line's direction
 *                      (usually the difference of two 3D points). Normalized directions are preferred with v_scale set.
 *
 * @param lines3D_p     Second part of 3D line features corresponding to lines2D that gives a 3D point on the line.
 *
 * @param gravity       Vector corresponding to a measurement of the world's y-axis in camera frame. Must have unit norm.
 *
 * @param R1            Placeholder to store first solution's rotation matrix (flattened by rows). If there is only one
 *                      solution, it will be stored here.
 *
 * @param T1            Placeholder to store first solution's translation vector. If there is only one solution, it will be
 *                      stored here.
 *
 * @param R2            Placeholder to store second solution's rotation matrix (flattened by rows).
 *
 * @param T2            Placeholder to store second solution's translation vector.
 *
 * @param cost_val      Optional value to store the objective function's value after completion. Used also for starting cost
 *                      during refinement optimization. Inputting cost_val >= 0 will compute and store the objective function
 *                      value. Otherwise, this will be skipped.
 *
 * @param v_scale       Tunable parameter balancing line loss terms (rotation and translation parts). Only used if line
 *                      features are present. It's effect is dependent on the problem (some are unaffected, some are
 *                      greatly affected). Recommended to be used in planar configurations or if lines3D_v are normalized.
 *                      Otherwise, it's recommended that lines3D_v and lines3D_p have similar magnitudes. Inputting
 *                      v_scale <= 0 will use a default value tuned from experiments.
 *
 * @return              The number of solutions obtained (0, 1, or 2)
 */
template <class T, bool perform_checks=false>
int estimatePoseWithGravity(std::vector<std::array<T, 3>> &pts2D, std::vector<std::array<T, 3>> &pts3D,
                             std::vector<std::array<T, 3>> &lines2D, std::vector<std::array<T, 3>> &lines3D_v,
                             std::vector<std::array<T, 3>> &lines3D_p, std::array<T, 3> gravity, std::array<T, 9> &R1,
                             std::array<T, 3> &T1, std::array<T, 9> &R2, std::array<T, 3> &T2, T &cost_val, T v_scale=-1.) {

    T tol = std::is_same<T, double>::value ? DBL_TOL : FLT_TOL;

    int num_pts = (int) pts2D.size();
    int num_lines = (int) lines2D.size();

    //basic checks, particularly for minimal configurations
    if (perform_checks) {
        //check 2D and 3D correspondence amounts are equal
        if (num_pts != pts3D.size() || num_lines != lines3D_v.size() || num_lines != lines3D_p.size())
            throw std::runtime_error("Number of 2D points != number of 3D points or number of 2D lines != number of 3D lines");

        //check if minimum information provided
        if (num_pts + num_lines < 2 || (num_pts == 0 && num_lines < 3))
            throw std::runtime_error("Need at least one of the following cases: 2 points, 3 lines, or a point and a line");

        //check for degenerate cases
        if (num_pts + num_lines == 2) {
            if (num_pts == 2 && (PoseGravityUtils::Vector::equals(pts2D[0], pts2D[1]) || (fabs(pts3D[0][0] - pts3D[1][0]) < tol && fabs(pts3D[0][1] - pts3D[1][1]) < tol && fabs(pts3D[0][2] - pts3D[1][2]) < tol)))
                throw std::runtime_error("Degenerate configuration, points are coincident");
            else if (num_lines == 1 && num_pts == 1) {
                T pt_norm = sqrt(pts2D[0][0] * pts2D[0][0] + pts2D[0][1] * pts2D[0][1] + pts2D[0][2] * pts2D[0][2]);
                T line_norm = sqrt(lines2D[0][0] * lines2D[0][0] + lines2D[0][1] * lines2D[0][1] + lines2D[0][2] * lines2D[0][2]);
                if (fabs(pts2D[0][0] * lines2D[0][0] + pts2D[0][1] * lines2D[0][1] + pts2D[0][2] * lines2D[0][2]) / (pt_norm * line_norm) < tol)
                    throw std::runtime_error("Degenerate configuration, point and line are coincident");
                else {
                    std::array<T, 3> pt_vec = {pts3D[0][0] - lines3D_p[0][0], pts3D[0][1] - lines3D_p[0][1], pts3D[0][2] - lines3D_p[0][2]};
                    std::array<T, 3> line_vec = lines3D_v[0];
                    if (PoseGravityUtils::Vector::equals(pt_vec, line_vec))
                        throw std::runtime_error("Degenerate configuration, point and line are coincident");
                }
            }
        }
    }
    bool minimal = num_pts + num_lines == 2; //minimal configuration

    //find rotation to align gravity vector with y-axis for mathematical convenience
    std::array<T, 9> R_gravity_alignment = alignGravity(gravity);

    //form matrix sums for optimization
    std::array<T, 9> P; //matrix to recover translation (relabeled S in paper)
    std::array<T, 6> loss_conic; //Omega loss function conic, Ax^2 + Bxy + Cy^2 + Dx + Ey + F (returned conic will be missing 2's on off diagonals)
    generateMatrixSums(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, R_gravity_alignment, P, loss_conic, v_scale, perform_checks);

    std::array<std::array<T, 2>, 4> roots;
    int num_roots = 0;
    T max = std::is_same<T, double>::value ? DBL_MAX : FLT_MAX;

    //Intersection of conics (hyperbola and circle)
    //solve for degenerate conic (linear sum has determinant 0)
    //decompose sum into a pair of lines that go through intersection points
    //Perspectives on Projective Geometry, Richter-Gerbert 2010, pg 188

    //determine if objects in planar configuration (must be in x-z plane orthogonal to world gravity and at y=0)
    bool planar = fabs(loss_conic[3]) < tol && fabs(loss_conic[4]) < tol && fabs(loss_conic[5]) < tol;

    if (planar) {
        //simple closed form solution exists in planar case
        T &a = loss_conic[0], b = T(2) * loss_conic[1], &c = loss_conic[2];
        T ac = a - c;
        if (fabs(b) < tol) {
            if (fabs(ac) >= tol) {
                if (ac < 0)
                    roots[0] = {1., 0.};
                else
                    roots[0] = {0., 1.};
            } else
                return 0;
        } else {
#if FMA
            roots[0] = {-b, ac + sqrt(fma(ac, ac, b * b))};
#else
            roots[0] = {-b, ac + sqrt(ac * ac + b * b)};
#endif
        }
        roots[1] = {-roots[0][0], -roots[0][1]};
        num_roots = 2;

    } else if (minimal) {
        //In minimal configurations, loss function = 0 for feasible problems and loss conic is rank 1.
        //We decompose loss conic into line l (single double line) and intersect directly with circle.
        T &l0 = loss_conic[3], &l1 = loss_conic[4], &l2 = loss_conic[5];

        //criteria if line intersects with circle. If no intersection, oftentimes close miss, so recover closest point on line to circle
        //distance squared of line to origin is c^2 / (a^2 + b^2), check when that is greater than 1 to see if solution exists
        //last column already confirmed to have a nonzero element during planar check, use that as line
#if FMA
        T circle_val = fma(l0, l0, l1 * l1);
#else
        T circle_val = l0 * l0 + l1 * l1;
#endif
        if (fabs(l0) > tol || fabs(l1) > tol) {
            if (l2 * l2 > circle_val) {
                //recover closest solution
                //closest point on line to origin (un-normalized closest point on circle to line) is [-a * c / (a^2 + b^2), -b * c / (a^2 + b^2)]
                //c / (a^2 + b^2) is guaranteed nonnegative factor that can be normalized out later
                roots[0] = {-l0, -l1};
                num_roots = 1;
            } else {
                num_roots = intersectLineCircle(l0, l1, l2, roots, num_roots);
            }
        }
    } else {
        //in overconstrained situations, we solve depressed cubic to find linear combination to make derivative conic (derived from Lagrangian optimization)
        //into a degenerate conic and decompose that conic into two intersecting lines
        T &lc0 = loss_conic[0], lc1 = T(2) * loss_conic[1], &lc2 = loss_conic[2], &lc3 = loss_conic[3], &lc4 = loss_conic[4];
        T dc1 = lc0 - lc2, lc32 = lc3 * lc3, lc42 = lc4 * lc4;
        std::array<T, 9> A;

        //solve depressed cubic from determinant = 0 constraint so conic sum is degenerate
#if FMA
        T a3 = lc42 + lc32 + fma(-lc1, lc1, -dc1 * dc1);
        T a4 = fma(lc1, (lc42 - lc32), 2 * dc1 * lc4 * lc3);
        if (fabs(a4) > tol) {
            T x = solveDepressedCubic(a3, a4);

            //root polishing via Halley's method iteration
            T ys = fma(fma(x, x, a3), x, a4);
            if (fabs(ys) > T(1e3) * tol) {
                //typically converges after one iteration, so skipping for loop
                T yd, denom;
//                int num_iters = 0;
//                for (; num_iters < 10; num_iters++) {
                    yd = fma(T(3) * x, x, a3);
                    denom = fma(yd, yd, T(-3) * ys * x);
//                    if (fabs(denom) < tol) break;
                    if (fabs(denom) >= tol)
                        x -= ys * yd / denom;
//                    ys = fma(fma(x, x, a3), x, a4);
//                    if (fabs(ys) < T(1e3) * tol) break;
//                }
            }
#else
        T a3 = lc42 + lc32 - lc1 * lc1 - dc1 * dc1;
        T a4 = lc1 * (lc42 - lc32) + T(2) * dc1 * lc4 * lc3;

        if (fabs(a4) > tol) {
            T x = solveDepressedCubic(a3, a4);

            //root polishing via Halley's method iteration
            T ys = (x * x + a3) * x + a4;
            if (fabs(ys) > T(1e3) * tol) {
                //typically converges after one iteration, so skipping for loop
                T yd, denom;
//                int num_iters = 0;
//                for (; num_iters < 10; num_iters++) {
                    yd = T(3) * x * x + a3;
                    denom = yd * yd - T(3) * ys * x;
//                    if (fabs(denom) < tol) break;
                    if (fabs(denom) >= tol)
                        x -= ys * yd / denom;
//                    ys = (x * x + a3) * x + a4;
//                    if (fabs(ys) < T(1e3) * tol) break;
//                }
            }
#endif

            //sum conics (hyperbola and circle) to get degenerate conic
            A = {-lc1 + x, dc1, -lc4, dc1, lc1 + x, lc3, -lc4, lc3, -x};
        } else {
            //derivative conic is already degenerate
            A = {-lc1, dc1, -lc4, dc1, lc1, lc3, -lc4, lc3, 0};
        }

        //assume rank 2 matrix, need to decompose to rank 1
        std::array<T, 9> B = PoseGravityUtils::Matrix3x3::adjoint(A);
        T p0 = max, p1, p2;
        //check diagonal for nonzero value to find intersection of degenerate conic's lines
        T b = B[0];
        if (b < -tol) {
            //Note there is error in source reference, sqrt(Bii) should be sqrt(-Bii) as Bii should be <0
            T factor = T(1.0) / sqrt(-b);
            p0 = b * factor, p1 = B[3] * factor, p2 = B[6] * factor;
        } else {
            b = B[4];
            if (b < -tol) {
                T factor = T(1.0) / sqrt(-b);
                p0 = B[1] * factor, p1 = b * factor, p2 = B[7] * factor;
            } else {
                b = B[8];
                if (b < -tol) {
                    T factor = T(1.0) / sqrt(-b);
                    p0 = B[2] * factor, p1 = B[5] * factor, p2 = b * factor;
                }
            }
        }

        if (p0 == max) {
            //if A is rank 1, adjoint is zero matrix
            for (int i = 0; i < 9; i++) {
                if (fabs(A[i]) > tol) {
                    i = i % 3;
                    num_roots = intersectLineCircle(A[i], A[i + 3], A[i + 6], roots, num_roots);
                    break;
                }
            }
        } else {
            //add skew-symmetric matrix formed from degenerate line intersection
            A[1] -= p2, A[2] += p1, A[3] += p2, A[5] -= p0, A[6] -= p1, A[7] += p0;

            //decompose degenerate conic to obtain 2 lines
            for (int i = 0; i < 9; i++) {
                if (fabs(A[i]) > tol) {
                    int row = (i / 3) * 3;
                    i = i % 3;
                    num_roots = intersectLineCircle(A[row], A[row + 1], A[row + 2], roots, num_roots);
                    num_roots += intersectLineCircle(A[i], A[i + 3], A[i + 6], roots, num_roots);
                    break;
                }
            }
        }
    }
    if (!num_roots) return 0;

    //normalize for better stability (and planar case)
    for (int i=0; i<num_roots; i++) {
        auto &x = roots[i][0], &y = roots[i][1];
        T norm = x * x + y * y;
        T norm_diff = fabs(norm - T(1));
        if (norm_diff > tol) {
            norm = T(1) / sqrt(norm);
            x *= norm, y *= norm;
        }
    }

    int num_sol = 0;
    if (minimal || planar) {
        //recover solutions
        num_sol = num_roots;
        recoverPose(roots[0], P, R_gravity_alignment, R1, T1);
        if (num_sol == 2)
            recoverPose(roots[1], P, R_gravity_alignment, R2, T2);

        //store objective function value
        if (cost_val >= T(0)) {
            T &lc0 = loss_conic[0], lc1 = 2 * loss_conic[1], &lc2 = loss_conic[2], lc3 = 2 * loss_conic[3], lc4 = 2 * loss_conic[4];
            T &x = roots[0][0], &y = roots[0][1];
#if FMA
            cost_val = fma(lc0, x * x, fma(lc1, x * y, fma(lc2, y * y, fma(lc3, x, fma(lc4, y, loss_conic[5])))));
#else
            cost_val = lc0 * (x * x) + lc1 * (x * y) + lc2 * (y * y) + lc3 * x + lc4 * y + loss_conic[5];
#endif
        }

    } else {
        //evaluate solutions and choose best one
        T min_loss = max;
        int root_index = -1;
        std::array<T, 4> losses;
        T &lc0 = loss_conic[0], lc1 = 2 * loss_conic[1], &lc2 = loss_conic[2], lc3 = 2 * loss_conic[3], lc4 = 2 * loss_conic[4];
        T scale = std::max(fabs(lc0), std::max(fabs(lc1), std::max(fabs(lc2), std::max(fabs(lc3), fabs(lc4)))));
        scale = scale < tol ? 1 / tol : 1. / scale; //scale for better stability and normalizing loss comparisons later
        for (int i = 0; i < num_roots; i++) {
            T &loss = losses[i];
            T &x = roots[i][0], &y = roots[i][1];
#if FMA
            loss = scale * fma(lc0, x * x, fma(lc1, x * y, fma(lc2, y * y, fma(lc3, x, lc4 * y)))); //conic offset doesn't matter
#else
            loss = scale * (lc0 * (x * x) + lc1 * (x * y) + lc2 * (y * y) + lc3 * x + lc4 * y);  //conic offset doesn't matter
#endif
            if (loss < min_loss) {
                min_loss = loss;
                root_index = i;
            }
        }
        //get solution
        num_sol = 1;
        recoverPose(roots[root_index], P, R_gravity_alignment, R1, T1);

        //in certain situations (e.g. linear dependence, planar configuration, etc), can yield more than one solution in overconstrained problems
        //attempt to find other solution if exists
        std::array<T, 2> root = {0};
        if (num_roots > 2) {
            //check if other minimum has same loss
            std::array<T, 4> loss_idx = {0, 1, 2, 3};
            std::stable_sort(loss_idx.begin(), loss_idx.begin() + num_roots, [&losses](size_t i1, size_t i2) {return losses[i1] < losses[i2];});
            if (losses[loss_idx[1]] - min_loss < tol) {
                root = roots[loss_idx[1]];
                num_sol = 2;
            }
        }
        if (cost_val >= T(0)) cost_val = (min_loss / scale) + loss_conic[5]; //store objective function value

        if (num_sol == 2)
            recoverPose(root, P, R_gravity_alignment, R2, T2);
    }

    //if we have points, we can conduct a cheirality check. Otherwise, with lines only we cannot disambiguate without further assumption
    if (perform_checks && num_pts && num_sol) {
        int c1 = 0, c2 = 0;
        for (int i = 0; i < num_pts; i++) {
            std::array<T, 3> &pt2D = pts2D[i], &pt3D = pts3D[i];
            c1 += cheiralityCheck(pt2D, pt3D, R1, T1);
            if (num_sol == 2)
                c2 += cheiralityCheck(pt2D, pt3D, R2, T2);
        }

        if (num_sol == 1) {
            if (minimal) {
                if (c1 < num_pts) {
                    num_sol = 0;
                    R1 = {0}, T1 = {0};
                }
            } else {
                if (c1 < num_pts / 2) {
                    num_sol = 0;
                    R1 = {0}, T1 = {0};
                }
            }
        } else {
            if (c1 > c2 && c1 >= num_pts / 2) {
                num_sol = 1;
                R2 = {0}, T2 = {0};
            } else if (c2 > c1 && c2 >= num_pts / 2) {
                num_sol = 1;
                R1 = R2, T1 = T2;
                R2 = {0}, T2 = {0};
            } else if (c1 < num_pts / 2 && c2 < num_pts / 2) {
                num_sol = 0;
                R1 = {0}, R2 = {0}, T1 = {0}, T2 = {0};
            }
        }
    }

    return num_sol;
}

//overloaded method for non-minimal and non-planar configurations where there is likely only one solution
template <class T, bool perform_checks=false>
int estimatePoseWithGravity(std::vector<std::array<T, 3>> &pts2D, std::vector<std::array<T, 3>> &pts3D,
                             std::vector<std::array<T, 3>> &lines2D, std::vector<std::array<T, 3>> &lines3D_v,
                             std::vector<std::array<T, 3>> &lines3D_p, std::array<T, 3> gravity, std::array<T, 9> &R1,
                             std::array<T, 3> &T1, T &cost_val, T v_scale=-1.) {
    std::array<T, 9> R2{};
    std::array<T, 3> T2{};
    if (pts2D.size() + lines2D.size() == 2)
        throw std::runtime_error("Minimal configuration detected. Please call method with additional R and T placeholders as minimal configurations can have up to 2 solutions");

    int num_sol = estimatePoseWithGravity<T, perform_checks>(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, gravity, R1, T1, R2, T2, cost_val, v_scale);
    if (num_sol == 2)
        throw std::runtime_error("Configuration resulted in two solutions. Please call method with additional R and T placeholders to return both solutions.");
    return num_sol;
}

////////////////////////////////////Pose Refinement from Initial Estimation

/** @brief Given an initial pose estimate, refines the pose to solve the general pose estimation problem
 *
 * With an initial rotation, translation, and objective function value (can be obtained from
 * estimatePoseWithGravity), this function refines the pose estimate to optimizes over the general objective
 * function (i.e. without axis prior). It achieves this by using Newton-Raphson iterations to find the
 * optimal gravity vector input to the PoseGravity algorithm which results in lowest overall loss. The gravity
 * vector can be generated from the 2D space of spherical coordinates, so we can construct the 2x2 Hessian for
 * quadratic convergence with 7 algorithm evaluations per iteration. However other approaches such as
 * Gauss-Newton or SE(3) manifold optimization techniques can be implemented instead.
 *
 * Experiments show promising results compared to existing general solvers (e.g. SQPnP). The method is able to
 * converge to better values even with moderate detection and input gravity vector noise with compute times
 * ~2-20x faster depending on number of inputs and quality of initial estimation. the converged estimate will be
 * better than initial estimate from PoseGravity when detection noise is low relative to gravity vector
 * measurement noise or number input features is large.
 *
 * This method is only valid for non-minimal PoseGravity configurations (num_pts + num_lines >= 3) and optimizes
 * from a single initialization point. For optimizing two solutions in the planar case, see refinePosePlanar.
 *
 * Both point and line features are accepted in float and double precision (double is recommended). All
 * inputs must have same floating point type (class T). Max_iterations sets the maximum number of optimization
 * iterations to run. Problems typically converge in less than 6 iterations. Input points and lines are assumed
 * to be same as in the problem cost_val corresponds to.
 *
 * @param pts2D         See estimatePoseWithGravity
 *
 * @param pts3D         See estimatePoseWithGravity
 *
 * @param lines2D       See estimatePoseWithGravity
 *
 * @param lines3D_v     See estimatePoseWithGravity
 *
 * @param lines3D_p     See estimatePoseWithGravity
 *
 * @param R             Initial estimate of rotation matrix (flattened by rows)
 *
 * @param Tr            Initial estimate of translation vector
 *
 * @param cost_val      Initial estimate of objective function value. Will be overwritten.
 *
 * @param v_scale       See estimatePoseWithGravity
 *
 * @param tol           Tolerance for iteration convergence
 *
 * @return              flag signaling whether the function was successful in reaching a new pose estimate.
 *                      If true, R and Tr are overwritten with new value.
 */
template<class T, int max_iterations=50>
bool refinePose(std::vector<std::array<T, 3>> &pts2D, std::vector<std::array<T, 3>> &pts3D,
                std::vector<std::array<T, 3>> &lines2D, std::vector<std::array<T, 3>> &lines3D_v,
                std::vector<std::array<T, 3>> &lines3D_p, std::array<T, 9> &R, std::array<T, 3> &Tr,
                T &cost_val, T v_scale=-1., T tol=1e-10) {

    if (cost_val < tol) return true; //nothing to optimize
    if (pts2D.size() + lines2D.size() < 3) return false; //too few parameters to optimize without gravity

    std::array<T, 3> gravity = {R[1], R[4], R[7]};
    //find max value to make sure spherical coordinates avoid gimbal lock
    int max_dir = 0;
    if (fabs(gravity[1]) > fabs(gravity[0]))
        max_dir = 1;
    if (fabs(gravity[2]) > fabs(gravity[max_dir]))
        max_dir = 2;
    T theta, phi;
    bool z_axis = max_dir == 0 || max_dir == 1;
    if (z_axis) {
        //max x or y, choose z-axis
        theta = acos(std::max(std::min(gravity[2], T(1)), T(-1)));
        phi = atan2(gravity[1], gravity[0]);
    } else {
        //max z, choose y-axis
        theta = acos(std::max(std::min(gravity[1], T(1)), T(-1)));
        phi = atan2(gravity[0], gravity[2]);
    }

    T delta = sqrt(tol); //derivative estimation step size
    T inv_delta = T(0.5) / delta, inv_delta2 = T(1) / (delta * delta);
    std::array<std::array<T, 2>, 6> stencil_pts{};
    stencil_pts[0] = {delta, 0}, stencil_pts[1] = {-delta, 0}, stencil_pts[2] = {0, delta}, stencil_pts[3] = {0, -delta};
    stencil_pts[4] = {delta, delta}, stencil_pts[5] = {-delta, -delta};
    std::array<T, 6> cost_vals = {0., 0., 0., 0., 0., 0.};
    T Hxx, Hyy, Hxy;
    std::array<T, 9> R1{}, R2{};
    std::array<T, 3> T1{}, T2{};

    //Newton iterations in two dimensions (spherical coordinates of gravity vector) using 7 point Hessian stencil
    int i = 0;
    for (; i<max_iterations; i++) {
        for (int j=0; j<stencil_pts.size(); j++) {
            T sin_theta = sin(theta + stencil_pts[j][0]);
            if (z_axis) {
                gravity = {sin_theta * cos(phi + stencil_pts[j][1]),
                           sin_theta * sin(phi + stencil_pts[j][1]),
                           cos(theta + stencil_pts[j][0])};
            } else {
                gravity = {sin_theta * sin(phi + stencil_pts[j][1]),
                           cos(theta + stencil_pts[j][0]),
                           sin_theta * cos(phi + stencil_pts[j][1])};
            }
            int num_sol = PoseGravity::estimatePoseWithGravity<T, false>(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, gravity, R1, T1, R2, T2, cost_vals[j], v_scale);

            //something went wrong
            if (num_sol == 0) return false;
        }

        //minimal finite difference method for Hessian and derivative vector
#if FMA
        Hxx = inv_delta2 * fma(T(-2), cost_val, cost_vals[0] + cost_vals[1]);
        Hyy = inv_delta2 * fma(T(-2), cost_val, cost_vals[2] + cost_vals[3]);
        Hxy = T(0.5) * fma(inv_delta2, T(-2) * cost_val + cost_vals[4] + cost_vals[5], -(Hxx + Hyy));
        T det = fma(Hxx, Hyy, Hxy * Hxy);
#else
        Hxx = (cost_vals[0] + cost_vals[1] - 2 * cost) * inv_delta2;
        Hyy = (cost_vals[2] + cost_vals[3] - 2 * cost) * inv_delta2;
        Hxy = T(0.5) * ((cost_vals[4] + cost_vals[5] - 2 * cost) * inv_delta2 - Hxx - Hyy);
        T det = Hxx * Hyy - Hxy * Hxy;
#endif
        if (fabs(det) < tol) return false; //saddle point
        T dx = (cost_vals[0] - cost_vals[1]) * inv_delta;
        T dy = (cost_vals[2] - cost_vals[3]) * inv_delta;

        //calculate best direction to move in
#if FMA
        T theta_dir = fma(Hyy, dx, -Hxy * dy) / det;
        T phi_dir = fma(Hxx, dy, -Hxy * dx) / det;
        if (fma(theta_dir, theta_dir, phi_dir * phi_dir) < tol * tol) break; //converged
#else
        T theta_dir = (Hyy * dx - Hxy * dy) / det;
        T phi_dir = (Hxx * dy - Hxy * dx) / det;
        if (theta_dir * theta_dir + phi_dir * phi_dir < tol * tol) break; //converged
#endif

        //update values
        theta -= theta_dir;
        phi -= phi_dir;

        //evaluate at new point
        T sin_theta = sin(theta);
        if (z_axis) {
            gravity = {sin_theta * cos(phi), sin_theta * sin(phi), cos(theta)};
        } else {
            gravity = {sin_theta * sin(phi), cos(theta), sin_theta * cos(phi)};
        }
        int num_sol = PoseGravity::estimatePoseWithGravity<T, false>(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, gravity, R1, T1, R2, T2, cost_val, v_scale);
        if (num_sol == 0) return false; //something went wrong
    }
    //store successful solution
    R = R1, Tr = T1;

    return true;
}

//convenience wrapper that applies pose refinement from first solution to second solution in planar cases
template<class T, int max_iterations=50>
bool refinePosePlanar(std::vector<std::array<T, 3>> &pts2D, std::vector<std::array<T, 3>> &pts3D,
                      std::vector<std::array<T, 3>> &lines2D, std::vector<std::array<T, 3>> &lines3D_v,
                      std::vector<std::array<T, 3>> &lines3D_p, std::array<T, 9> &R1, std::array<T, 3> &T1,
                      std::array<T, 9> &R2, std::array<T, 3> &T2, T &cost, T v_scale=-1., T tol=1e-10) {

    constexpr T minus_1 = -1;

    //refine pose
    bool success = PoseGravity::refinePose(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, R1, T1, cost, v_scale);
    if (success) {
        //solutions are related, flip x/z axis of rotation and entire translation
        R2 = R1, T2 = T1;
        for (int j=0; j<9; j++) if (j % 3 != 1) R2[j] *= minus_1;
        T2[0] *= minus_1, T2[1] *= minus_1, T2[2] *= minus_1;
    }
    return success;
}

}

#endif //POSEGRAVITY_POSEGRAVITY_HPP

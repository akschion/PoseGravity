/*************************************************
 * @author Akshay Chandrasekhar
 * @brief sample script for PoseGravity algorithm
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

#include <iostream>
#include "PoseGravity.hpp"

/** Simple example of a problem with two points and one line*/
int main() {

    //2D and 3D objects must be the same size
    //all inputs must be same type (float or double)
    //detections and directions need not be unit norm but their magnitude will weight the correspondence if overconstrained
    std::vector<std::array<double, 3>> pts2D; //2D points, must be size 3 (z=1 if image point)
    std::vector<std::array<double, 3>> pts3D; //3D points
    std::vector<std::array<double, 3>> lines2D; //2D line normals ([a,b,c] for ax + by + c = 0)
    std::vector<std::array<double, 3>> lines3D_v; //3D line direction
    std::vector<std::array<double, 3>> lines3D_p; //3D point on 3D line

    //random sample inputs
    pts2D.push_back({0.550532662442763, 0.0136912471333144, 1});
    pts2D.push_back({-0.793436715050471, 0.654254257598229, 1});
    pts3D.push_back({36.6341025550057, -74.971133872143, 68.2198171039264});
    pts3D.push_back({-51.8937718046653, -42.0408721982793, 18.5303816958825});
    lines2D.push_back({-0.823573562699887, 0.338480752134259, -0.455145435280243});
    lines3D_v.push_back({0.540856976765624, 0.637951153615376, -0.548171557347479});
    lines3D_p.push_back({-3.81891504465349, -5.99964359636223, 1.34539199446023});

    //input gravity vector (measurement of world y-axis in camera frame, usually from IMU)
    std::array<double, 3> gravity = {0.218169533298947, 0.331012748045705, -0.918059156792933};

    //placeholders for rotation and translation solutions
    //rotations are stored as 3x3 matrices flattened
    std::array<double, 9> R1{}, R2{};
    std::array<double, 3> T1{}, T2{};

    //whether to refine pose (estimates solution to general problem)
    //helps account for noise from gravity vector measurement if detection noise is lower
    bool refine = true;
    double cost_val = refine ? 0. : -1.; //skips computing loss value unless to be used in refinement

    //solve
    int num_sol = PoseGravity::estimatePoseWithGravity(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, gravity, R1, T1, R2, T2, cost_val);

    //refine (overconstrained cases only)
    if (refine && num_sol > 0 && pts2D.size() + lines2D.size() >= 3) {
        bool success; //whether optimization converged to a new result
        if (num_sol == 1)
            success = PoseGravity::refinePose(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, R1, T1, cost_val);
        else //most likely planar case if overconstrained inputs and still two solutions
            success = PoseGravity::refinePosePlanar(pts2D, pts3D, lines2D, lines3D_v, lines3D_p, R1, T1, R2, T2, cost_val);
    }

    //print solution(s)
    //for minimal problems (2 points or 1 point 1 line), would expect mostly two solutions
    //for planar problems (y=0 for all 3D objects), would expect two solutions exactly
    //for other problems, would expect one solution mostly
    std::cout << "Number of solutions: " << num_sol << std::endl;
    if (num_sol > 0) {
        std::cout << "Rotation 1:" << std::endl;
        for (int i=0; i<3; i++)
            std::cout << R1[3*i] << ", " << R1[3*i + 1] << ", " << R1[3*i + 2] << std::endl;
        std::cout << "Translation 1: " << T1[0] << ", " << T1[1] << ", " << T1[2] << std::endl;
        if (num_sol > 1) {
            std::cout << "Rotation 2:" << std::endl;
            for (int i=0; i<3; i++)
                std::cout << R2[3*i] << ", " << R2[3*i + 1] << ", " << R2[3*i + 2] << std::endl;
            std::cout << "Translation 2: " << T2[0] << ", " << T2[1] << ", " << T2[2] << std::endl;
        }
    }

    return 0;
}
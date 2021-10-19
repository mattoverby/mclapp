// Copyright Matt Overby 2021.
// Distributed under the MIT License.

#ifndef MCL_HEADLESS_HPP
#define MCL_HEADLESS_HPP 1

#include "Interface.hpp"
#include <functional>

namespace mcl
{

class Headless : public Interface
{
public:
	using Interface::RowMatrixXi;
	using Interface::RowMatrixXd;

    Headless()
    {
        options.animate = true;
    }

    void start() override;

protected:
    // Saves mesh to file using vertex locations
    // from solve_frame(...) callback
    void save_frame_obj(
        const std::string &prefix,
        int frame_idx,
        const RowMatrixXd &X);
};

} // end namespace mcl

#endif

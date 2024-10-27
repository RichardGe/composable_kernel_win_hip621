// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct Layernorm2dBwdGammaBetaPipelineDefaultPolicy
{
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeDyBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M>,
                      sequence<S::WarpPerBlock_N, S::ThreadPerWarp_N>>,
                tuple<sequence<1, 2>, sequence<1, 2>>,
                tuple<sequence<1, 0>, sequence<2, 1>>,
                sequence<1>,
                sequence<0>>{});
    }
    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeMeanBlockTileDistribution()
    {
        using S = typename Problem::BlockShape;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<S::WarpPerBlock_N, S::ThreadPerWarp_N>,
                tuple<sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M>>,
                tuple<sequence<1, 0>, sequence<1, 0>>,
                tuple<sequence<1, 0>, sequence<2, 1>>,
                sequence<1>,
                sequence<0>>{});
    }

    // template <typename Problem>
    // CK_TILE_DEVICE static constexpr auto MakeGammaBetaBlockTileDistribution()
    // {
    //     using S = typename Problem::BlockShape;

    //     return make_static_tile_distribution(
    //         tile_distribution_encoding<
    //             sequence<S::Repeat_M, S::WarpPerBlock_M, S::ThreadPerWarp_M>,
    //             tuple<sequence<S::WarpPerBlock_N, S::ThreadPerWarp_N>>,
    //             tuple<sequence<0, 1>, sequence<0, 1>>,
    //             tuple<sequence<1, 0>, sequence<2, 1>>,
    //             sequence<0>,
    //             sequence<0>>{});
    // }
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return 1;
    }
};
} // namespace ck_tile

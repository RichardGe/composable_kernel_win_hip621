// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_bwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Layernorm2dBwdGammaBetaPipelineDefaultPolicy>
struct Layernorm2dBwdGammaBetaPipeline
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using BetaDataType    = ck_tile::remove_cvref_t<typename Problem::BetaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using MeanDataType    = ck_tile::remove_cvref_t<typename Problem::MeanDataType>;
    using InvStdDataType  = ck_tile::remove_cvref_t<typename Problem::InvStdDataType>;

    static constexpr bool kPadM              = false;
    static constexpr bool kPadN              = Problem::kPadN;

    static constexpr const char* name = []() {
        return "bwd_gamma_beta";
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }
    template <typename XWindow,
              typename MeanWindow,
              typename InvStdWindow,
              typename DGammaWindow,
              typename DBetaWindow>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const XWindow& dy_window_,
                                   const MeanWindow& mean_window_,
                                   const InvStdWindow& inv_std_window_,
                                   DGammaWindow& gamma_window_,
                                   DBetaWindow& beta_window_,
                                   ck_tile::index_t row_size,
                                   void* smem) const
    {
        auto gamma_beta_dist = Policy::template MakeGammaBetaBlockTileDistribution<Problem>();
        auto mean_dist = Policy::template MakeMeanBlockTileDistribution<Problem>();
        auto x_dist = Policy::template MakeXBlockTileDistribution<Problem>();

        const auto x_window = make_tile_window(x_window_, x_dist);
        const auto dy_window = make_tile_window(dy_window_, x_dist);
        const auto mean_window = make_tile_window(mean_window_, mean_dist);
        const auto inv_std_window = make_tile_window(inv_std_window_, mean_dist);
        const auto x_tile  = load_tile(x_window);
        const auto dy_tile  = load_tile(dy_window);
        const auto mean_tile = load_tile(mean_window);
        const auto inv_std_tile = load_tile(inv_std_window);
        
        auto gamma_window = make_tile_window(gamma_window_, gamma_beta_dist);
        auto beta_window = make_tile_window(beta_window_, gamma_beta_dist);
        auto gamma_tile = make_static_distributed_tensor<GammaDataType>(gamma_beta_dist);
        auto beta_tile = make_static_distributed_tensor<BetaDataType>(gamma_beta_dist);
        sweep_tile(x_tile, [&](auto idx) {
            constexpr auto i_idx = make_tuple(idx[number<0>{}]);
            constexpr auto j_idx = make_tuple(idx[number<1>{}]);
            constexpr auto gb_idx = make_tuple(number<0>{}, idx[number<1>{}]);
            auto &gamma = gamma_tile(gb_idx);
            auto &beta = beta_tile(gb_idx);
            const auto x = type_convert<ComputeDataType>(x_tile[idx]);
            const auto dy = type_convert<ComputeDataType>(dy_tile[idx]);
            const auto mean = type_convert<ComputeDataType>(mean_tile[i_idx]);
            const auto inv_std = type_convert<ComputeDataType>(inv_std_tile[i_idx]);
            beta += type_convert<BetaDataType>(dy);
            gamma += type_convert<GammaDataType>(dy * (x - mean) * inv_std);
            // index_t tid = (threadIdx.y * blockDim.x) + threadIdx.x;
            // if(blockIdx.x < 3 && blockIdx.y == 0 && tid < 3) {
            //     printf("bid %d tid %d count %d gb %f %f\n",blockIdx.x, tid, count, type_convert<float>(g), type_convert<float>(b));
            // }
        });
        store_tile(gamma_window, gamma_tile);
        store_tile(beta_window, beta_tile);

    }
};
} // namespace ck_tile

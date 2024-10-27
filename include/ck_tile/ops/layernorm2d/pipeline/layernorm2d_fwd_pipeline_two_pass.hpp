// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/layernorm2d/pipeline/layernorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = Layernorm2dFwdPipelineDefaultPolicy>
struct Layernorm2dFwdPipelineTwoPass
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

    using SXDataType = XDataType;
    using SYDataType = XDataType;

    static constexpr bool kHasGamma   = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kHasBeta    = !std::is_same_v<BetaDataType, ck_tile::null_type>;
    static constexpr bool kSaveMean   = Problem::Traits::kSaveMeanInvStd;
    static constexpr bool kSaveInvStd = Problem::Traits::kSaveMeanInvStd;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM              = false; // TODO - BlockLayernorm2dFwdProblem::kPadM
    static constexpr bool kPadN              = Problem::Traits::kPadN;
    static constexpr auto kFusedAdd          = Problem::Traits::kFusedAdd;
    static constexpr auto kFusedSweep        = Problem::Traits::kFusedSweep;

    static constexpr const char* name = []() {
        if constexpr(kNeedCrossWarpSync)
            return "bpr"; // block per row
        else
            return "wpr"; // warp per row
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename XWindow,
              typename SXWindow,
              typename GammaWindow,
              typename BetaWindow,
              typename YWindow,
              typename SYWindow,
              typename MeanWindow,
              typename InvStdWindow,
              typename Epilogue>
    CK_TILE_DEVICE auto operator()(const XWindow& x_window_,
                                   const SXWindow& sx_window_,
                                   const GammaWindow& gamma_window_,
                                   const BetaWindow& beta_window_,
                                   YWindow& y_window,
                                   const SYWindow& sy_window_,
                                   MeanWindow& mean_window,
                                   InvStdWindow& inv_std_window,
                                   ComputeDataType epsilon,
                                   ck_tile::index_t row_size,
                                   void* smem,
                                   Epilogue) const
    {
        auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        auto beta_window = make_tile_window(
            beta_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        auto sx_window =
            make_tile_window(sx_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        auto sy_window =
            make_tile_window(sy_window_, Policy::template MakeXBlockTileDistribution<Problem>());

        // Problem::BlockShape
        static constexpr index_t Block_N = Problem::BlockShape::Block_N;
        index_t num_n_tile_iteration =
            __builtin_amdgcn_readfirstlane(integer_divide_ceil(row_size, Block_N));

        // total number of count assume current iter have no pad(only last iter has pad)
        constexpr index_t count_per_iter =
            Problem::BlockShape::Repeat_N * Problem::BlockShape::Vector_N;
        const index_t last_iter_n = row_size - (num_n_tile_iteration - 1) * Block_N;

        int cur_count = 0;
        int max_count =
            (num_n_tile_iteration - 1) * count_per_iter +
            block_tile_welford_calculate_max_count<typename Problem::BlockShape>(last_iter_n);
        auto block_welford      = Policy::template GetBlockWelford<Problem>();
        auto block_welford_sync = Policy::template GetBlockWelfordSync<Problem>();
        auto block_welford_cross_warp_sync =
            Policy::template GetBlockWelfordCrossWarpSync<Problem>();

        using XTensorType = decltype(load_tile(x_window));
        auto mean         = block_welford.template MakeMeanVarBlockTile<XTensorType>();
        auto var          = block_welford.template MakeMeanVarBlockTile<XTensorType>();

        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            auto x  = load_tile(x_window);
            auto sx = load_tile(sx_window);

            move_tile_window(x_window, {0, Block_N});
            move_tile_window(sx_window, {0, Block_N});
            if constexpr(kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD_STORE ||
                         kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD)
            {
                sweep_tile(sx, [&](auto idx) {
                    // compute x = sx + x
                    x(idx) = type_convert<SYDataType>(sx(idx)) + type_convert<SYDataType>(x(idx));
                });
                if constexpr(kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD_STORE)
                {
                    store_tile(sy_window, x);
                    move_tile_window(sy_window, {0, Block_N});
                }
            }
            block_welford(x, mean, var, cur_count, max_count);
        }

        block_welford_sync(mean, var, cur_count);
        block_welford_cross_warp_sync(mean, var, cur_count, smem);
        block_tile_welford_post_scale_var(var, cur_count);

        // compute inv-std
        auto inv_std = tile_elementwise_in(
            [&](const auto& v_) {
                return type_convert<ComputeDataType>(1.0f) / (sqrt(v_) + epsilon);
            },
            var);

        if constexpr(kSaveMean)
            store_tile(mean_window, cast_tile<MeanDataType>(mean));
        if constexpr(kSaveInvStd)
            store_tile(inv_std_window, cast_tile<InvStdDataType>(inv_std));

        // reverse read x to reuse cache
        ck_tile::index_t stride_to_right_most_window =
            row_size % Block_N == 0 ? row_size - Block_N : row_size - row_size % Block_N;

        // x_window.foo();
        // gamma_window.foo();
        move_tile_window(x_window, {0, -Block_N});
        move_tile_window(sx_window, {0, -Block_N});
        move_tile_window(gamma_window, {stride_to_right_most_window});
        move_tile_window(beta_window, {stride_to_right_most_window});
        move_tile_window(y_window, {0, stride_to_right_most_window});

        // layernorm computation
        for(int iN = __builtin_amdgcn_readfirstlane(0); iN < num_n_tile_iteration; ++iN)
        {
            auto x  = load_tile(x_window);
            auto sx = load_tile(sx_window);
            if constexpr(kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD_STORE ||
                         kFusedAdd == Layernorm2dFusedAddEnum::PRE_ADD)
            {
                sweep_tile(sx, [&](auto idx) {
                    // compute x = sx + x
                    x(idx) = type_convert<SYDataType>(sx(idx)) + type_convert<SYDataType>(x(idx));
                });
            }
            // load gamma/beta (TODO: support no gamma/beta?)
            const auto gamma = load_tile(gamma_window);
            const auto beta  = load_tile(beta_window);

            auto ln = make_static_distributed_tensor<ComputeDataType>(x.get_tile_distribution());

            sweep_tile(ln, [&, mean_ = mean](auto idx) {
                constexpr auto i_idx = make_tuple(idx[number<0>{}]);
                constexpr auto j_idx = make_tuple(idx[number<1>{}]);

                const auto gamma_ = type_convert<ComputeDataType>(gamma[j_idx]);
                const auto beta_  = type_convert<ComputeDataType>(beta[j_idx]);

                const auto x_ = type_convert<ComputeDataType>(x[idx]);
                auto ln_      = (x_ - mean_[i_idx]) * inv_std[i_idx] * gamma_ + beta_;

                ln(idx) = ln_;
            });

            Epilogue{}(y_window, ln);

            move_tile_window(x_window, {0, -Block_N});
            move_tile_window(sx_window, {0, -Block_N});
            move_tile_window(gamma_window, {-Block_N});
            move_tile_window(beta_window, {-Block_N});
            move_tile_window(y_window, {0, -Block_N});
        }
    }
};
} // namespace ck_tile

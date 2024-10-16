// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/layernorm2d.hpp"
#include <string>

template <typename DataType>
struct LayerNormTypeConfig;

template <>
struct LayerNormTypeConfig<ck_tile::half_t>
{
    using XDataType       = ck_tile::half_t;
    using YDataType       = ck_tile::half_t;
    using GammaDataType   = ck_tile::half_t;
    using BetaDataType    = ck_tile::half_t;
    using MeanDataType    = ck_tile::half_t;
    using InvStdDataType  = ck_tile::half_t;
    using ComputeDataType = float;
};

template <>
struct LayerNormTypeConfig<ck_tile::bf16_t>
{
    using XDataType       = ck_tile::bf16_t;
    using YDataType       = ck_tile::bf16_t;
    using GammaDataType   = ck_tile::bf16_t;
    using BetaDataType    = ck_tile::bf16_t;
    using MeanDataType    = ck_tile::bf16_t;
    using InvStdDataType  = ck_tile::bf16_t;
    using ComputeDataType = float;
};

// runtime args
struct layernorm2d_fwd_args
{
    const void* p_x;
    const void* p_gamma;
    const void* p_beta;
    void* p_y;
    void* p_mean;
    void* p_invStd;
    float epsilon;
    ck_tile::index_t M;
    ck_tile::index_t N;
};

// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename DataType_,
          ck_tile::index_t NRepeat,
          ck_tile::index_t NThread,
          ck_tile::index_t VectorAccessSize,
          bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kTwoPass_>
struct layernorm2d_fwd_traits_
{
    using DataType = ck_tile::remove_cvref_t<DataType_>;

    static constexpr ck_tile::index_t MRepeat = 1;
    static_assert(NThread <= 64, "We only support intra-wave reduction");
    static constexpr ck_tile::index_t WaveNum = NThread / 16;

    using thread_tile = ck_tile::sequence<MRepeat, NRepeat, VectorAccessSize>;
    using warp_tile =
        ck_tile::sequence<MRepeat * 64 / NThread, NRepeat * NThread * VectorAccessSize>;
    using block_tile =
        ck_tile::sequence<MRepeat * WaveNum * 64 / NThread, NRepeat * NThread * VectorAccessSize>;

    using Shape = ck_tile::TileLayernorm2dShape<thread_tile, warp_tile, block_tile>;

    static constexpr bool kPadN           = kPadN_;
    static constexpr bool kSaveMeanInvStd = kSaveMeanInvStd_;
    static constexpr bool kTwoPass        = kTwoPass_;
};

template <typename Traits_>
float layernorm2d_fwd_(const ck_tile::stream_config& s, layernorm2d_fwd_args a);

// This is the public API, will be generated by script
struct layernorm2d_fwd_traits
{
    std::string data_type;
};

float layernorm2d_fwd(layernorm2d_fwd_traits, layernorm2d_fwd_args, const ck_tile::stream_config&);

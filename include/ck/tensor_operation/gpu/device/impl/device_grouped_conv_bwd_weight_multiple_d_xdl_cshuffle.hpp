// SPDX-License-Identifier: MIT
// // Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <iostream>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"

#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight_multiple_d.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_2d.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_bwd_weight.hpp"
#include <ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp>
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_bwd_weight_v3.hpp"

#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseGemmXdlCShuffleV3,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffsetOfBatch,
          index_t NumGroupsToMerge,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3(
            typename GridwiseGemmXdlCShuffleV3::Argument karg,
            [[maybe_unused]] const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
            [[maybe_unused]] const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
            [[maybe_unused]] const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            [[maybe_unused]] const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
            [[maybe_unused]] const index_t num_k_per_block)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx94__))
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.z * NumGroupsToMerge);
    const index_t k_idx = __builtin_amdgcn_readfirstlane(blockIdx.y * num_k_per_block);

    const long_index_t a_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
    const long_index_t b_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

    __shared__ char p_shared[GridwiseGemmXdlCShuffleV3::GetSharedMemoryNumberOfByte()];

    GridwiseGemmXdlCShuffleV3::template Run<AGridDesc_AK0_M_K1,
                                            BGridDesc_BK0_N_K1,
                                            CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                            HasMainKBlockLoop,
                                            CGlobalMemoryDataOperation,
                                            TailNum>(karg.p_a_grid + a_batch_offset,
                                                     karg.p_b_grid + b_batch_offset,
                                                     karg.p_c_grid + e_batch_offset,
                                                     p_shared,
                                                     karg,
                                                     a_grid_desc_ak0_m_ak1,
                                                     b_grid_desc_bk0_n_bk1,
                                                     c_grid_desc_mblock_mperblock_nblock_nperblock,
                                                     k_idx);
#else
    ignore = karg;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename GridwiseGemmXdlCShuffleV3,
          typename AGridDesc_AK0_M_K1,
          typename BGridDesc_BK0_N_K1,
          typename CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
          typename ComputePtrOffsetOfBatch,
          index_t NumGroupsToMerge,
          bool HasMainKBlockLoop,
          InMemoryDataOperationEnum CGlobalMemoryDataOperation,
          index_t MinimumOccupancy = 1,
          TailNumber TailNum       = TailNumber::Full>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds(
            typename GridwiseGemmXdlCShuffleV3::Argument karg,
            [[maybe_unused]] const AGridDesc_AK0_M_K1 a_grid_desc_ak0_m_ak1,
            [[maybe_unused]] const BGridDesc_BK0_N_K1 b_grid_desc_bk0_n_bk1,
            [[maybe_unused]] const CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
                c_grid_desc_mblock_mperblock_nblock_nperblock,
            [[maybe_unused]] const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
            [[maybe_unused]] const index_t num_k_per_block)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    // offset base pointer for each work-group
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.z * NumGroupsToMerge);
    const index_t k_idx = __builtin_amdgcn_readfirstlane(blockIdx.y * num_k_per_block);

    const long_index_t a_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx));
    const long_index_t b_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx));
    const long_index_t e_batch_offset =
        amd_wave_read_first_lane(compute_ptr_offset_of_batch.GetEPtrOffset(g_idx));

    // Pass two lds pointer is the key to tell compiler that ds_read/write
    // operate on different lds chunk at same time without order dependecy
    __shared__ char p_shared_0[GridwiseGemmXdlCShuffleV3::GetSharedMemoryNumberOfByte()];
    __shared__ char p_shared_1[GridwiseGemmXdlCShuffleV3::GetSharedMemoryNumberOfByte()];

    GridwiseGemmXdlCShuffleV3::template Run_2Lds<AGridDesc_AK0_M_K1,
                                                 BGridDesc_BK0_N_K1,
                                                 CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                                                 HasMainKBlockLoop,
                                                 CGlobalMemoryDataOperation,
                                                 TailNum>(
        karg.p_a_grid + a_batch_offset,
        karg.p_b_grid + b_batch_offset,
        karg.p_c_grid + e_batch_offset,
        p_shared_0,
        p_shared_1,
        karg,
        a_grid_desc_ak0_m_ak1,
        b_grid_desc_bk0_n_bk1,
        c_grid_desc_mblock_mperblock_nblock_nperblock,
        k_idx);
#else
    ignore = karg;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename DsLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename DsDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXdl,
          ck::index_t NPerXdl,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CBlockTransferScalarPerVector_NWaveNPerXdl,
          typename ComputeTypeA                       = InDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          index_t NumGroupsToMerge                    = 1>
struct DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle
    : public DeviceGroupedConvBwdWeightMultipleD<NDimSpatial,
                                                 InLayout,
                                                 WeiLayout,
                                                 OutLayout,
                                                 DsLayout,
                                                 InDataType,
                                                 WeiDataType,
                                                 OutDataType,
                                                 DsDataType,
                                                 InElementwiseOperation,
                                                 WeiElementwiseOperation,
                                                 OutElementwiseOperation,
                                                 ComputeTypeA,
                                                 ComputeTypeB>
{

    using DeviceOp = DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle;

    using ADataType = OutDataType;
    using BDataType = InDataType;
    using EDataType = WeiDataType;

    static constexpr index_t NumDTensor = DsLayout::Size();

    using AElementwiseOperation   = OutElementwiseOperation;
    using BElementwiseOperation   = InElementwiseOperation;
    using CDEElementwiseOperation = WeiElementwiseOperation;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static constexpr auto K1Number = Number<K1>{};

    static constexpr GemmSpecialization GemmSpec = GemmSpecialization::Default;
    static constexpr auto conv_to_gemm_transformer =
        TransformConvBwdWeightToGemmV2<NDimSpatial,
                                       MPerBlock,
                                       NPerBlock,
                                       K1Number,
                                       K0PerBlock / K1Number,
                                       NumGroupsToMerge,
                                       ConvBackwardWeightSpecialization>{};

    static constexpr index_t MaxScalarPerVectorFP32 = 4;
    static constexpr index_t WorkspaceInOutScalarPerVector =
        is_same_v<AccDataType, float>
            ? math::min(CBlockTransferScalarPerVector_NWaveNPerXdl, MaxScalarPerVectorFP32)
            : CBlockTransferScalarPerVector_NWaveNPerXdl;

    // Bytes per 32 lds bank: 32 * 4 bytes
    static constexpr auto BankLength = 128;
    static constexpr auto ElePerBank = BankLength / sizeof(ADataType);

    // M1 & M0
    static constexpr auto ABlockLdsM1PerBlock = ElePerBank / K1;
    static constexpr auto ABlockLdsM0PerBlock = MPerBlock / ABlockLdsM1PerBlock;
    static constexpr auto ABlockLdsM1Padding  = 4;

    // N1 & N0
    static constexpr auto BBlockLdsN1PerBlock = ElePerBank / K1;
    static constexpr auto BBlockLdsN0PerBlock = NPerBlock / BBlockLdsN1PerBlock;
    static constexpr auto BBlockLdsN1Padding  = 4;

    template <ck::index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<1>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    template <ck::index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        const ck::index_t dim   = 1;
        const ck::index_t batch = 1;
        const std::array<ck::index_t, NDimSpatial> lengths{1, 1, 1};
        const std::array<ck::index_t, NDimSpatial + 3> strides{1, 1, 1, 1, 1, 1};
        const std::array<ck::index_t, NDimSpatial> params{1, 1, 1};
        return conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<3>(
            dim,
            dim,
            dim,
            lengths,
            lengths,
            lengths,
            strides,
            strides,
            strides,
            params,
            params,
            params,
            params,
            batch);
    }

    using ABCGridDescs = decltype(GetABCGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    using GridwiseGemmXdlCShuffleV3 = GridwiseGemm_xdl_cshuffle_v3<
        tensor_layout::gemm::RowMajor, // this is a type of tensor A -> this s first one matrix with
                                       // a data
        tensor_layout::gemm::ColumnMajor, // this is a type of data major in matrix C -> this a
                                          // weights
        tensor_layout::gemm::RowMajor,    // this is a type of data major in matrix C -> this a lost
                                          // one
        ADataType,                        // this is a type of data
        BDataType,
        AccDataType,
        AccDataType,
        AccDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        GemmSpec,
        BlockSize,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        K1,
        K1,
        MPerXdl,
        NPerXdl,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CBlockTransferScalarPerVector_NWaveNPerXdl,
        BlkGemmPipeSched,
        BlkGemmPipelineVer,
        ComputeTypeA,
        ComputeTypeB>;

    static constexpr auto MakeElementwiseInputSequence()
    {
        return generate_sequence_v2(
            [&](auto) constexpr { return Number<WorkspaceInOutScalarPerVector>{}; },
            Number<NumDTensor + 1>{});
    }

    static constexpr auto GetDsGridPointerTuple()
    {
        return generate_tuple(
            [&](auto i) {
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                return static_cast<const DDataType*>(nullptr);
            },
            Number<NumDTensor>{});
    }

    template <index_t NDim, typename ck::enable_if<NDim == 1, bool>::type = false>
    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_strides)
    {
        return generate_tuple(
            [&](auto i) {
                const index_t K       = ds_g_k_c_xs_lengths[i][I1];
                const index_t C       = ds_g_k_c_xs_lengths[i][I2];
                const index_t X       = ds_g_k_c_xs_lengths[i][I3];
                const index_t CStride = ds_g_k_c_xs_strides[I2];
                const index_t KStride = ds_g_k_c_xs_strides[I1];

                const auto wei_grid_desc = make_naive_tensor_descriptor(
                    make_tuple(K, X * C), make_tuple(KStride, CStride));

                if constexpr(ConvBackwardWeightSpecialization ==
                             device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
                {
                    return wei_grid_desc;
                }
                else
                {
                    const index_t GemmM = K;
                    const index_t GemmN = C * X;
                    const auto PadGemmM = MPerBlock - GemmM % MPerBlock;
                    const auto PadGemmN = NPerBlock - GemmN % NPerBlock;

                    return transform_tensor_descriptor(
                        wei_grid_desc,
                        make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                   make_right_pad_transform(GemmN, PadGemmN)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
            },
            Number<NumDTensor>{});
    }

    template <index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_strides)
    {
        return generate_tuple(
            [&](auto i) {
                const index_t K = ds_g_k_c_xs_lengths[i][I1];
                const index_t C = ds_g_k_c_xs_lengths[i][I2];
                const index_t Y = ds_g_k_c_xs_lengths[i][I3];
                const index_t X = ds_g_k_c_xs_lengths[i][I4];

                const auto wei_grid_desc =
                    conv_to_gemm_transformer.template make_wei_grid_desc<NDim>(
                        K, Y, X, C, ds_g_k_c_xs_strides[i]);

                if constexpr(ConvBackwardWeightSpecialization ==
                             device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
                {
                    return wei_grid_desc;
                }
                else
                {
                    const index_t GemmM = K;
                    const index_t GemmN = C * X * Y;
                    const auto PadGemmM = MPerBlock - GemmM % MPerBlock;
                    const auto PadGemmN = NPerBlock - GemmN % NPerBlock;

                    return transform_tensor_descriptor(
                        wei_grid_desc,
                        make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                   make_right_pad_transform(GemmN, PadGemmN)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
            },
            Number<NumDTensor>{});
    }

    template <index_t NDim, typename ck::enable_if<NDim == 3, bool>::type = false>
    static auto MakeDsGridDescriptor_M_N(
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDim + 3>, NumDTensor>& ds_g_k_c_xs_strides)
    {
        return generate_tuple(
            [&](auto i) {
                const index_t K = ds_g_k_c_xs_lengths[i][I1];
                const index_t C = ds_g_k_c_xs_lengths[i][I2];
                const index_t Z = ds_g_k_c_xs_lengths[i][I3];
                const index_t Y = ds_g_k_c_xs_lengths[i][I4];
                const index_t X = ds_g_k_c_xs_lengths[i][I5];

                const auto wei_grid_desc =
                    conv_to_gemm_transformer.template make_wei_grid_desc<NDim>(
                        K, Z, Y, X, C, ds_g_k_c_xs_strides[i]);

                if constexpr(ConvBackwardWeightSpecialization ==
                             device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
                {
                    return wei_grid_desc;
                }
                else
                {
                    const index_t GemmM = K;
                    const index_t GemmN = C * X * Y * Z;
                    const auto PadGemmM = MPerBlock - GemmM % MPerBlock;
                    const auto PadGemmN = NPerBlock - GemmN % NPerBlock;

                    return transform_tensor_descriptor(
                        wei_grid_desc,
                        make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                   make_right_pad_transform(GemmN, PadGemmN)),
                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                        make_tuple(Sequence<0>{}, Sequence<1>{}));
                }
            },
            Number<NumDTensor>{});
    }

    template <typename ComputePtrOffsetOfBatch>
    static void
    InitElementwiseBatchStrides(const ComputePtrOffsetOfBatch& compute_ptr_offset_of_batch_,
                                std::array<index_t, NumDTensor + I1>& input_batch_strides,
                                std::array<index_t, I1>& output_batch_strides)
    {
        input_batch_strides[I0]  = compute_ptr_offset_of_batch_.BatchStrideC_;
        output_batch_strides[I0] = compute_ptr_offset_of_batch_.BatchStrideC_;

        // input_batch_strides = {C, Ds...}
        static_for<0, NumDTensor, 1>{}([&](auto i) {
            input_batch_strides[i + 1] = compute_ptr_offset_of_batch_.BatchStrideDs_[i];
        });
    }

    using DsGridDesc_M_N     = decltype(MakeDsGridDescriptor_M_N<NDimSpatial>({}, {}));
    using CDGridDesc_M_N     = decltype(concat_tuple(Tuple<CGridDesc_M_N>{}, DsGridDesc_M_N{}));
    using DsGridPointerTuple = decltype(GetDsGridPointerTuple());
    using CDDataTypes   = decltype(concat_tuple(Tuple<const AccDataType*>{}, DsGridPointerTuple{}));
    using EGridDesc_M_N = CGridDesc_M_N;
    static constexpr index_t ClusterLengthMPerBlock =
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(1);
    static constexpr index_t ClusterLengthNPerBlock =
        CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock::At(3);
    using Block2TileMapElementwise = BlockToCTileMap_M00_N0_M01Adapt<MPerBlock, NPerBlock>;

    using GridwiseElementwise =
        GridwiseElementwise<CDGridDesc_M_N,
                            Tuple<EGridDesc_M_N>,
                            CDDataTypes,
                            Tuple<EDataType*>,
                            Block2TileMapElementwise,
                            CDEElementwiseOperation,
                            BlockSize,
                            MPerBlock,
                            NPerBlock,
                            MPerBlock / ClusterLengthMPerBlock,
                            NPerBlock / ClusterLengthNPerBlock,
                            Sequence<0, 1>,
                            decltype(MakeElementwiseInputSequence()),
                            Sequence<CBlockTransferScalarPerVector_NWaveNPerXdl>,
                            I1,
                            I1>;

    // Argument
    using CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock =
        decltype(GridwiseGemmXdlCShuffleV3::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
            CGridDesc_M_N{}, 1, 1));

    // using Block2CTileMap =
    //    decltype(GridwiseGemm::MakeCBlockClusterAdaptor(CGridDesc_M_N{}, 1, 1, 1));

    struct Argument : public BaseArgument
    {
        Argument(
            const InDataType* p_in_grid,
            WeiDataType* p_wei_grid,
            const OutDataType* p_out_grid,
            const std::array<const void*, NumDTensor>& p_ds,
            const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
            const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
            const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
            const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
            const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
            const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
            const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
            const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_strides,
            const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
            const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
            const std::array<ck::index_t, NDimSpatial>& input_left_pads,
            const std::array<ck::index_t, NDimSpatial>& input_right_pads,
            const ck::index_t M01,
            const ck::index_t N01,
            InElementwiseOperation in_element_op,
            WeiElementwiseOperation wei_element_op,
            OutElementwiseOperation out_element_op,
            ck::index_t split_k)
            : p_a_grid_{p_out_grid},
              p_b_grid_{p_in_grid},
              p_ds_grid_{},
              p_e_grid_{p_wei_grid},
              a_grid_desc_kbatch_k0_m_k1_{},
              b_grid_desc_kbatch_k0_n_k1_{},
              ce_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              // block_2_ctile_map_{},
              compute_ptr_offset_of_batch_{},
              M01_{M01},
              N01_{N01},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              cde_element_op_{wei_element_op},
              Conv_G_{b_g_n_c_wis_lengths[0]},
              Conv_N_{b_g_n_c_wis_lengths[1]},
              Conv_K_{e_g_k_c_xs_lengths[1]},
              Conv_C_{b_g_n_c_wis_lengths[2]},
              input_spatial_lengths_{},
              filter_spatial_lengths_{},
              output_spatial_lengths_{},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads},
              k_batch_{split_k}
        {
            constexpr index_t spatial_offset = 3;
            std::copy(begin(b_g_n_c_wis_lengths) + spatial_offset,
                      end(b_g_n_c_wis_lengths),
                      begin(input_spatial_lengths_));
            std::copy(begin(e_g_k_c_xs_lengths) + spatial_offset,
                      end(e_g_k_c_xs_lengths),
                      begin(filter_spatial_lengths_));
            std::copy(begin(a_g_n_k_wos_lengths) + spatial_offset,
                      end(a_g_n_k_wos_lengths),
                      begin(output_spatial_lengths_));

            const auto descs =
                conv_to_gemm_transformer
                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                        Conv_N_,
                        Conv_K_,
                        Conv_C_,
                        input_spatial_lengths_,
                        filter_spatial_lengths_,
                        output_spatial_lengths_,
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        k_batch_);

            static_for<0, NumDTensor, 1>{}([&](auto i) {
                using DLayout   = remove_cvref_t<tuple_element_t<i.value, DsLayout>>;
                using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;

                static_assert(is_same_v<DLayout, WeiLayout>, "Not supported D data layout");

                // D pointer
                p_ds_grid_(i) = static_cast<const DDataType*>(p_ds[i]);
                compute_ptr_offset_of_batch_.BatchStrideDs_(i) = ds_g_k_c_xs_strides[i][0];
            });

            a_grid_desc_kbatch_k0_m_k1_ = descs[I0];
            b_grid_desc_kbatch_k0_n_k1_ = descs[I1];
            ce_grid_desc_m_n_           = descs[I2];

            ds_grid_descs_tuple_ =
                MakeDsGridDescriptor_M_N<NDimSpatial>(ds_g_k_c_xs_lengths, ds_g_k_c_xs_strides);

            // block_2_ctile_map_ =
            //    GridwiseGemm::MakeCBlockClusterAdaptor(ce_grid_desc_m_n_, M01, N01, k_batch_);
            elementwise_block_2_ctile_map_ = Block2TileMapElementwise{
                ce_grid_desc_m_n_.GetLength(I0), ce_grid_desc_m_n_.GetLength(I1)};

            // A/B/C Batch Stride
            compute_ptr_offset_of_batch_.BatchStrideA_ = a_g_n_k_wos_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideB_ = b_g_n_c_wis_strides[0];
            compute_ptr_offset_of_batch_.BatchStrideC_ =
                Conv_K_ * Conv_C_ *
                std::accumulate(begin(filter_spatial_lengths_),
                                end(filter_spatial_lengths_),
                                index_t{1},
                                std::multiplies<>{});

            const index_t GemmM = a_grid_desc_kbatch_k0_m_k1_.GetLength(I1);
            const index_t GemmN = b_grid_desc_kbatch_k0_n_k1_.GetLength(I1);
            // if(GridwiseGemm::CheckValidity(a_grid_desc_kbatch_k0_m_k1_,
            //                               b_grid_desc_kbatch_k0_n_k1_,
            //                               ce_grid_desc_m_n_,
            //                               block_2_ctile_map_))
            //{
            c_grid_desc_mblock_mperblock_nblock_nperblock_ =
                GridwiseGemmXdlCShuffleV3::MakeCGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                    ce_grid_desc_m_n_,
                    GridwiseGemmXdlCShuffleV3::CalculateMBlock(GemmM),
                    GridwiseGemmXdlCShuffleV3::CalculateNBlock(GemmN));
            //}
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            return sizeof(AccDataType) * ce_grid_desc_m_n_.GetElementSpaceSize() * Conv_G_;
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        DsGridPointerTuple p_ds_grid_;
        EDataType* p_e_grid_;

        AGridDesc_K0_M_K1 a_grid_desc_kbatch_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_kbatch_k0_n_k1_;
        CGridDesc_M_N ce_grid_desc_m_n_;
        CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock c_grid_desc_mblock_mperblock_nblock_nperblock_;
        DsGridDesc_M_N ds_grid_descs_tuple_;

        /// Block2CTileMap block_2_ctile_map_;
        Block2TileMapElementwise elementwise_block_2_ctile_map_;

        // for computing batch offset
        ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor> compute_ptr_offset_of_batch_;

        index_t M01_;
        index_t N01_;

        OutElementwiseOperation a_element_op_;
        InElementwiseOperation b_element_op_;
        WeiElementwiseOperation cde_element_op_;

        // for checking IsSupportedArgument()
        const index_t Conv_G_;
        const index_t Conv_N_;
        const index_t Conv_K_;
        const index_t Conv_C_;
        std::array<ck::index_t, NDimSpatial> input_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> output_spatial_lengths_;
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<ck::index_t, NDimSpatial>& input_left_pads_;
        const std::array<ck::index_t, NDimSpatial>& input_right_pads_;
        const index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument& arg)
        {
            std::cout << "arg.a_grid_desc_kbatch_k0_m_k1_{"
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I2) << ", "
                      << arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_kbatch_k0_n_k1_{"
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I0) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I2) << ", "
                      << arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I3) << "}" << std::endl;

            std::cout << "arg.ce_grid_desc_m_n_{" << arg.ce_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.ce_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }

        float RunGemmV3(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {

            const index_t GemmM = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I1); // the size of M
            const index_t GemmN = arg.b_grid_desc_kbatch_k0_n_k1_.GetLength(I1); // the size of N
            const index_t GemmK = arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I0) *
                                  arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(I2); // the size of K

            AccDataType* p_c_grid = type_convert<AccDataType*>(arg.p_workspace_); // Add workspace

            const ADataType* p_a_grid = arg.p_a_grid_; // ADataType is eaual
            const BDataType* p_b_grid = arg.p_b_grid_; // BData type

            // nullptr for output, will be set after workspace set
            typename GridwiseGemmXdlCShuffleV3::Argument gemm_arg{
                p_a_grid, p_b_grid, p_c_grid, GemmM, GemmN, GemmK, I0, I0, I0, arg.k_batch_};

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemmXdlCShuffleV3::CalculateGridSize(
                gemm_arg.M,
                gemm_arg.N,
                gemm_arg.KBatch,
                arg.Conv_G_ /
                    NumGroupsToMerge); // !! TODO: NumGroupsToMerge has not been defined yet

            float ave_time = 0;

            index_t k_grain = gemm_arg.KBatch * K0PerBlock;
            index_t K_split = (gemm_arg.K + k_grain - 1) / k_grain * (K0PerBlock);

            const bool has_main_k_block_loop =
                GridwiseGemmXdlCShuffleV3::CalculateHasMainKBlockLoop(K_split);

            const auto num_k_per_block =
                arg.a_grid_desc_kbatch_k0_m_k1_.GetLength(Number<0>{}) / gemm_arg.KBatch;

            const auto clear_workspace = [&]() {
                hip_check_error(hipMemsetAsync(
                    gemm_arg.p_c_grid, 0, arg.GetWorkspaceSizeBytes(), stream_config.stream_id_));
            };

            const auto Run = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {
                    typename GridwiseGemmXdlCShuffleV3::Argument gemm_arg_ = gemm_arg;
                    ck::utility::RotatingMemWrapper<typename GridwiseGemmXdlCShuffleV3::Argument>
                        rotating_mem(gemm_arg_,
                                     stream_config.rotating_count,
                                     gemm_arg_.M * gemm_arg_.K * sizeof(ADataType),
                                     gemm_arg_.K * gemm_arg_.N * sizeof(BDataType));
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        clear_workspace();
                    };
                    ave_time += ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg_,
                        arg.a_grid_desc_kbatch_k0_m_k1_,
                        arg.b_grid_desc_kbatch_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block);
                }
                else
                {
                    ave_time += launch_and_time_kernel_with_preprocess(
                        stream_config,
                        clear_workspace,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg,
                        arg.a_grid_desc_kbatch_k0_m_k1_,
                        arg.b_grid_desc_kbatch_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block);
                }
            };

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1 ||
                             BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                            GridwiseGemmXdlCShuffleV3,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                            NumGroupsToMerge,
                            true,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                            GridwiseGemmXdlCShuffleV3,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                            NumGroupsToMerge,
                            true,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
                // Tail number could be One to Seven
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2)
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                           TailNumber::One)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     2)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Two)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     3)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     4)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     5)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     6)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Six)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     7)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::AtomicAdd,
                                    minimum_occupancy,
                                    TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                    else
                    {
                        if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                           TailNumber::One)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::One>;
                            Run(kernel);
                        }
                        else if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                                TailNumber::Full)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Full>;
                            Run(kernel);
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     2)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Two)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Two>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     3)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Three)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Three>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     4)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Four)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Four>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     5)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Five)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Five>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     6)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Six)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Six>;
                                Run(kernel);
                            }
                        }

                        if constexpr(GridwiseGemmXdlCShuffleV3::BlockwiseGemmPipe::PrefetchStages >
                                     7)
                        {
                            if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                               TailNumber::Seven)
                            {
                                const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                    GridwiseGemmXdlCShuffleV3,
                                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                    remove_reference_t<
                                        DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                    ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                    NumGroupsToMerge,
                                    true,
                                    InMemoryDataOperationEnum::Set,
                                    minimum_occupancy,
                                    TailNumber::Seven>;
                                Run(kernel);
                            }
                        }
                    }
                }

                // Tail number could be Odd or Even
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v4)
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                           TailNumber::Odd)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                           TailNumber::Odd)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3_2lds<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
                else
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                           TailNumber::Odd)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::AtomicAdd,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                    else
                    {
                        if(GridwiseGemmXdlCShuffleV3::CalculateKBlockLoopTailNum(K_split) ==
                           TailNumber::Odd)
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Odd>;
                            Run(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                                GridwiseGemmXdlCShuffleV3,
                                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                                remove_reference_t<
                                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                                ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                                NumGroupsToMerge,
                                true,
                                InMemoryDataOperationEnum::Set,
                                minimum_occupancy,
                                TailNumber::Even>;
                            Run(kernel);
                        }
                    }
                }
            }
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    if(gemm_arg.KBatch > 1)
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                            GridwiseGemmXdlCShuffleV3,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                            NumGroupsToMerge,
                            false,
                            InMemoryDataOperationEnum::AtomicAdd,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_grouped_conv_bwd_weight_xdl_cshuffle_v3<
                            GridwiseGemmXdlCShuffleV3,
                            remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                            remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                            remove_reference_t<
                                DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                            ComputePtrOffsetOfStridedBatch<I1, I1, NumDTensor>,
                            NumGroupsToMerge,
                            false,
                            InMemoryDataOperationEnum::Set,
                            minimum_occupancy>;
                        Run(kernel);
                    }
                }
            }

            return ave_time;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            float avg_time = 0.0f;
            /*          if(has_main_k0_block_loop)
                      {
                          //   avg_time = launch_gemm_kernel(integral_constant<bool, true>{});
                      }
                      else
                      {
                          // avg_time = launch_gemm_kernel(integral_constant<bool, false>{});
                      }*/
            avg_time += RunGemmV3(arg, stream_config);
            // avg_time += launch_elementwise_kernel();
            return avg_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        if(!ck::is_xdl_supported())
        {
            return false;
        }
        if constexpr(NDimSpatial == 1)
        {
            if constexpr(!is_GNWC_GKXC_GNWK<InLayout, WeiLayout, OutLayout>())
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 2)
        {
            if constexpr(!(is_NHWGC_GKYXC_NHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_GNHWC_GKYXC_GNHWK<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else if constexpr(NDimSpatial == 3)
        {
            if constexpr(!(is_NDHWGC_GKZYXC_NDHWGK<InLayout, WeiLayout, OutLayout>() ||
                           is_GNDHWC_GKZYXC_GNDHWK<InLayout, WeiLayout, OutLayout>()))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        if constexpr(ConvBackwardWeightSpecialization ==
                     ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.filter_spatial_lengths_[i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 2 &&
             arg.Conv_K_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_C_ % CBlockTransferScalarPerVector_NWaveNPerXdl == 0 &&
             arg.Conv_C_ % WorkspaceInOutScalarPerVector == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        return true;
        // return GridwiseGemm::CheckValidity(arg.a_grid_desc_kbatch_k0_m_k1_,
        //                                   arg.b_grid_desc_kbatch_k0_n_k1_,
        //                                   arg.ce_grid_desc_m_n_,
        //                                   arg.block_2_ctile_map_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(
        const InDataType* p_in_grid,
        WeiDataType* p_wei_grid,
        const OutDataType* p_out_grid,
        const std::array<const void*, NumDTensor>& p_ds,
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
        InElementwiseOperation in_element_op,
        WeiElementwiseOperation wei_element_op,
        OutElementwiseOperation out_element_op,
        const ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        p_ds,
                        b_g_n_c_wis_lengths, // input
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths, // weight
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths, // output
                        a_g_n_k_wos_strides,
                        ds_g_k_c_xs_lengths,
                        ds_g_k_c_xs_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        1,
                        1,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_in_grid,
        void* p_wei_grid,
        const void* p_out_grid,
        const std::array<const void*, NumDTensor>& p_ds,
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_lengths,
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>& ds_g_k_c_xs_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
        InElementwiseOperation in_element_op,
        WeiElementwiseOperation wei_element_op,
        OutElementwiseOperation out_element_op,
        const ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          p_ds,
                                          b_g_n_c_wis_lengths, // input
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths, // weight
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths, // output
                                          a_g_n_k_wos_strides,
                                          ds_g_k_c_xs_lengths,
                                          ds_g_k_c_xs_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          1,
                                          1,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << getConvBackwardWeightSpecializationString(ConvBackwardWeightSpecialization) << ", "
            << K1 << ", "
            << MXdlPerWave << ", "
            << NXdlPerWave << ", "
            << ABlockTransferSrcScalarPerVector << ", "
            << ABlockTransferDstScalarPerVector_K1 << ", "
            << BBlockTransferSrcScalarPerVector << ", "
            << BBlockTransferDstScalarPerVector_K1 << ", "
            << CShuffleMXdlPerWavePerShuffle << ", "
            << CShuffleNXdlPerWavePerShuffle << ", "
            << CBlockTransferScalarPerVector_NWaveNPerXdl
            << ">";
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = dynamic_cast<const Argument*>(p_arg);
        if(arg)
        {
            return arg->GetWorkspaceSizeBytes();
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle::Argument structure!");
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck

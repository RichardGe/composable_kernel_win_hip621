// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using FP16 = ck::half_t;
using FP8  = ck::f8_t;
using F32  = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = FP16;
using B0DataType       = ck::pk_i4_t;
using B1DataType       = FP16;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using EDataType        = FP16;

using A0Layout = Row;
using B0Layout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp   = PassThrough;
using BElementOp   = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;

// static constexpr ck::index_t Scale_Block_M = 128;
static constexpr ck::index_t Scale_Block_N = 128;
static constexpr ck::index_t Scale_Block_K = 128;

using DeviceOpInstance = ck::tensor_operation::device::DeviceGemmMultiD_BScale_Xdl_CShuffle_V3
    // clang-format off
         <Row, Col, DsLayout, ELayout,
          A0DataType, B0DataType, B1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType, 
          AElementOp,  BElementOp, CDEElementOp, GemmSpec,
          256, Scale_Block_N, Scale_Block_K,
          128, 128, 128,
        //   16, 16,
          8, 8,
          16,   16,
          4,    4,
        //   S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
        //   S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
          S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
          S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
          1,    2,  S<1, 32, 1, 8>,  S<8, 8, 1>,
        //   ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, FP8>;
          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3>;

[[maybe_unused]] static static int KPerBlock = 128; // need to be aligned to the KPerBlock set in the device kernel above.
// clang-format on

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 1;
    bool time_kernel     = false;

    // GEMM shape
    ck::index_t M = 3840;
    ck::index_t N = 4096;
    ck::index_t K = 4096;

    ck::index_t StrideA = K;
    ck::index_t StrideB = K;
    ck::index_t StrideE = N;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else if(argc == 10)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);

        M = std::stoi(argv[4]);
        N = std::stoi(argv[5]);
        K = std::stoi(argv[6]);

        StrideA = std::stoi(argv[7]);
        StrideB = std::stoi(argv[8]);
        StrideE = std::stoi(argv[9]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        printf("arg4 to 9: M (256x), N(128x), K(32x), StrideA, StrideB, StrideE\n");
        exit(0);
    }

    ck::index_t Scale_Stride_BN = (K + Scale_Block_K - 1) / Scale_Block_K;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    Tensor<A0DataType> a0_m_k(f_host_tensor_descriptor(M, K, StrideA, A0Layout{}));
    Tensor<B0DataType> b0_k_n(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    Tensor<B1DataType> b0_k_n_permute(f_host_tensor_descriptor(K, N, StrideB, B0Layout{}));
    Tensor<B1DataType> b1_k_n(f_host_tensor_descriptor((K + Scale_Block_K - 1) / Scale_Block_K,
                                                       (N + Scale_Block_N - 1) / Scale_Block_N,
                                                       Scale_Stride_BN,
                                                       B0Layout{}));
    Tensor<EDataType> e_m_n_host_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));
    Tensor<EDataType> e_m_n_device_result(f_host_tensor_descriptor(M, N, StrideE, ELayout{}));

    std::cout << "a0_m_k: " << a0_m_k.mDesc << std::endl;
    std::cout << "b0_k_n: " << b0_k_n.mDesc << std::endl;
    std::cout << "b1_k_n: " << b1_k_n.mDesc << std::endl;
    std::cout << "e_m_n: " << e_m_n_host_result.mDesc << std::endl;

#if 1
    switch(init_method)
    {
    case 0: break;
    case 1:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        break;
    case 2:
        a0_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 3:
        a0_m_k.GenerateTensorValue(GeneratorTensor_2<A0DataType>{-2, 2});
        b0_k_n.GenerateTensorValue(GeneratorTensor_2<B0DataType>{-2, 2});
        b1_k_n.GenerateTensorValue(GeneratorTensor_1<B1DataType>{});
        break;
    case 4:
        a0_m_k.GenerateTensorValue(GeneratorTensor_1<A0DataType>{});
        b0_k_n.GenerateTensorValue(GeneratorTensor_1<B0DataType>{});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
        break;
    default:
        a0_m_k.GenerateTensorValue(GeneratorTensor_3<A0DataType>{-0.5, 0.5});
        b0_k_n.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-0.5, 0.5});
        b1_k_n.GenerateTensorValue(GeneratorTensor_3<B1DataType>{0, 1.0});
    }
#endif

    DeviceMem a0_device_buf(sizeof(A0DataType) * a0_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b0_device_buf(sizeof(B0DataType) * b0_k_n.mDesc.GetElementSpaceSize());
    DeviceMem b1_device_buf(sizeof(B1DataType) * b1_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n_device_result.mDesc.GetElementSpaceSize());

    // weight permute
    if constexpr(PermuteB)
    {
        int K1 = KPerBlock;
        int K0 = K / KPerBlock;

        // int K0, N, K1
        for(int j = 0; j < K0; j++)
        {
            for(int i = 0; i < N; i++)
            {
                for(int jj = 0; jj < K1; jj++)
                {
                    b_k_n_permute(j * N * K1 + i * K1 + jj) = b_k_n(i * K + (j * K1 + jj));
                }
            }
        }
    }
    else
    {
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < K; j++)
            {
                b_k_n_permute(i * K + j) = b_k_n(i * K + j);
            }
        }
    }

    // vector pk_i4x4 permute
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < K; j += 8)
        {
            int input[8];

            for(int k = 0; k < 4; k++)
            {
                int i4x2         = b_k_n_permute(j + k * 2, i);
                input[k * 2 + 0] = (i4x2 >> 4) & 0xf;
                input[k * 2 + 1] = (i4x2 >> 0) & 0xf;
            }

            // permute 01234567->20643175
            {
                int hi   = input[2];
                int lo   = input[0];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 0, i) = i4x2;
            }

            {
                int hi   = input[6];
                int lo   = input[4];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 2, i) = i4x2;
            }

            {
                int hi   = input[3];
                int lo   = input[1];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 4, i) = i4x2;
            }

            {
                int hi   = input[7];
                int lo   = input[5];
                int i4x2 = (hi << 4) | lo;

                b_k_n_permute(j + 6, i) = i4x2;
            }
        }
    }

    a0_device_buf.ToDevice(a0_m_k.mData.data());
    b0_device_buf.ToDevice(b0_k_n_permute.mData.data());
    b1_device_buf.ToDevice(b1_k_n.mData.data());
    e_device_buf.ToDevice(e_m_n_device_result.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();

    // do GEMM
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(a0_device_buf.GetDeviceBuffer(),
                                           b0_device_buf.GetDeviceBuffer(),
                                           std::array<const void*, NumDTensor>{},
                                           e_device_buf.GetDeviceBuffer(),
                                           M,
                                           N,
                                           K,
                                           StrideA,
                                           StrideB,
                                           std::array<ck::index_t, NumDTensor>{},
                                           StrideE,
                                           b1_device_buf.GetDeviceBuffer(),
                                           a_element_op,
                                           b_element_op,
                                           cde_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel, 20, 50});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N + sizeof(EDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    e_device_buf.FromDevice(e_m_n_device_result.mData.data());

    if(do_verification)
    {
        Tensor<AccDataType> c_m_n({M, N});
        Tensor<float> a_m_k({M, K});
        Tensor<float> b_k_n({K, N});

        for(int m = 0; m < M; m++)
        {
            for(int k = 0; k < K; k++)
            {
                a_m_k(m, k) = ck::type_convert<float>(a0_m_k(m, k));
            }
        }

        for(int n = 0; n < N; n++)
        {
            for(int k = 0; k < K; k++)
            {
                b_k_n(k, n) = ck::type_convert<float>(quant_b0_k_n(k, n)) *
                              ck::type_convert<float>(b1_k_n(k / Scale_Block_K, n / Scale_Block_N));
            }
        }

        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<float,
                                                                                float,
                                                                                CShuffleDataType,
                                                                                AccDataType,
                                                                                PassThrough,
                                                                                PassThrough,
                                                                                PassThrough>;
        auto ref_gemm               = ReferenceGemmInstance{};
        auto ref_invoker            = ref_gemm.MakeInvoker();

        auto ref_argument =
            ref_gemm.MakeArgument(a_m_k, b_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

        ref_invoker.Run(ref_argument);

#if 1
        for(int m = 0; m < M; ++m)
        {
            for(int n = 0; n < N; ++n)
            {
                e_m_n_host_result(m, n) = ck::type_convert<EDataType>(c_m_n(m, n));
            }
        }
#endif

        e_device_buf.FromDevice(e_m_n_device_result.mData.data());

        return ck::utils::check_err(
                   e_m_n_device_result, e_m_n_host_result, "Error: Incorrect results!", 5e-2, 5e-2)
                   ? 0
                   : 1;
    }

    return 0;
}

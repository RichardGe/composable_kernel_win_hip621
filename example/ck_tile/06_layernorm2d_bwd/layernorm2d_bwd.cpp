#include "ck_tile/host.hpp"
#include "layernorm2d_bwd.hpp"
#include <cstring>

// different threshold for different dtype
template <typename DataType>
auto get_elimit()
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "3328", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t m      = arg_parser.get_int("m");
    ck_tile::index_t n      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");
    if(stride < 0)
        stride = n;
    std::string data_type = arg_parser.get_str("prec");
    int kname             = arg_parser.get_int("kname");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    assert(stride >= n);

    using TypeConfig = LayerNormTypeConfig<DataType>;

    using XDataType     = typename TypeConfig::XDataType;
    using YDataType     = typename TypeConfig::YDataType;
    using GammaDataType = typename TypeConfig::GammaDataType;
    using BetaDataType  = typename TypeConfig::BetaDataType;

    using MeanDataType = typename TypeConfig::MeanDataType;
    using InvStdDataType = typename TypeConfig::InvStdDataType;

    using ComputeDataType = typename TypeConfig::ComputeDataType;

    // host verify
    ck_tile::HostTensor<XDataType> x_host({m, n}, {stride, 1});
    ck_tile::HostTensor<YDataType> dy_host({m, n}, {stride, 1});
    ck_tile::HostTensor<MeanDataType> mean_host({m});
    ck_tile::HostTensor<InvStdDataType> invStd_host({m});

    ck_tile::index_t blockM = layernorm2d_bwd_block_m<XDataType>();
    ck_tile::index_t reduce_m = (m + blockM - 1) / blockM;
    ck_tile::HostTensor<GammaDataType> dgamma_host_dev({reduce_m, n});
    ck_tile::HostTensor<BetaDataType> dbeta_host_dev({reduce_m, n});
    ck_tile::HostTensor<GammaDataType> dgamma_host_ref({reduce_m, n});
    ck_tile::HostTensor<BetaDataType> dbeta_host_ref({reduce_m, n});


    // ck_tile::FillMonotonicSeq<YDataType>{}(dy_host);
    ck_tile::FillUniformDistribution<YDataType>{-.5f, .5f}(dy_host);
    ck_tile::FillUniformDistribution<MeanDataType>{-.5f, .5f}(mean_host);
    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);
    // ck_tile::FillMonotonicSeq<MeanDataType>{}(mean_host);
    ck_tile::FillUniformDistribution<InvStdDataType>{-.5f, .5f}(invStd_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dy_buf(dy_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem mean_buf(mean_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem invStd_buf(invStd_host.get_element_space_size_in_bytes());

    ck_tile::DeviceMem dgamma_buf(dgamma_host_dev.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dbeta_buf(dbeta_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());
    dy_buf.ToDevice(dy_host.data());
    mean_buf.ToDevice(mean_host.data());
    invStd_buf.ToDevice(invStd_host.data());

    std::cout << "[" << data_type << "]"
              << " m:" << m << ", n:" << n << ", stride:" << stride << std::flush;

    layernorm2d_bwd_traits traits{data_type};
    layernorm2d_bwd_args args{x_buf.GetDeviceBuffer(),
                              dy_buf.GetDeviceBuffer(),
                              mean_buf.GetDeviceBuffer(),
                              invStd_buf.GetDeviceBuffer(),
                              dgamma_buf.GetDeviceBuffer(),
                              dbeta_buf.GetDeviceBuffer(),
                              m,
                              n,
                              stride};

    float ave_time = layernorm2d_bwd(
        traits, args, ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    std::size_t num_byte = sizeof(XDataType) * m * n + sizeof(GammaDataType) * n +
                           sizeof(BetaDataType) * n + sizeof(YDataType) * m * n;

    float gb_per_sec = num_byte / 1.E6 / ave_time;
    std::cout << sizeof(ComputeDataType) << ", " << ave_time * 1.E3 << " us, " << gb_per_sec << " GB/s" << std::flush;

    bool pass = true;

    if(do_validation)
    {
        // reference
        ck_tile::reference_layernorm2d_bwd_gamma_part<XDataType,
                                                      GammaDataType,
                                                      BetaDataType,
                                                      ComputeDataType,
                                                      YDataType,
                                                      MeanDataType,
                                                      InvStdDataType>(
            x_host, dy_host, mean_host, invStd_host, dgamma_host_ref, dbeta_host_ref);

        dgamma_buf.FromDevice(dgamma_host_dev.data());
        dbeta_buf.FromDevice(dbeta_host_dev.data());

        auto [rtol, atol] = get_elimit<DataType>();
        pass = ck_tile::check_err(
            dgamma_host_dev, dgamma_host_ref, std::string("GAMMA OUT Error: Incorrect results!"), rtol, atol);
        pass &= ck_tile::check_err(
            dbeta_host_dev, dbeta_host_ref, std::string("BETA OUT Error: Incorrect results!"), rtol, atol);
        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;
    }

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<ck_tile::bf16_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}

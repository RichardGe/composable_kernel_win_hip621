#pragma once

TYPED_TEST(TestCkTileGemmMemPipeline, SmallM)
{
    std::vector<int> Ms{128};
    std::vector<int> Ns{128}; // M K  K N   M N
    std::vector<int> Ks{33};

    for(int M : Ms)
        for(int N : Ns)
            for(int K : Ks)
                this->Run(M, N, K);
}

// TYPED_TEST(TestCkTileGemmMemPipeline, MidLargeM)
// {
//     std::vector<int> Ms{127, 255, 312, 799, 1573};
//     constexpr int N = 1024;
//     constexpr int K = 321;

//     for(int M : Ms)
//         this->Run(M, N, K);
// }

// TODO: Seems like padding is not working!
// Works only when K is a multiple of KPerBlock
// TYPED_TEST(TestCkTileGemmMemPipeline, PaddK)
// {
//     std::vector<int> Ms{1};
//     constexpr int N = 128;
//     constexpr int K = 320;

//     for(int M : Ms)
//         this->Run(M, N, K);
// }

// TYPED_TEST(TestCkTileGemmMemPipeline, PaddKInv)
// {
//     std::vector<int> Ms{1};
//     constexpr int N = 128;
//     constexpr int K = 322;

//     for(int M : Ms)
//         this->Run(M, N, K);
// }

// TYPED_TEST(TestCkTileGemmMemPipeline, PaddKInv2)
// {
//     std::vector<int> Ms{1};
//     constexpr int N = 128;
//     constexpr int K = 346;

//     for(int M : Ms)
//         this->Run(M, N, K);
// }

// TYPED_TEST(TestCkTileGemmMemPipeline, Regular)
// {
//     std::vector<int> Ms{512};
//     constexpr int N = 1024;
//     constexpr int K = 512;

//     for(int M : Ms)
//         this->Run(M, N, K);
// }

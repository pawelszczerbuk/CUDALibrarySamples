/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <functional>

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

typedef cublasStatus_t (*cublasLtCreate_t)(cublasLtHandle_t *);
typedef cublasStatus_t (*cublasLtDestroy_t)(cublasLtHandle_t);
typedef cublasStatus_t (*cublasLtMatmul_t)(cublasLtHandle_t, cublasLtMatmulDesc_t,
                                           const void *, const void *, const cublasLtMatrixLayout_t,
                                           const void *, const cublasLtMatrixLayout_t,
                                           const void *, const void *, const cublasLtMatrixLayout_t,
                                           void *, const cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t*,
                                           void *, size_t, cudaStream_t);
typedef cublasStatus_t (*cublasLtMatmulDescCreate_t)(cublasLtMatmulDesc_t *, cublasComputeType_t, cudaDataType_t);
typedef cublasStatus_t (*cublasLtMatmulDescSetAttribute_t)(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *, size_t);
typedef cublasStatus_t (*cublasLtMatrixLayoutCreate_t)(cublasLtMatrixLayout_t *, cudaDataType_t, uint64_t, uint64_t, int64_t);
typedef cublasStatus_t (*cublasLtMatmulPreferenceCreate_t)(cublasLtMatmulPreference_t *);
typedef cublasStatus_t (*cublasLtMatmulPreferenceSetAttribute_t)(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void *, size_t);
typedef cublasStatus_t (*cublasLtMatmulAlgoGetHeuristic_t)(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t,
                                                            cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                                            cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t*, int*);
typedef cublasStatus_t (*cublasLtMatmulPreferenceDestroy_t)(cublasLtMatmulPreference_t);
typedef cublasStatus_t (*cublasLtMatmulDescDestroy_t)(cublasLtMatmulDesc_t);
typedef cublasStatus_t (*cublasLtMatrixLayoutDestroy_t)(cublasLtMatrixLayout_t);
                                                        

typedef cudaError_t (*cudaMalloc_t)(void **, size_t);
typedef cudaError_t (*cudaFree_t)(void *);
typedef cudaError_t (*cudaStreamCreate_t)(cudaStream_t *);
typedef cudaError_t (*cudaStreamDestroy_t)(cudaStream_t);
typedef cudaError_t (*cudaStreamSynchronize_t)(cudaStream_t);
typedef cudaError_t (*cudaMemcpyAsync_t)(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);
typedef const char* (*cudaGetErrorString_t)(cudaError_t);

extern cublasLtCreate_t _cublasLtCreate;
extern cublasLtDestroy_t _cublasLtDestroy;
extern cublasLtMatmul_t _cublasLtMatmul;
extern cublasLtMatmulDescCreate_t _cublasLtMatmulDescCreate;
extern cublasLtMatmulDescSetAttribute_t _cublasLtMatmulDescSetAttribute;
extern cublasLtMatrixLayoutCreate_t _cublasLtMatrixLayoutCreate;
extern cublasLtMatmulPreferenceCreate_t _cublasLtMatmulPreferenceCreate;
extern cublasLtMatmulPreferenceSetAttribute_t _cublasLtMatmulPreferenceSetAttribute;
extern cublasLtMatmulAlgoGetHeuristic_t _cublasLtMatmulAlgoGetHeuristic;
extern cublasLtMatmulPreferenceDestroy_t _cublasLtMatmulPreferenceDestroy;
extern cublasLtMatmulDescDestroy_t _cublasLtMatmulDescDestroy;
extern cublasLtMatrixLayoutDestroy_t _cublasLtMatrixLayoutDestroy;
extern cudaMalloc_t _cudaMalloc;
extern cudaFree_t _cudaFree;
extern cudaStreamCreate_t _cudaStreamCreate;
extern cudaStreamDestroy_t _cudaStreamDestroy;
extern cudaStreamSynchronize_t _cudaStreamSynchronize;
extern cudaMemcpyAsync_t _cudaMemcpyAsync;
extern cudaGetErrorString_t _cudaGetErrorString;

extern void* cublasLtHandle;
extern void* cudaHandle;


inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("cuda API failed with status %d: %s\n", status, _cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

inline void loadCudaFunctions() {
    cublasLtHandle = dlopen("libcublasLt.so", RTLD_LAZY);
    if (!cublasLtHandle) {
        printf("Failed to load libcublasLt.so: %s\n", dlerror());
        throw std::logic_error("Failed to load libcublasLt.so");
    }

    _cublasLtCreate = (cublasLtCreate_t)dlsym(cublasLtHandle, "cublasLtCreate");
    if (!_cublasLtCreate) {
        printf("Failed to load cublasLtCreate: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtCreate");
    }

    _cublasLtDestroy = (cublasLtDestroy_t)dlsym(cublasLtHandle, "cublasLtDestroy");
    if (!_cublasLtDestroy) {
        printf("Failed to load cublasLtDestroy: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtDestroy");
    }

    _cublasLtMatmul = (cublasLtMatmul_t)dlsym(cublasLtHandle, "cublasLtMatmul");
    if (!_cublasLtMatmul) {
        printf("Failed to load cublasLtMatmul: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatmul");
    }

    _cublasLtMatmulDescCreate = (cublasLtMatmulDescCreate_t)dlsym(cublasLtHandle, "cublasLtMatmulDescCreate");
    if (!_cublasLtMatmulDescCreate) {
        printf("Failed to load cublasLtMatmulDescCreate: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatmulDescCreate");
    }

    _cublasLtMatmulDescSetAttribute = (cublasLtMatmulDescSetAttribute_t)dlsym(cublasLtHandle, "cublasLtMatmulDescSetAttribute");
    if (!_cublasLtMatmulDescSetAttribute) {
        printf("Failed to load cublasLtMatmulDescSetAttribute: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatmulDescSetAttribute");
    }

    _cublasLtMatrixLayoutCreate = (cublasLtMatrixLayoutCreate_t)dlsym(cublasLtHandle, "cublasLtMatrixLayoutCreate");
    if (!_cublasLtMatrixLayoutCreate) {
        printf("Failed to load cublasLtMatrixLayoutCreate: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatrixLayoutCreate");
    }

    _cublasLtMatmulPreferenceCreate = (cublasLtMatmulPreferenceCreate_t)dlsym(cublasLtHandle, "cublasLtMatmulPreferenceCreate");
    if (!_cublasLtMatmulPreferenceCreate) {
        printf("Failed to load cublasLtMatmulPreferenceCreate: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatmulPreferenceCreate");
    }

    _cublasLtMatmulPreferenceSetAttribute = (cublasLtMatmulPreferenceSetAttribute_t)dlsym(cublasLtHandle, "cublasLtMatmulPreferenceSetAttribute");
    if (!_cublasLtMatmulPreferenceSetAttribute) {
        printf("Failed to load cublasLtMatmulPreferenceSetAttribute: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatmulPreferenceSetAttribute");
    }

    _cublasLtMatmulAlgoGetHeuristic = (cublasLtMatmulAlgoGetHeuristic_t)dlsym(cublasLtHandle, "cublasLtMatmulAlgoGetHeuristic");
    if (!_cublasLtMatmulAlgoGetHeuristic) {
        printf("Failed to load cublasLtMatmulAlgoGetHeuristic: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatmulAlgoGetHeuristic");
    }

    _cublasLtMatmulPreferenceDestroy = (cublasLtMatmulPreferenceDestroy_t)dlsym(cublasLtHandle, "cublasLtMatmulPreferenceDestroy");
    if (!_cublasLtMatmulPreferenceDestroy) {
        printf("Failed to load cublasLtMatmulPreferenceDestroy: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatmulPreferenceDestroy");
    }

    _cublasLtMatmulDescDestroy = (cublasLtMatmulDescDestroy_t)dlsym(cublasLtHandle, "cublasLtMatmulDescDestroy");
    if (!_cublasLtMatmulDescDestroy) {
        printf("Failed to load cublasLtMatmulDescDestroy: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatmulDescDestroy");
    }

    _cublasLtMatrixLayoutDestroy = (cublasLtMatrixLayoutDestroy_t)dlsym(cublasLtHandle, "cublasLtMatrixLayoutDestroy");
    if (!_cublasLtMatrixLayoutDestroy) {
        printf("Failed to load cublasLtMatrixLayoutDestroy: %s\n", dlerror());
        throw std::logic_error("Failed to load cublasLtMatrixLayoutDestroy");
    }

    cudaHandle = dlopen("libcudart.so", RTLD_LAZY);
    if (!cudaHandle) {
        printf("Failed to load libcudart.so: %s\n", dlerror());
        throw std::logic_error("Failed to load libcudart.so");
    }

    _cudaMalloc = (cudaMalloc_t)dlsym(cudaHandle, "cudaMalloc");
    if (!_cudaMalloc) {
        printf("Failed to load cudaMalloc: %s\n", dlerror());
        throw std::logic_error("Failed to load cudaMalloc");
    }

    _cudaFree = (cudaFree_t)dlsym(cudaHandle, "cudaFree");
    if (!_cudaFree) {
        printf("Failed to load cudaFree: %s\n", dlerror());
        throw std::logic_error("Failed to load cudaFree");
    }

    _cudaStreamCreate = (cudaStreamCreate_t)dlsym(cudaHandle, "cudaStreamCreate");
    if (!_cudaStreamCreate) {
        printf("Failed to load cudaStreamCreate: %s\n", dlerror());
        throw std::logic_error("Failed to load cudaStreamCreate");
    }

    _cudaStreamDestroy = (cudaStreamDestroy_t)dlsym(cudaHandle, "cudaStreamDestroy");
    if (!_cudaStreamDestroy) {
        printf("Failed to load cudaStreamDestroy: %s\n", dlerror());
        throw std::logic_error("Failed to load cudaStreamDestroy");
    }

    _cudaStreamSynchronize = (cudaStreamSynchronize_t)dlsym(cudaHandle, "cudaStreamSynchronize");
    if (!_cudaStreamSynchronize) {
        printf("Failed to load cudaStreamSynchronize: %s\n", dlerror());
        throw std::logic_error("Failed to load cudaStreamSynchronize");
    }

    _cudaMemcpyAsync = (cudaMemcpyAsync_t)dlsym(cudaHandle, "cudaMemcpyAsync");
    if (!_cudaMemcpyAsync) {
        printf("Failed to load cudaMemcpyAsync: %s\n", dlerror());
        throw std::logic_error("Failed to load cudaMemcpyAsync");
    }

    _cudaGetErrorString = (cudaGetErrorString_t)dlsym(cudaHandle, "cudaGetErrorString");
    if (!_cudaGetErrorString) {
        printf("Failed to load cudaGetErrorString: %s\n", dlerror());
        throw std::logic_error("Failed to load cudaGetErrorString");
    }

}

inline void unloadCudaFunctions() {
    dlclose(cublasLtHandle);
    dlclose(cudaHandle);
}

template <typename InType, typename OutType = InType, typename ComputeType = OutType>
struct TestBench {
    using SampleRunner = std::function<void()>;

    TestBench(int m, int n, int k,
            ComputeType alpha = ComputeType{0.0f}, ComputeType beta = ComputeType{0.0f},
            size_t workspaceSize = 1024 * 1024 * 4, int N = 1,
            ComputeType Ascale = ComputeType{2.0f}, ComputeType Bscale = ComputeType{0.5f},
            ComputeType Cscale = ComputeType{1.0f}, ComputeType Dscale = ComputeType{1.0f}) :
        m(m), n(n), k(k), N(N), alpha(alpha), beta(beta), workspaceSize(workspaceSize), Ahost(m * k * N), Bhost(n * k * N),
        Chost(m * n * N), Dhost(m * n * N), biasHost(m * N), AscaleHost(Ascale), BscaleHost(Bscale), CscaleHost(Cscale), DscaleHost(Dscale) {
        checkCublasStatus(_cublasLtCreate(&ltHandle));
        checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&Adev), m * k * N * sizeof(InType)));
        checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&Bdev), n * k * N  * sizeof(InType)));
        checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&Cdev), m * n * N  * sizeof(__half)));
        checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&Ddev), m * n * N  * sizeof(OutType)));
        checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&biasDev), m * N * sizeof(OutType)));
        checkCudaStatus(_cudaMalloc(&workspace, workspaceSize));
        checkCudaStatus(_cudaStreamCreate(&stream));

        // Currently only fp8 supports per-tensor scaling
        perTensorScalingEnabled = std::is_same<InType, __nv_fp8_e4m3>::value || std::is_same<InType, __nv_fp8_e5m2>::value;

        if (perTensorScalingEnabled) {
            checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&AscaleDev), sizeof(*AscaleDev)));
            checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&BscaleDev), sizeof(*BscaleDev)));
            checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&CscaleDev), sizeof(*CscaleDev)));
            checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&DscaleDev), sizeof(*DscaleDev)));
            checkCudaStatus(_cudaMalloc(reinterpret_cast<void**>(&DamaxDev), sizeof(*DamaxDev)));
        }

        fillData();
    }

    ~TestBench() {
        // checkCublasStatus(_cublasLtDestroy(ltHandle));
        // checkCudaStatus(_cudaFree(Adev));
        // checkCudaStatus(_cudaFree(Bdev));
        // checkCudaStatus(_cudaFree(Cdev));
        // checkCudaStatus(_cudaFree(Ddev));
        // checkCudaStatus(_cudaFree(biasDev));
        // checkCudaStatus(_cudaFree(workspace));
        // if (perTensorScalingEnabled) {
        //     checkCudaStatus(_cudaFree(AscaleDev));
        //     checkCudaStatus(_cudaFree(BscaleDev));
        //     checkCudaStatus(_cudaFree(CscaleDev));
        //     checkCudaStatus(_cudaFree(DscaleDev));
        //     checkCudaStatus(_cudaFree(DamaxDev));
        // }
        // checkCudaStatus(_cudaStreamDestroy(stream));
    }

    void fillData() {
        // for (int i = 0; i < m * k * N; i++) Ahost[i] = InType(i);
        // for (int i = 0; i < n * k * N; i++) Bhost[i] = InType(i);
        // for (int i = 0; i < m * N; i++) biasHost[i] = InType(i + 1);
        for (int i = 0; i < m * k * N; i++) ((uint8_t*)Ahost.data())[i] = rand() % 256;
        for (int i = 0; i < n * k * N; i++) ((uint8_t*)Bhost.data())[i] = rand() % 256;
        for (int i = 0; i < m * n * N; i++) ((__half*)Chost.data())[i] = __float2half_rn(rand() / 256.f);
        for (int i = 0; i < m * N; i++) ((uint8_t*)biasHost.data())[i] = rand() % 256;
    }

    void copyDataToDevice() {
        checkCudaStatus(_cudaMemcpyAsync(Adev, Ahost.data(), Ahost.size() * sizeof(Ahost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(_cudaMemcpyAsync(Bdev, Bhost.data(), Bhost.size() * sizeof(Bhost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(_cudaMemcpyAsync(Cdev, Chost.data(), Chost.size() * sizeof(Chost[0]), cudaMemcpyHostToDevice, stream));
        checkCudaStatus(_cudaMemcpyAsync(biasDev, biasHost.data(), biasHost.size() * sizeof(biasHost[0]), cudaMemcpyHostToDevice, 0));
        if (perTensorScalingEnabled) {
            checkCudaStatus(_cudaMemcpyAsync(AscaleDev, &AscaleHost, sizeof(AscaleHost), cudaMemcpyHostToDevice, 0));
            checkCudaStatus(_cudaMemcpyAsync(BscaleDev, &BscaleHost, sizeof(BscaleHost), cudaMemcpyHostToDevice, 0));
            checkCudaStatus(_cudaMemcpyAsync(CscaleDev, &CscaleHost, sizeof(CscaleHost), cudaMemcpyHostToDevice, 0));
            checkCudaStatus(_cudaMemcpyAsync(DscaleDev, &DscaleHost, sizeof(DscaleHost), cudaMemcpyHostToDevice, 0));
            checkCudaStatus(_cudaMemcpyAsync(DamaxDev, &DamaxHost, sizeof(DamaxHost), cudaMemcpyHostToDevice, 0));
        }
    }

    void copyDataFromDevice() {
        checkCudaStatus(_cudaMemcpyAsync(Dhost.data(), Ddev, Dhost.size() * sizeof(Dhost[0]), cudaMemcpyDeviceToHost, stream));
    }

    void streamSynchronize() {
        checkCudaStatus(_cudaStreamSynchronize(stream));
    }

    void run(const SampleRunner& runSample) {
        copyDataToDevice();

        runSample();

        copyDataFromDevice();
        streamSynchronize();
    }

    bool perTensorScalingEnabled;
    int m, n, k, N;
    ComputeType alpha, beta;
    size_t workspaceSize;
    std::vector<InType> Ahost, Bhost;
    std::vector<OutType> Dhost, biasHost;
    std::vector<__half> Chost;
    void *workspace;
    InType *Adev, *Bdev;
    OutType *Ddev, *biasDev;
    __half *Cdev;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
    ComputeType AscaleHost, BscaleHost, CscaleHost, DscaleHost, DamaxHost;
    ComputeType *AscaleDev, *BscaleDev, *CscaleDev, *DscaleDev, *DamaxDev;
};

template <>
inline void TestBench<__half, __half, float>::fillData() {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i);
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

template <>
inline void TestBench<__half, __half, cuComplex>::fillData() {
    for (int i = 0; i < m * k * N; i++) Ahost[i] = __float2half_rn(i/100.);
    for (int i = 0; i < n * k * N; i++) Bhost[i] = __float2half_rn(i/100.);
    for (int i = 0; i < m * N; i++) biasHost[i] = __float2half_rn(i + 1);
}

/// Sample wrapper executing fp8 matmul with cublasLtMatmul, with addition of per-tensor scaling, amax calculations, and
/// the workspace to support split-K algorithms.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle
void LtFp8Matmul(cublasLtHandle_t ltHandle,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const float *beta, /* host pointer */
                 const float *a_scale, /* device pointer */
                 const __nv_fp8_e4m3 *A,
                 int lda,
                 const float *b_scale, /* device pointer */
                 const __nv_fp8_e4m3 *B,
                 int ldb,
                 const float *c_scale, /* device pointer */
                 __half *C,
                 __nv_fp8_e4m3 *D,
                 int ldc,
                 const float *d_scale, /* device pointer */
                 float *amax_d, /* device pointer */
                 void *workspace,
                 size_t workspaceSize);

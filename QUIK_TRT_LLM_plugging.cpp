/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "QUIKPlugin.h"

#include <cstring>
#include <cuda_fp16.h>
#include <iostream>
#include <string>
#include <numeric>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdio.h>
#include "kernel/int8FusedDequantizeCUDA.h"

#define CUBLAS_CHECK(call)                                                                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS)                                                                           \
        {                                                                                                              \
            fprintf(stderr, "cuBLAS error at %s:%d : %d\n", __FILE__, __LINE__, status);                               \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

void gemm(
    const int8_t * mat1,
    const int8_t * mat2, int *mat3, int m, int n, int k,cublasHandle_t handle, cudaStream_t stream) {
 

  static int64_t _beta = 0;
  static  int64_t _alpha = 1;

  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)mat3;
  auto mat1_ptr = (void*)mat1;
  auto mat2_ptr = (void*)mat2;
    //cublasHandle_t handle; 
   
 

  (cublasGemmEx(
       handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      alpha_ptr,
      mat2_ptr,
      CUDA_R_8I,
      k,
      mat1_ptr,
      CUDA_R_8I,
      k,
      beta_ptr,
      input_ptr,
      CUDA_R_32I,
      n,
      CUBLAS_COMPUTE_32I,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

 

}


void gemmfp16add(
    const half * mat1,
    const half * mat2, half *mat3, int m, int n, int k,cublasHandle_t handle, cudaStream_t stream) {
 

  static float _beta = 1.0;
  static  float _alpha = 1.0;

  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)mat3;
  auto mat1_ptr = (void*)mat1;
  auto mat2_ptr = (void*)mat2;
 
 
    
  (cublasGemmEx(
       handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      alpha_ptr,
      mat2_ptr,
      CUDA_R_16F,
      k,
      mat1_ptr,
      CUDA_R_16F,
      k,
      beta_ptr,
      input_ptr,
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

 

}

void gemmfp16(
    const half * mat1,
    const half * mat2, half *mat3, int m, int n, int k, cublasHandle_t handle, cudaStream_t stream) {
 

  static float _beta = 0.0;
  static  float _alpha = 1.0;

  auto beta_ptr = (void*)&_beta;
  auto alpha_ptr = (void*)&_alpha;

  auto input_ptr = (void*)mat3;
  auto mat1_ptr = (void*)mat1;
  auto mat2_ptr = (void*)mat2;

    
  (cublasGemmEx(
       handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      n,
      m,
      k,
      alpha_ptr,
      mat2_ptr,
      CUDA_R_16F,
      k,
      mat1_ptr,
      CUDA_R_16F,
      k,
      beta_ptr,
      input_ptr,
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  

}
 

using namespace nvinfer1;
using quik_for_trtllm::plugin::QUIKPluginCreator;
using quik_for_trtllm::plugin::QUIKPlugin;

static char const* TRITON_FLASH_ATTENTION_PLUGIN_VERSION{"1"};
static char const* TRITON_FLASH_ATTENTION_PLUGIN_NAME{"QUIK"};
PluginFieldCollection QUIKPluginCreator::mFC{};
std::vector<PluginField> QUIKPluginCreator::mPluginAttributes;

namespace quik_for_trtllm::plugin
{

// Write values into buffer
template <typename T>
void writeArg(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void readArg(char const*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

std::uintptr_t constexpr kCudaMemAlign = 128;

int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    if (addr % kCudaMemAlign)
    {
        addr += kCudaMemAlign - addr % kCudaMemAlign;
    }
    return (int8_t*) addr;
}

QUIKPlugin::QUIKPlugin(
    int m, int n, int k)
    : mm(m)
    , mn(n)
    , mk(k)
  
{
}

// Parameterized constructor
QUIKPlugin::QUIKPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    readArg(d, mm);
    readArg(d, mn);
    readArg(d, mk);
 
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* QUIKPlugin::clone() const noexcept
{
    auto* plugin = new QUIKPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs QUIKPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output shape.
    //   output tensor [batchSize, seqLen, mNumHeads, head_size]
    assert(outputIndex == 0);
     int const nbDimsA = inputs[0].nbDims;
        int const nbDimsB = inputs[1].nbDims;
        DimsExprs ret;
        ret.nbDims = nbDimsA;
        for (int ii = 0; ii < nbDimsA - 1; ++ii)
        {
            ret.d[ii] = inputs[0].d[ii];
        }
        ret.d[nbDimsA - 1] = exprBuilder.constant(inputs[1].d[0]->getConstantValue());
        
        return ret;
}

bool QUIKPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{


    switch (pos)
    {

    case 0:
        // activation
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    case 1:
        // weights
        // Weights stored in checkpoint must have int8 type  
        return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    case 2:
        // scales channels
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    case 3:
        // fp weight
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    case 4:

        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;

    case 5:
        // out
        return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
    default:
        // Never should be here
        assert(false);
        return false;
    }


}


void QUIKPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    auto const minM = std::accumulate(in[0].min.d, in[0].min.d + in[0].min.nbDims - 1, 1, std::multiplies<int>());
    auto const maxM = std::accumulate(in[0].max.d, in[0].max.d + in[0].max.nbDims - 1, 1, std::multiplies<int>());

    int const maxK = in[0].max.d[in[0].max.nbDims - 1] ;
    int const maxN = in[1].max.d[0];
    int const minK = in[0].min.d[in[0].min.nbDims - 1] ;
    int const minN = in[1].min.d[0];

    assert(minN == maxN );
    assert(minK == maxK );

 
    //  int8 quant + scale factor + fp16 weight + grand
    m_workspaceMaxSize =    maxM * maxK * sizeof(int8_t) +  maxM * sizeof(half) 
    +  maxK * maxN * sizeof(half)  ;


}

size_t QUIKPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{

    return m_workspaceMaxSize;
}





int QUIKPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, 
    void* workspace,
    cudaStream_t stream)
{
  
   int M = 1;
    for (int ii = 0; ii < inputDesc[0].dims.nbDims - 1; ++ii)
    {
        M *= inputDesc[0].dims.d[ii];
    }
    int K = inputDesc[0].dims.d[inputDesc[0].dims.nbDims - 1];
    int N = inputDesc[1].dims.d[0];
 
 
    int res = 0;
    half * Out = reinterpret_cast<half *>(outputs[0]);
    half* actPtr = reinterpret_cast<half*>(workspace);

    int8_t* int8_out = reinterpret_cast<int8_t*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), 0));
    const size_t bufSize_int8_out = sizeof(int8_t) * (M) *  K  ;

    half* scale_a = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(int8_out), 
    bufSize_int8_out));

    half* fp_activation = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(scale_a), 
     sizeof(half) * (M)  ));


    const half * A = reinterpret_cast<half const*>(inputs[0]);
    const int8_t * W = reinterpret_cast<int8_t const*>(inputs[1]);
    const half * scale_b = reinterpret_cast<half const* >(inputs[2]);
    // outliers
    const half * fp_weight = reinterpret_cast<half const*>(inputs[3]);
    const int * ind  = reinterpret_cast<int const*>(inputs[4]);
    

    const int num_ind = 128;
    ExtractOutliersAndSetToZeros(M, K, A, fp_activation, ind, num_ind, stream);
    cublasSetStream(handle,stream);
    gemmfp16(fp_activation,fp_weight,Out, M, N, num_ind, handle, stream);
    int8quant(M, K, A, int8_out, scale_a, stream);

    int8FusedDequantizeCUDA(int8_out, W, scale_a,
                            scale_b, Out, Out, M, N, K, 
                            reinterpret_cast<char*>(workspace),
                            stream);
    return res;
}

int QUIKPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{

    {
        return enqueueImpl(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }

    return 1;
}

// IPluginV2Ext Methods
nvinfer1::DataType QUIKPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    assert(index == 0);
    return nvinfer1::DataType::kHALF;
}

// IPluginV2 Methods

char const* QUIKPlugin::getPluginType() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_NAME;
}

char const* QUIKPlugin::getPluginVersion() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_VERSION;
}

int QUIKPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int QUIKPlugin::initialize() noexcept
{

    cublasCreate(&handle);
    return 0;
}

void QUIKPlugin::terminate() noexcept
{
}

size_t QUIKPlugin::getSerializationSize() const noexcept
{
    return sizeof(mm) + sizeof(mn) + sizeof(mk) ;
}

void QUIKPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    writeArg(d, mm);
    writeArg(d, mn);
    writeArg(d, mk);

}

// bool QUIKPlugin::supportsFormatCombination(
//     int pos, nvinfer1::PluginTensorDesc const* inOut, 
//     int nbInputs, int nbOutputs) noexcept
// {
//     switch (pos)
//     {
//     case 0:
//         // activation
//         return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
//     case 1:
//         // weights
//         // Weights stored in checkpoint must have int8 type
//         return inOut[pos].type == nvinfer1::DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
//     case 2:
//         // scales channels
//         return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
//     case 3:
//         // scales tokens
//         return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
//     case 4:
//         // out
//         return inOut[pos].type == nvinfer1::DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
//     default:
//         // Never should be here
//         assert(false);
//         return false;
//     }
// }

void QUIKPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void QUIKPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QUIKPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

///////////////

QUIKPluginCreator::QUIKPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("mm", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("mn", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("mk", nullptr, PluginFieldType::kINT32, -1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* QUIKPluginCreator::getPluginName() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_NAME;
}

char const* QUIKPluginCreator::getPluginVersion() const noexcept
{
    return TRITON_FLASH_ATTENTION_PLUGIN_VERSION;
}

PluginFieldCollection const* QUIKPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* QUIKPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int m = 0;
    int n = 0;
    int k = 0;
   
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "m"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            m = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "n"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            n = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }else if (!strcmp(attrName, "k"))
        {
            assert(fields[i].type == PluginFieldType::kINT32);
            k = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }

    }
    try
    {
        auto* obj = new QUIKPlugin(m, n, k);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return nullptr;
}

IPluginV2* QUIKPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call QUIKPlugin::destroy()
    try
    {
        auto* obj = new QUIKPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return nullptr;
}

void QUIKPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* QUIKPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

} 

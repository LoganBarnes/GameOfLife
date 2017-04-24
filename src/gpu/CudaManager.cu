// CudaManager.cu
#include "CudaManager.hpp"

#include <helper_cuda.h>
#include <cuda_runtime.h>

#include <iostream>


namespace
{

bool cudaInitialized = false;

} // namespace



CudaManager::CudaManager( const bool print )
  : print_( print )
{
  if ( !cudaInitialized )
  {
    // use device with highest Gflops/s
    int devID = findCudaDevice( 0, 0, print_ );

    if ( devID < 0 )
    {
      throw std::runtime_error( "No CUDA capable devices found" );
    }
    cudaInitialized = true;
  }
}


CudaManager::~CudaManager( )
{
  if ( cudaInitialized )
  {
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset( );
    cudaInitialized = false;

    if ( print_ )
    {
      std::cout << "CUDA device reset" << std::endl;
    }
  }
}


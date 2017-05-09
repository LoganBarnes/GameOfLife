// GameOfLifeCuda.cu
#include "GameOfLifeCuda.hpp"

#include "GameOfLifeAlgorithm.hpp"
#include "CudaManager.hpp"

#include <helper_cuda.h>
#include <helper_grid.h>

#include <cuda_runtime.h>

#include <stdexcept>


namespace gol
{


__global__
void
propogateKernel(
                const GolBool *dpPrev,
                GolBool       *dpCurr,
                const dim3     dim
                )
{
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if ( idx < dim.z )
  {
    uint x = idx % dim.x;
    uint y = idx / dim.x;

    uint index = y * dim.x + x;

    dpCurr[ index ] = findNeighbors( dpPrev, dim, x, y );
  }
}


struct GameOfLifeCuda::MemberVars
{
  GolBool *dpCurrState;
  GolBool *dpPrevState;

  bool updateSinceGetState;

  MemberVars()
    : dpCurrState( nullptr )
    , dpPrevState( nullptr )
    , updateSinceGetState( true )
  {}
};



///
/// \brief GameOfLifeCuda::GameOfLifeCuda
/// \param initState
/// \param width
///
GameOfLifeCuda::GameOfLifeCuda(
                               std::vector< GolBool >            initState,
                               std::vector< GolBool >::size_type width,
                               std::vector< GolBool >::size_type height
                               )
  : GameOfLife( initState, width, height )
  , cuda_( )
  , m_( new MemberVars )
{
  auto sizeBytes = initState.size( ) * sizeof( GolBool );

  checkCudaErrors( cudaMalloc( &m_->dpCurrState, sizeBytes ) );
  checkCudaErrors( cudaMalloc( &m_->dpPrevState, sizeBytes ) );

  checkCudaErrors( cudaMemcpy(
                              m_->dpCurrState,
                              initState.data( ),
                              sizeBytes,
                              cudaMemcpyHostToDevice
                              ) );
}




GameOfLifeCuda::~GameOfLifeCuda( )
{
  checkCudaErrors( cudaFree( m_->dpCurrState ) );
  checkCudaErrors( cudaFree( m_->dpPrevState ) );
}



///
/// \brief GameOfLifeCuda::propogateState
///
void
GameOfLifeCuda::propogateState( )
{
  std::swap( m_->dpPrevState, m_->dpCurrState );

  dim3 dim(
           static_cast< unsigned >( width_ ),
           static_cast< unsigned >( height_ )
           );
  dim.z = dim.x * dim.y;

  dim3 threads( 128 ); // potentially overwritten by computeGridSize
  dim3 blocks( 1 );    // potentially overwritten by computeGridSize

  computeGridSize( dim.z, threads.x, blocks.x, threads.x );

  propogateKernel << < blocks, threads >> > ( m_->dpPrevState, m_->dpCurrState, dim );

  m_->updateSinceGetState = true;
} // propogateState




///
/// \brief GameOfLifeCuda::getState
/// \return
///
const std::vector< GolBool >&
GameOfLifeCuda::getState( )
{
  if ( m_->updateSinceGetState )
  {
    m_->updateSinceGetState = false;
    cudaDeviceSynchronize( );
    checkCudaErrors( cudaMemcpy(
                                state_.data( ),
                                m_->dpCurrState,
                                state_.size( ) * sizeof( GolBool ),
                                cudaMemcpyDeviceToHost
                                ) );
  }

  return state_;
} // getState



} // namespace gol

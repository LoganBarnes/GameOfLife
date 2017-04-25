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



///
/// \brief The GameOfLifeCuda::GoLCudaImpl class
///
class GameOfLifeCuda::GoLCudaImpl
{
public:

  explicit
  GoLCudaImpl(
              const std::vector< GolBool >      &initState,
              std::vector< GolBool >::size_type width,
              std::vector< GolBool >::size_type height
              );

  ~GoLCudaImpl( );

  void propogateState ( );

  const GolBool *getState ( );

  bool
  updateSinceGetState( ) const { return updateSinceGetState_; }


private:

  GolBool *dpCurrState_;
  GolBool *dpPrevState_;

  std::vector< float >::size_type width_, height_;

  bool updateSinceGetState_;

};



///
/// \brief GameOfLifeCuda::GoLCudaImpl::GoLCudaImpl
/// \param initState
/// \param width
///
GameOfLifeCuda::GoLCudaImpl::GoLCudaImpl(
                                         const std::vector< GolBool >      &initState,
                                         std::vector< GolBool >::size_type width,
                                         std::vector< GolBool >::size_type height
                                         )
  : dpCurrState_( nullptr )
  , dpPrevState_( nullptr )
  , width_( width )
  , height_( height )
  , updateSinceGetState_( true )
{
  auto sizeBytes = initState.size( ) * sizeof( GolBool );

  checkCudaErrors( cudaMalloc( &dpCurrState_, sizeBytes ) );
  checkCudaErrors( cudaMalloc( &dpPrevState_, sizeBytes ) );

  checkCudaErrors( cudaMemcpy(
                              dpCurrState_,
                              initState.data( ),
                              sizeBytes,
                              cudaMemcpyHostToDevice
                              ) );
}



GameOfLifeCuda::GoLCudaImpl::~GoLCudaImpl( )
{
  checkCudaErrors( cudaFree( dpCurrState_ ) );
  checkCudaErrors( cudaFree( dpPrevState_ ) );
}



///
/// \brief GameOfLifeCuda::GoLCudaImpl::propogateState
///
void
GameOfLifeCuda::GoLCudaImpl::propogateState( )
{
  std::swap( dpPrevState_, dpCurrState_ );

  dim3 dim(
           static_cast< unsigned >( width_ ),
           static_cast< unsigned >( height_ )
           );
  dim.z = dim.x * dim.y;

  dim3 threads( 128 ); // potentially overwritten by computeGridSize
  dim3 blocks( 1 );    // potentially overwritten by computeGridSize

  computeGridSize( dim.z, threads.x, blocks.x, threads.x );

  propogateKernel << < blocks, threads >> > ( dpPrevState_, dpCurrState_, dim );

  updateSinceGetState_ = true;
} // propogateState



///
/// \brief GameOfLifeCuda::GoLCudaImpl::getState
/// \return
///
const GolBool*
GameOfLifeCuda::GoLCudaImpl::getState( )
{
  updateSinceGetState_ = false;
  return dpCurrState_;
}



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
  , upImpl_( new GameOfLifeCuda::GoLCudaImpl(
                                             state_,
                                             width,
                                             height
                                             ) )
{}



///
/// \brief GameOfLifeCuda::~GameOfLifeCuda
///
GameOfLifeCuda::~GameOfLifeCuda( )
{}



///
/// \brief GameOfLifeCuda::propogateState
///
void
GameOfLifeCuda::propogateState( )
{
  upImpl_->propogateState( );
}



///
/// \brief GameOfLifeCuda::getState
/// \return
///
const std::vector< GolBool >&
GameOfLifeCuda::getState( )
{
  const bool updateSinceGetState = upImpl_->updateSinceGetState( );

  const GolBool *dpState = upImpl_->getState( );

  if ( updateSinceGetState )
  {
    cudaDeviceSynchronize( );
    checkCudaErrors( cudaMemcpy(
                                state_.data( ),
                                dpState,
                                state_.size( ) * sizeof( GolBool ),
                                cudaMemcpyDeviceToHost
                                ) );
  }

  return state_;
} // getState



} // namespace gol

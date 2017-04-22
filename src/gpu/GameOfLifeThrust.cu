// GameOfLifeThrust.cu
#include "GameOfLifeThrust.hpp"

#include <helper_cuda.h>
#include <helper_grid.h>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <stdexcept>


namespace gol
{


namespace
{

///
/// \brief The CudaPrep struct
///
struct CudaPrep
{

  CudaPrep( )
  {
    // use device with highest Gflops/s
    int devID = findCudaDevice( 0, 0, false );

    if ( devID < 0 )
    {
      throw std::runtime_error( "No CUDA capable devices found" );
    }

    std::cout << "CUDA device initialized" << std::endl;
  }


  ~CudaPrep( )
  {
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset( );
    std::cout << "CUDA device reset" << std::endl;
  }


};

///
/// \brief cudaPrep
///
const CudaPrep cudaPrep;

} // namespace



///
/// \brief propogateState_k
/// \param pPrev
/// \param dim
/// \param x
/// \param y
/// \return
///
__device__
bool
propogateState_k(
                 const bool *pPrev,
                 const dim3  dim,
                 const uint  x,
                 const uint  y
                 )
{
  uint neighbors = 0;

  // find number of living neighbors
  // top row
  uint iy = ( y + dim.y - 1 ) % dim.y;
  uint ix = ( x + dim.x - 1 ) % dim.x;

  neighbors += ( pPrev[ iy * dim.x + ix ] ? 1 : 0 );

  ix         = x;
  neighbors += ( pPrev[ iy * dim.x + ix ] ? 1 : 0 );

  ix         = ( x + 1 ) % dim.x;
  neighbors += ( pPrev[ iy * dim.x + ix ] ? 1 : 0 );

  // middle row
  iy         = y;
  ix         = ( x + dim.x - 1 ) % dim.x;
  neighbors += ( pPrev[ iy * dim.x + ix ] ? 1 : 0 );

  ix         = ( x + 1 ) % dim.x;
  neighbors += ( pPrev[ iy * dim.x + ix ] ? 1 : 0 );

  // bottom row
  iy         = ( y + 1 ) % dim.y;
  ix         = ( x + dim.x - 1 ) % dim.x;
  neighbors += ( pPrev[ iy * dim.x + ix ] ? 1 : 0 );

  ix         = x;
  neighbors += ( pPrev[ iy * dim.x + ix ] ? 1 : 0 );

  ix         = ( x + 1 ) % dim.x;
  neighbors += ( pPrev[ iy * dim.x + ix ] ? 1 : 0 );

  bool state = pPrev[ y * dim.x + x ];

  if ( state && ( neighbors != 2 && neighbors != 3 ) )
  {
    return false;
  }
  else
  if ( !state && neighbors == 3 )
  {
    return true;
  }

  return state;
} // propogateState_k



///
/// \brief The PropogateFunctor struct
///
struct PropogateFunctor
{
  const bool *d_prev;
  const dim3 dim;

  PropogateFunctor(
                   const bool *d_prev_,
                   const dim3  dim_
                   )
    : d_prev( d_prev_ )
    , dim( dim_ )
  {}

  ///
  /// \brief operator ()
  /// \param t
  ///
  template< typename Tuple >
  __device__
  void
  operator()( Tuple t ) const
  {
    // get neighbors
    uint idx = thrust::get< 0 >( t );
    uint x   = idx % dim.x;
    uint y   = idx / dim.x;

    thrust::get< 1 >( t ) = propogateState_k( d_prev, dim, x, y );
  }


};



///
/// \brief The GameOfLifeThrust::GoLThrustImpl class
///
class GameOfLifeThrust::GoLThrustImpl
{
public:

  explicit
  GoLThrustImpl(
                std::vector< bool >            initState,
                std::vector< bool >::size_type width,
                std::vector< bool >::size_type height
                );

  ~GoLThrustImpl( ) = default;

  void propogateState ( );

  const thrust::device_vector< bool > &getState ( );

  bool updateSinceGetState ( ) const { return updateSinceGetState_; }


private:

  thrust::device_vector< bool > dCurrState_;
  thrust::device_vector< bool > dPrevState_;

  std::vector< float >::size_type width_, height_;

  bool updateSinceGetState_;

};



///
/// \brief GameOfLifeThrust::GoLThrustImpl::GoLThrustImpl
/// \param initState
/// \param width
///
GameOfLifeThrust::GoLThrustImpl::GoLThrustImpl(
                                               std::vector< bool >
                                               initState,
                                               std::vector< bool >::size_type width,
                                               std::vector< bool >::size_type height
                                               )
  : dCurrState_( initState )
  , dPrevState_( dCurrState_.size( ) )
  , width_( width )
  , height_( height )
  , updateSinceGetState_( true )
{}



///
/// \brief GameOfLifeThrust::GoLThrustImpl::propogateState
///
void
GameOfLifeThrust::GoLThrustImpl::propogateState( )
{
  dPrevState_.swap( dCurrState_ ); // O(1) just swaps pointers

  dim3 dim( width_, height_ );

  thrust::counting_iterator< uint > first( 0 );
  thrust::counting_iterator< uint > last = first + dCurrState_.size( );

  thrust::for_each(
                   thrust::make_zip_iterator( thrust::make_tuple( first, dCurrState_.begin( ) ) ),
                   thrust::make_zip_iterator( thrust::make_tuple( last, dCurrState_.end( ) ) ),
                   PropogateFunctor( thrust::raw_pointer_cast( dPrevState_.data( ) ), dim )
                   );

  updateSinceGetState_ = true;
}



///
/// \brief GameOfLifeThrust::GoLThrustImpl::getState
/// \return
///
const thrust::device_vector< bool >&
GameOfLifeThrust::GoLThrustImpl::getState( )
{
  updateSinceGetState_ = false;
  return dCurrState_;
}



///
/// \brief GameOfLifeThrust::GameOfLifeThrust
/// \param initState
/// \param width
///
GameOfLifeThrust::GameOfLifeThrust(
                                   std::vector< bool >            initState,
                                   std::vector< bool >::size_type width,
                                   std::vector< bool >::size_type height
                                   )
  : GameOfLife( initState, width, height )
  , upImpl_( new GameOfLifeThrust::GoLThrustImpl(
                                                 initState,
                                                 width,
                                                 height
                                                 ) )
{}



///
/// \brief GameOfLifeThrust::propogateState
///
void
GameOfLifeThrust::propogateState( )
{
  upImpl_->propogateState( );
}



///
/// \brief GameOfLifeThrust::getState
/// \return
///
const std::vector< bool >&
GameOfLifeThrust::getState( )
{
  const thrust::device_vector< bool > &dState = upImpl_->getState( );

  if ( upImpl_->updateSinceGetState( ) )
  {
    cudaDeviceSynchronize( );
    thrust::copy( dState.begin( ), dState.end( ), state_.begin( ) );
  }

  return state_;
}



} // namespace gol

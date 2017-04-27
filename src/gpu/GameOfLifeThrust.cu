// GameOfLifeThrust.cu
#include "GameOfLifeThrust.hpp"

#include "GameOfLifeAlgorithm.hpp"
#include "CudaManager.hpp"

#include <helper_cuda.h>

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

#include <stdexcept>


namespace gol
{


///
/// \brief The PropogateFunctor struct
///
struct PropogateFunctor
{
  const GolBool *d_prev;
  const dim3 dim;

  PropogateFunctor(
                   const GolBool *d_prev_,
                   const dim3     dim_
                   )
    : d_prev( d_prev_ )
    , dim   ( dim_ )
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

    thrust::get< 1 >( t ) = findNeighbors( d_prev, dim, x, y );
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
                const std::vector< GolBool >      &initState,
                std::vector< GolBool >::size_type width,
                std::vector< GolBool >::size_type height
                );

  ~GoLThrustImpl( ) = default;

  void propogateState ( );

  const thrust::device_vector< GolBool > &getState ( );

  bool
  updateSinceGetState( ) const { return updateSinceGetState_; }


private:

  thrust::device_vector< GolBool > dCurrState_;
  thrust::device_vector< GolBool > dPrevState_;

  std::vector< float >::size_type width_, height_;

  bool updateSinceGetState_;

};



///
/// \brief GameOfLifeThrust::GoLThrustImpl::GoLThrustImpl
/// \param initState
/// \param width
///
GameOfLifeThrust::GoLThrustImpl::GoLThrustImpl(
                                               const std::vector< GolBool >      &initState,
                                               std::vector< GolBool >::size_type width,
                                               std::vector< GolBool >::size_type height
                                               )
  : dCurrState_( initState.size( ) )
  , dPrevState_( dCurrState_.size( ) )
  , width_( width )
  , height_( height )
  , updateSinceGetState_( true )
{
  checkCudaErrors( cudaMemcpy(
                              thrust::raw_pointer_cast( dCurrState_.data( ) ),
                              initState.data( ),
                              dCurrState_.size( ) * sizeof( GolBool ),
                              cudaMemcpyHostToDevice
                              ) );
}



///
/// \brief GameOfLifeThrust::GoLThrustImpl::propogateState
///
void
GameOfLifeThrust::GoLThrustImpl::propogateState( )
{
  dPrevState_.swap( dCurrState_ ); // O(1) just swaps pointers

  dim3 dim( static_cast< unsigned >( width_ ), static_cast< unsigned >( height_ ) );

  thrust::counting_iterator< SizeType > first( 0 );
  thrust::counting_iterator< SizeType > last( dCurrState_.size( ) );

  thrust::for_each(
                   thrust::make_zip_iterator( thrust::make_tuple( first, dCurrState_.begin( ) ) ),
                   thrust::make_zip_iterator( thrust::make_tuple( last,  dCurrState_.end( ) ) ),
                   PropogateFunctor( thrust::raw_pointer_cast( dPrevState_.data( ) ), dim )
                   );

  updateSinceGetState_ = true;
} // propogateState



///
/// \brief GameOfLifeThrust::GoLThrustImpl::getState
/// \return
///
const thrust::device_vector< GolBool >&
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
                                   std::vector< GolBool >            initState,
                                   std::vector< GolBool >::size_type width,
                                   std::vector< GolBool >::size_type height
                                   )
  : GameOfLife( initState, width, height )
  , cuda_( )
  , upImpl_( new GameOfLifeThrust::GoLThrustImpl(
                                                 state_,
                                                 width,
                                                 height
                                                 ) )
{}



///
/// \brief GameOfLifeThrust::~GameOfLifeThrust
///
GameOfLifeThrust::~GameOfLifeThrust( )
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
const std::vector< GolBool >&
GameOfLifeThrust::getState( )
{
  const bool updateSinceGetState = upImpl_->updateSinceGetState( );

  const thrust::device_vector< GolBool > &dState = upImpl_->getState( );

  if ( updateSinceGetState )
  {
    cudaDeviceSynchronize( );
    thrust::copy( dState.begin( ), dState.end( ), state_.begin( ) );
  }

  return state_;
}



} // namespace gol

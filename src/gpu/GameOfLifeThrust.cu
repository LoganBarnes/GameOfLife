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
/// \brief The GameOfLifeThrust::MemberVars struct
///
class GameOfLifeThrust::MemberVars
{
  thrust::device_vector< GolBool > dCurrState;
  thrust::device_vector< GolBool > dPrevState;

  bool updateSinceGetState;

  MemberVars( const std::vector< GolBool >::size_type initSize )
    : dCurrState( initSize )
    , dPrevState( initSize )
    , updateSinceGetState( true )
};



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
  , m_( new MemberVars( state_.size() ) )
{
  checkCudaErrors( cudaMemcpy(
                              thrust::raw_pointer_cast( m_->dCurrState.data( ) ),
                              initState.data( ),
                              m_->dCurrState.size( ) * sizeof( GolBool ),
                              cudaMemcpyHostToDevice
                              ) );
}



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
  m_->dPrevState.swap( m_->dCurrState ); // O(1) just swaps pointers

  dim3 dim( static_cast< unsigned >( width_ ), static_cast< unsigned >( height_ ) );

  thrust::counting_iterator< SizeType > first( 0 );
  thrust::counting_iterator< SizeType > last( m_->dCurrState.size( ) );

  thrust::for_each(
                   thrust::make_zip_iterator( thrust::make_tuple( first, m_->dCurrState.begin( ) ) ),
                   thrust::make_zip_iterator( thrust::make_tuple( last,  m_->dCurrState.end( ) ) ),
                   PropogateFunctor( thrust::raw_pointer_cast( m_->dPrevState.data( ) ), dim )
                   );

  m_->updateSinceGetState = true;
} // propogateState




///
/// \brief GameOfLifeThrust::getState
/// \return
///
const std::vector< GolBool >&
GameOfLifeThrust::getState( )
{
  if ( m_->updateSinceGetState )
  {
    m_->updateSinceGetState = false;
    cudaDeviceSynchronize( );
    thrust::copy( m_->dCurrState.begin( ), m_->dCurrState.end( ), state_.begin( ) );
  }

  return state_;
}



} // namespace gol

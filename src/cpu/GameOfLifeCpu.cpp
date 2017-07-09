// GameOfLifeCpu.cpp
#include "GameOfLifeCpu.hpp"

#include "GameOfLifeAlgorithm.hpp"

#include <ctpl_stl.h>

#include <iostream>
#include <cassert>


namespace gol
{

namespace
{

void
propState(
          unsigned y,
          dim3     dim,
          GolBool *pState,
          const GolBool *pPrevState
          )
{
  for ( unsigned x = 0; x < dim.x; ++x )
  {
    pState[ y * dim.x + x ] = findNeighbors( pPrevState, dim, x, y );
  }
} 

}

struct GameOfLifeCpu::CtplStuff
{
  const bool multiThreaded;
  ctpl::thread_pool tp;
  std::vector<std::future<void>> results;

  CtplStuff(
            const bool multiThreaded_,
            std::vector< GolBool >::size_type height
            )
    : multiThreaded( multiThreaded_ )
    , tp( multiThreaded ? std::thread::hardware_concurrency() : 0)
    , results( height )
    {}
};


GameOfLifeCpu::GameOfLifeCpu(
                             std::vector< GolBool >            initState,
                             std::vector< GolBool >::size_type width,
                             std::vector< GolBool >::size_type height,
                             const bool                        multiThreading
                             )
  : GameOfLife( initState, width, height )
  , prevState_( state_.size( ) )
  , m_( std::make_shared< CtplStuff >( multiThreading, height ) )
{
}


void
GameOfLifeCpu::propogateState( )
{
  prevState_.swap( state_ );

  dim3 dim(
           static_cast< unsigned >( width_ ),
           static_cast< unsigned >( height_ )
           );

  if ( m_->multiThreaded )
  {    
    std::vector<std::future<void>> &results = m_->results;
    for ( unsigned y = 0; y < height_; ++y )
    {
      results[y] = m_->tp.push([this, y, &dim](int) {
        propState( y, dim, state_.data(), prevState_.data() );
      });
    }

    for ( unsigned y = 0; y < height_; ++y )
    {
      results[y].wait();
    }
  }
  else
  {
    for ( unsigned y = 0; y < height_; ++y )
    {
      propState( y, dim, state_.data(), prevState_.data() );
    }
  }
} // GameOfLifeCpu::propogateState

} // namespace gol

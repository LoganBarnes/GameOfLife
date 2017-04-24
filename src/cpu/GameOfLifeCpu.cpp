// GameOfLifeCpu.cpp
#include "GameOfLifeCpu.hpp"

#include "Semaphore.hpp"
#include "GameOfLifeAlgorithm.hpp"

#include <iostream>


namespace
{

Semaphore startingSemaphore;
Semaphore finishedSemaphore1;
Semaphore finishedSemaphore2;
//Semaphore finishedSemaphore;

}


namespace gol
{


GameOfLifeCpu::GameOfLifeCpu(
                             std::vector< GolBool >            initState,
                             std::vector< GolBool >::size_type width,
                             std::vector< GolBool >::size_type height,
                             const bool                        multiThreading
                             )
  : GameOfLife( initState, width, height )
  , prevState_( state_.size( ) )
  , threads_( 0 )
  , threadsRunning_( false )
{
  if ( multiThreading )
  {
    _startThreadPool( std::thread::hardware_concurrency( ) );
  }
}



GameOfLifeCpu::~GameOfLifeCpu( )
{
  try
  {
    _killThreads( );
  }
  catch ( const std::exception &e )
  {
    std::cerr << "Caught exception while stopping threadpool"
              << e.what( ) << std::endl;
  }
}



void
GameOfLifeCpu::propogateState( )
{
  prevState_ = state_;

  if ( threads_.empty( ) )
  {
    _propogateState( 0, height_ );
  }
  else
  {
    //
    // allow threads to run
    //
    for ( auto i = 0; i < threads_.size( ); ++i )
    {
      startingSemaphore.notify( );
    }

    //
    // wait for threads to finish
    //
    for ( auto i = 0; i < threads_.size( ); ++i )
    {
      finishedSemaphore1.wait( );
    }

    //
    // allow threads to continue back to start of function
    //
    for ( auto i = 0; i < threads_.size( ); ++i )
    {
      finishedSemaphore2.notify( );
    }
  }
} // GameOfLifeCpu::propogateState



void
GameOfLifeCpu::_propogateState(
                               unsigned rowStart,
                               unsigned rowEnd
                               )
{
  dim3 dim(
           static_cast< unsigned >( width_ ),
           static_cast< unsigned >( height_ )
           );

  for ( auto y = rowStart; y < rowEnd; ++y )
  {
    for ( auto x = 0; x < dim.x; ++x )
    {
      state_[ y * dim.x + x ] = findNeighbors( prevState_.data( ), dim, x, y );
    }
  }
} // GameOfLifeCpu::propogateState



void
GameOfLifeCpu::_propogateStateThreaded(
                                       unsigned rowStart,
                                       unsigned rowEnd
                                       )
{
  while ( true )
  {
    startingSemaphore.wait( );

    //
    // exit variable
    //
    if ( !threadsRunning_ )
    {
      break;
    }

    _propogateState( rowStart, rowEnd );

    finishedSemaphore1.notify( );
    finishedSemaphore2.wait( );
  }
} // GameOfLifeCpu::propogateState



void
GameOfLifeCpu::_startThreadPool( unsigned numThreads )
{
  //
  // already started threadpool or requested 0 threads
  //
  if ( !threads_.empty( ) || numThreads == 0 )
  {
    return;
  }

  // prevent threads from exiting
  threadsRunning_ = true;

  std::cout << "Orig: " << numThreads << std::endl;
  //
  // determine the number of rows each thread should process
  //
  auto rowsPerThread = height_ / numThreads;
  numThreads = height_ / rowsPerThread;

  auto extraRows = height_ - numThreads * rowsPerThread;

  std::cout << "New: " << numThreads << std::endl;
  std::cout << "Extra: " << extraRows << std::endl;

  for ( auto i = 0; i < numThreads; ++i )
  {
    threads_.push_back( std::thread(
                                    &GameOfLifeCpu::_propogateStateThreaded,
                                    this,
                                    i * rowsPerThread,
                                    i * rowsPerThread + rowsPerThread
                                    ) );
  }

  if ( extraRows > 0 )
  {
    threads_.push_back( std::thread(
                                    &GameOfLifeCpu::_propogateStateThreaded,
                                    this,
                                    numThreads * rowsPerThread,
                                    height_
                                    ) );
  }
} // GameOfLifeCpu::_startThreadPool



void
GameOfLifeCpu::_killThreads( )
{
  //
  // set bool that causes threads to exit
  //
  threadsRunning_ = false;

  //
  // Unblock all threads
  //
  for ( auto i = 0; i < threads_.size( ); ++i )
  {
    startingSemaphore.notify( );
  }

  //
  // Wait for all threads to finish
  //
  for ( auto &thread : threads_ )
  {
    thread.join( );
  }

  threads_.clear( );
} // GameOfLifeCpu::_killThreads



} // namespace gol

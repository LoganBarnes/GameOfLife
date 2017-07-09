// GameOfLifeSimpleMain.cpp

#include "cpu/GameOfLifeCpu.hpp"

#include "ProjectConfig.hpp"

#ifndef CI_SERVER
#ifdef CUDA_FOUND
#define GPU
#endif
#endif

#ifdef GPU
#include "gpu/GameOfLifeCuda.hpp"
#include "gpu/GameOfLifeThrust.hpp"
#endif

#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <memory>

void showState(const std::unique_ptr< gol::GameOfLife > &upGame )
{
  auto &state = upGame->getState();
  for ( unsigned r = 0; r < upGame->getHeight(); ++r )
  {
    for ( unsigned c = 0; c < upGame->getWidth(); ++c )
    {
      std::cout << (state[ r * upGame->getWidth() + c ] ? 'O' : ' ' ) << ' ';
    }
    std::cout << '\n';
  }
  std::cout << std::endl;
}

int
main(
     const int    argc,
     const char **argv
     )
{
  try
  {
    //
    // parse arguments
    //
    bool noPrint     = false;
    bool multithread = false;
    bool cuda        = false;
    bool thrust      = false;

    std::vector< GolBool >::size_type w = 10;
    std::vector< GolBool >::size_type h = 10;
    unsigned long long maxIterations    = 100;

    unsigned long long seed = std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );

    std::string wStr( "-w=" );
    std::string hStr( "-h=" );
    std::string sStr( "-s=" );
    std::string iStr( "-i=" );

    for ( int i = 1; i < argc; ++i )
    {
      std::string arg( argv[ i ] );

      noPrint     |= ( arg == "-np" );
      multithread |= ( arg == "--threads" );
      cuda        |= ( arg == "--cuda"   );
      thrust      |= ( arg == "--thrust" );

      if ( arg.size( ) > wStr.size( ) &&
          std::mismatch( wStr.begin( ), wStr.end( ), arg.begin( ) ).first == wStr.end( ) )
      {
        w = std::stoul( arg.substr( wStr.size( ) ) );
      }

      if ( arg.size( ) > hStr.size( ) &&
          std::mismatch( hStr.begin( ), hStr.end( ), arg.begin( ) ).first == hStr.end( ) )
      {
        h = std::stoul( arg.substr( hStr.size( ) ) );
      }

      if ( arg.size( ) > sStr.size( ) &&
          std::mismatch( sStr.begin( ), sStr.end( ), arg.begin( ) ).first == sStr.end( ) )
      {
        seed = std::stoll( arg.substr( sStr.size( ) ) );
      }

      if ( arg.size( ) > iStr.size( ) &&
          std::mismatch( iStr.begin( ), iStr.end( ), arg.begin( ) ).first == iStr.end( ) )
      {
        maxIterations = std::stoll( arg.substr( iStr.size( ) ) );
      }
    }

    std::default_random_engine gen( static_cast< unsigned >( seed ) );
    std::bernoulli_distribution dist;

    std::vector< GolBool > state( w * h );

    auto genLambda = [ &gen, &dist ]( )
    {
      return dist( gen );
    };

    std::generate(
                  std::begin( state ),
                  std::end( state ),
                  genLambda
                  );

    std::unique_ptr< gol::GameOfLife > upGame( new gol::GameOfLifeCpu( state, w, h, multithread ) );

#ifdef GPU

    if ( cuda )
    {
      upGame =
        std::unique_ptr< gol::GameOfLifeCuda >( new gol::GameOfLifeCuda( state, w, h ) );
    }

    if ( thrust )
    {
      upGame =
        std::unique_ptr< gol::GameOfLifeThrust >( new gol::GameOfLifeThrust( state, w, h ) );
    }

#endif

    if ( !noPrint )
    {
      std::cout << "START:\n";
      showState( upGame );
    }

    for ( decltype( maxIterations ) iteration = 0; iteration < maxIterations; ++iteration )
    {
      upGame->propogateState();
    }

    if ( !noPrint )
    {
      std::cout << "END:\n";
      showState( upGame );
    }
  }
  catch ( const std::exception &e )
  {
    std::cerr << "Program failed: " << e.what( ) << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
} // main

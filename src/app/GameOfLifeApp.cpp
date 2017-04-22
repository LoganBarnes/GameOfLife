// Main.cpp
#include "GameOfLifeApp.hpp"
#include "cpu/GameOfLifeCpu.hpp"

#include <iostream>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <chrono>
#include <ncurses.h>



namespace
{

void
renderState(
            const std::vector< bool >           &state,
            const std::vector< bool >::size_type width,
            const std::vector< bool >::size_type height
            )
{
  typedef std::vector< bool >::size_type SizeType;

  SizeType index = 0;

  for ( SizeType c = 0; c < width + 1; ++c )
  {
    mvaddstr( 0, c * 2, "==" );
  }

  for ( SizeType r = 1; r <= height; ++r )
  {
    mvaddstr( r, 0, "|" );

    for ( SizeType c = 0; c < width; ++c )
    {
      mvaddstr( r, c * 2 + 1, state[ index++ ] ? "()" : "  " );
    }

    mvaddstr( r, width * 2 + 1, "|" );
  }

  for ( SizeType c = 0; c < width + 1; ++c )
  {
    mvaddstr( height + 1, c * 2, "==" );
  }

  mvaddstr( 0, 0, "press 'q' to quit" );
  //
  // save all changes to screen
  //
  refresh( );

} // renderState



} // namespace


namespace gol
{


///
/// \brief GameOfLifeApp::exec
/// \param argc
/// \param argv
///
void
GameOfLifeApp::exec(
                    const int    argc,
                    const char **argv
                    )
{
  //
  // parse arguments
  //
  bool runFast  = false;
  bool sameSeed = false;

  std::vector< bool >::size_type w = 10;
  std::vector< bool >::size_type h = 10;
  double propStep                  = 0.0;
  double renderStep                = 0.0;

  auto seed = std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );

  std::string wStr( "-w=" );
  std::string hStr( "-h=" );
  std::string tStr( "-t=" );
  std::string rStr( "-r=" );
  std::string sStr( "-s=" );

  for ( int i = 1; i < argc; ++i )
  {
    std::string arg( argv[ i ] );

    runFast |= ( arg == "-f" );

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

    if ( arg.size( ) > tStr.size( ) &&
        std::mismatch( tStr.begin( ), tStr.end( ), arg.begin( ) ).first == tStr.end( ) )
    {
      propStep = std::stod( arg.substr( tStr.size( ) ) );
    }

    if ( arg.size( ) > rStr.size( ) &&
        std::mismatch( rStr.begin( ), rStr.end( ), arg.begin( ) ).first == rStr.end( ) )
    {
      renderStep = std::stod( arg.substr( rStr.size( ) ) );
    }

    if ( arg.size( ) > sStr.size( ) &&
        std::mismatch( sStr.begin( ), sStr.end( ), arg.begin( ) ).first == sStr.end( ) )
    {
      seed = std::stoll( arg.substr( sStr.size( ) ) );
    }
  }

  renderStep = std::max( renderStep, propStep );

  //
  //
  //
  WINDOW*pWindow( nullptr );


  if ( ( pWindow = initscr( ) ) == 0 )
  {
    throw std::runtime_error( "Could not initialize ncurses window" );
  }

  //
  // non-blocking character reads and no
  // echo back to terminal on keystroke
  //
  nodelay( pWindow, TRUE );
  cbreak( );
  noecho( );


  std::default_random_engine gen( seed );
  std::bernoulli_distribution dist;

  std::vector< bool > state( w * h );

  auto genLambda = [ &gen, &dist ]( )
                   {
                     return dist( gen );
                   };

  std::generate(
                std::begin( state ),
                std::end( state ),
                genLambda
                );

  GameOfLifeCpu game( state, w, h );
  renderState( game.getState( ), game.getWidth( ), game.getHeight( ) );


  auto propStart   = std::chrono::steady_clock::now( );
  auto renderStart = propStart;
  decltype( propStart )propEnd, renderEnd;
  std::chrono::duration< double > seconds;

  bool quitLoop = false;

  while ( !quitLoop )
  {
    propEnd = renderEnd = std::chrono::steady_clock::now( );
    seconds = propEnd - propStart;

    if ( seconds.count( ) > propStep )
    {
      game.propogateState( );
      propStart = propEnd;
    }

    seconds = renderEnd - renderStart;

    if ( seconds.count( ) > renderStep )
    {
      renderState( game.getState( ), game.getWidth( ), game.getHeight( ) );
      renderStart = renderEnd;
    }

    //
    // check input
    //
    int ch = getch( );

    switch ( ch )
    {
    case 'q':
      quitLoop = true;
      break;

    default:
      break;
    } // switch

  }

  if ( pWindow )
  {
    delwin( pWindow );
    pWindow = nullptr;
    endwin( );
    refresh( );
  }

} // GameOfLifeApp::exec



} // namespace gol

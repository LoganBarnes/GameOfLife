// GameOfLifeCpu.cpp
#include "GameOfLifeCpu.hpp"


namespace gol
{


GameOfLifeCpu::GameOfLifeCpu(
                             std::vector< char >            initState,
                             std::vector< char >::size_type width,
                             std::vector< char >::size_type height,
                             const                          bool /*multiThreading*/
                             )
  : GameOfLife( initState, width, height )
  , prevState_( state_.size( ) )
{}



void
GameOfLifeCpu::propogateState( )
{
  prevState_ = state_;

  int iw = static_cast< int >( width_ );
  int ih = static_cast< int >( height_ );

  for ( decltype( height_ ) y = 0; y < height_; ++y )
  {
    for ( decltype( width_ ) x = 0; x < width_; ++x )
    {
      int neighbors = 0;

      for ( int yy = -1; yy <= 1; ++yy )
      {
        for ( int xx = -1; xx <= 1; ++xx )
        {
          if ( xx == 0 && yy == 0 )
          {
            continue;
          }

          auto ex =
            static_cast< decltype( width_ ) >( ( static_cast< int >( x ) + iw + xx ) % iw );
          auto why =
            static_cast< decltype( width_ ) >( ( static_cast< int >( y ) + ih + yy ) % ih );

          if ( prevState_[ why * width_ + ex ] )
          {
            ++neighbors;
          }
        }
      }

      auto index = y * width_ + x;
      auto state = prevState_[ index ];

      if ( state && ( neighbors != 2 && neighbors != 3 ) )
      {
        state_[ index ] = false;
      }
      else
      if ( !state && neighbors == 3 )
      {
        state_[ index ] = true;
      }
      else
      {
        state_[ index ] = state;
      }

    }
  }
} // GameOfLife::GameOfLifeCpu::propogateState



} // namespace gol

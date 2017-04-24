// GameOfLifeAlgorithm.hpp
#pragma once

#include "GameOfLife.hpp"


#ifdef __CUDACC__
#include "helper_math.h"
#define CUDA_DEVICE __device__
#else
typedef unsigned uint;

struct dim3
{
  uint x;
  uint y;
  uint z;

  dim3(
       uint x_ = 1,
       uint y_ = 1,
       uint z_ = 1
       )
    : x( x_ )
    , y( y_ )
    , z( z_ )
  {}
};

#define CUDA_DEVICE
#endif // ifdef __CUDACC__


namespace gol
{


///
/// \brief findNeighbors
/// \param pPrev
/// \param dim
/// \param x
/// \param y
/// \return
///
CUDA_DEVICE
GolBool
findNeighbors(
              const GolBool *pPrev,
              const dim3     dim,
              const uint     x,
              const uint     y
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

  GolBool state = pPrev[ y * dim.x + x ];

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
} // findNeighbors



} // namespace gol

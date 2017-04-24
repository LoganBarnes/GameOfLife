// GameOfLife.hpp
#pragma once

#include <vector>

//
// using char for CUDA compatibility but calling
// it a bool since that is how it's used
//
typedef char GolBool;


namespace gol
{

typedef std::vector< GolBool >::size_type SizeType;

class GameOfLife
{

public:

  explicit
  GameOfLife(
             std::vector< GolBool > initState,
             SizeType               width,
             SizeType               height
             )
    : state_ ( initState )
    , width_ ( width )
    , height_( height )
  {}

  virtual
  ~GameOfLife( ) = default;

  SizeType
  getWidth( ) { return width_; }

  SizeType
  getHeight( ) { return height_; }


  virtual
  const std::vector< GolBool >&
  getState( ) { return state_; }


  virtual
  void propogateState ( ) = 0;


protected:

  std::vector< GolBool > state_;

  SizeType width_;
  SizeType height_;

};

} // namespace gol

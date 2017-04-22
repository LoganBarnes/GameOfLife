// GameOfLife.hpp
#pragma once

#include <vector>


namespace gol
{

typedef std::vector< bool >::size_type SizeType;

class GameOfLife
{

public:

  explicit
  GameOfLife(
             std::vector< bool > initState,
             SizeType            width,
             SizeType            height
             )
    : state_ ( initState )
    , width_ ( width )
    , height_( height )
  {}

  virtual
  ~GameOfLife() = default;

  SizeType
  getWidth( ) { return width_; }

  SizeType
  getHeight( ) { return height_; }


  virtual
  const std::vector< bool >&
  getState( ) { return state_; }


  virtual
  void propogateState ( ) = 0;


protected:

  std::vector< bool > state_;

  SizeType width_;
  SizeType height_;

};

} // namespace gol

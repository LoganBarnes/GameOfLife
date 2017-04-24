// GameOfLife.hpp
#pragma once

#include <vector>


namespace gol
{

typedef std::vector< char >::size_type SizeType;

class GameOfLife
{

public:

  explicit
  GameOfLife(
             std::vector< char > initState,
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
  const std::vector< char >&
  getState( ) { return state_; }


  virtual
  void propogateState ( ) = 0;


protected:

  std::vector< char > state_;

  SizeType width_;
  SizeType height_;

};

} // namespace gol

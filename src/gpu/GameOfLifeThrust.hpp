// GameOfLifeThrust.cpp
#include "GameOfLife.hpp"

#include <memory>


namespace gol
{


class GameOfLifeThrust : public GameOfLife
{
public:

  explicit
  GameOfLifeThrust(
                   std::vector< bool > initState,
                   SizeType            width,
                   SizeType            height
                   );

  virtual
  void propogateState ( ) final;

  virtual
  const std::vector< bool > &getState ( ) final;


private:

  class GoLThrustImpl;
  std::unique_ptr< GoLThrustImpl > upImpl_;

};



} // namespace gol

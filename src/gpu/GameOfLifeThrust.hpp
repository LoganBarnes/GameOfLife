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
                   std::vector< GolBool > initState,
                   SizeType               width,
                   SizeType               height
                   );

  virtual
  void propogateState ( ) final;

  virtual
  const std::vector< GolBool > &getState ( ) final;


private:

  class GoLThrustImpl;
  std::unique_ptr< GoLThrustImpl > upImpl_;

};



} // namespace gol

// GameOfLifeThrust.cpp
#include "GameOfLife.hpp"

#include <memory>
#include "gpu/CudaManager.hpp"


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

  ~GameOfLifeThrust( );

  virtual
  void propogateState ( ) final;

  virtual
  const std::vector< GolBool > &getState ( ) final;


private:

  CudaManager cuda_;

  class GoLThrustImpl;
  std::unique_ptr< GoLThrustImpl > upImpl_;

};



} // namespace gol

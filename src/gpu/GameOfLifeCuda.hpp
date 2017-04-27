// GameOfLifeCuda.cpp
#include "GameOfLife.hpp"

#include <memory>
#include "gpu/CudaManager.hpp"


namespace gol
{


class GameOfLifeCuda : public GameOfLife
{
public:

  explicit
  GameOfLifeCuda(
                 std::vector< GolBool > initState,
                 SizeType               width,
                 SizeType               height
                 );

  ~GameOfLifeCuda( );

  virtual
  void propogateState ( ) final;

  virtual
  const std::vector< GolBool > &getState ( ) final;


private:

  CudaManager cuda_;

  class GoLCudaImpl;
  std::unique_ptr< GoLCudaImpl > upImpl_;

};



} // namespace gol

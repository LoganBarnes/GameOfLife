// GameOfLifeCpu.cpp
#include "GameOfLife.hpp"


namespace gol
{


class GameOfLifeCpu : public GameOfLife
{
public:

  explicit
  GameOfLifeCpu(
                std::vector< bool > initState,
                SizeType            width,
                SizeType            height,
                const bool          multiThreading = false
                );

  virtual
  void propogateState ( ) final;


private:

  std::vector< bool > prevState_;

};



} // namespace gol

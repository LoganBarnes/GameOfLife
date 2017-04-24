// GameOfLifeCpu.cpp
#include "GameOfLife.hpp"


namespace gol
{


class GameOfLifeCpu : public GameOfLife
{
public:

  explicit
  GameOfLifeCpu(
                std::vector< char > initState,
                SizeType            width,
                SizeType            height,
                const bool          multiThreading = false
                );

  virtual
  void propogateState ( ) final;


private:

  std::vector< char > prevState_;

};



} // namespace gol

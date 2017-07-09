// GameOfLifeCpu.cpp
#include "GameOfLife.hpp"

#include <thread>
#include <memory>

namespace gol
{


class GameOfLifeCpu : public GameOfLife
{
public:

  explicit
  GameOfLifeCpu(
                std::vector< GolBool > initState,
                SizeType               width,
                SizeType               height,
                const bool             multiThreading = false
                );

  ~GameOfLifeCpu( ) = default;

  virtual
  void propogateState ( ) final;

  //
  // Copying gets complicated with threads so
  // we don't allow it
  //
  GameOfLifeCpu( GameOfLifeCpu& )           = delete;
  GameOfLifeCpu&operator=( GameOfLifeCpu& ) = delete;

  //
  // Default move operators are fine since
  // references are maintained
  //
  GameOfLifeCpu( GameOfLifeCpu&& )           = default;
  GameOfLifeCpu&operator=( GameOfLifeCpu&& ) = default;


private:

  void _propogateState (
                        unsigned rowStart,
                        unsigned rowEnd
                        );

  void _propogateStateThreaded (
                                unsigned rowStart,
                                unsigned rowEnd
                                );

  std::vector< GolBool > prevState_;
  
  struct CtplStuff;
  std::shared_ptr< CtplStuff > m_;

};



} // namespace gol

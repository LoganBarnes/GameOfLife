// GameOfLifeCpu.cpp
#include "GameOfLife.hpp"

#include <thread>


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

  ~GameOfLifeCpu( );

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

  void _startThreadPool ( unsigned numThreads );
  void _killThreads ( );

  std::vector< GolBool >     prevState_;
  std::vector< std::thread > threads_;

  bool threadsRunning_;

};



} // namespace gol

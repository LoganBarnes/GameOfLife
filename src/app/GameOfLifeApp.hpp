// GameOfLifeApp.cpp
#pragma once

namespace gol
{

class GameOfLifeApp
{
public:

  explicit
  GameOfLifeApp( ) = default;

  void exec (
             const int    argc,
             const char **argv
             );
};

} // namespace gol

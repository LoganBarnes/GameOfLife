// GameOfLifeMain.cpp

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include "GameOfLifeApp.hpp"


int
main(
     const int    argc,
     const char **argv
     )
{
  try
  {
    gol::GameOfLifeApp app;

    app.exec( argc, argv );
  }
  catch ( const std::exception &e )
  {
    std::cerr << "Program failed: " << e.what( ) << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Exiting..." << std::endl;
  return EXIT_SUCCESS;
} // main

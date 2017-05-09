// CpuUnitTests.cpp
#include "cpu/GameOfLifeCpu.hpp"
#include "gmock/gmock.h"
#include <vector>


namespace
{


///
/// \brief The CpuUnitTests class
///
class CpuUnitTests : public ::testing::Test
{

protected:

  /////////////////////////////////////////////////////////////////
  /// \brief CpuUnitTests
  /////////////////////////////////////////////////////////////////
  CpuUnitTests( )
  {}


  /////////////////////////////////////////////////////////////////
  /// \brief ~CpuUnitTests
  /////////////////////////////////////////////////////////////////
  virtual
  ~CpuUnitTests( )
  {}

};


/////////////////////////////////////////////////////////////////
/// \brief TestSimpleSquareState
/////////////////////////////////////////////////////////////////
TEST_F( CpuUnitTests, TestSimpleSquareState )
{
  std::vector< char > initState = 
  {
    false, false, false, true,
    false, true,  true,  false,
    false, false, true,  false,
    false, true,  false, false,
  };

  gol::GameOfLifeCpu game( initState, 4, 4 );
  game.propogateState( );
}



} // namespace

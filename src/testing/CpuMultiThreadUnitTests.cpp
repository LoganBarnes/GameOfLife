// CpuMultiThreadUnitTests.cpp
#include "cpu/GameOfLifeCpu.hpp"
#include "gmock/gmock.h"
#include <vector>


namespace
{


///
/// \brief The CpuMultiThreadUnitTests class
///
class CpuMultiThreadUnitTests : public ::testing::Test
{

protected:

  /////////////////////////////////////////////////////////////////
  /// \brief CpuMultiThreadUnitTests
  /////////////////////////////////////////////////////////////////
  CpuMultiThreadUnitTests( )
  {}


  /////////////////////////////////////////////////////////////////
  /// \brief ~CpuMultiThreadUnitTests
  /////////////////////////////////////////////////////////////////
  virtual
  ~CpuMultiThreadUnitTests( )
  {}

};


/////////////////////////////////////////////////////////////////
/// \brief TestSimpleSquareState
/////////////////////////////////////////////////////////////////
TEST_F( CpuMultiThreadUnitTests, TestSimpleSquareState )
{
  std::vector< char > initState = 
  {
    false, false, false, true,
    false, true,  true,  false,
    false, false, true,  false,
    false, true,  false, false,
  };

  gol::GameOfLifeCpu game( initState, 4, 4, true );
  game.propogateState( );
}



} // namespace

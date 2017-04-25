// AllTimingUnitTests.cpp

#include "gpu/GameOfLifeThrust.hpp"
#include "gpu/GameOfLifeCuda.hpp"
#include "cpu/GameOfLifeCpu.hpp"

#include "gmock/gmock.h"


namespace
{


///
/// \brief The AllTimingUnitTests class
///
class AllTimingUnitTests : public ::testing::Test
{

protected:

  /////////////////////////////////////////////////////////////////
  /// \brief AllTimingUnitTests
  /////////////////////////////////////////////////////////////////
  AllTimingUnitTests( )
  {}


  /////////////////////////////////////////////////////////////////
  /// \brief ~AllTimingUnitTests
  /////////////////////////////////////////////////////////////////
  virtual
  ~AllTimingUnitTests( )
  {}

};


/////////////////////////////////////////////////////////////////
/// \brief UnimplementedTest
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, UnimplementedTest )
{
  std::vector< GolBool > v = { true, false, false, true };
  gol::GameOfLifeThrust gol( v, 2, 2 );
  gol.propogateState( );
  std::cout << "gol1" << std::endl;

  {
    gol::GameOfLifeThrust gol2( v, 2, 2 );
    gol2.propogateState( );
    std::cout << "gol2" << std::endl;
  }

}



} // namespace

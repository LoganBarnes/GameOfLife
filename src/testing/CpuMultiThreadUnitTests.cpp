// CpuMultiThreadUnitTests.cpp

#include "cpu/GameOfLifeCpu.hpp"

#include "gmock/gmock.h"

#include <random>
#include <vector>
#include <chrono>


namespace
{

constexpr unsigned long lowIterations  = 100ul;
constexpr unsigned long highIterations = 1000ul;

constexpr std::vector< GolBool >::size_type widthSmall  = 200;
constexpr std::vector< GolBool >::size_type heightSmall = 150;

constexpr std::vector< GolBool >::size_type widthBig  = 500;
constexpr std::vector< GolBool >::size_type heightBig = 300;

const auto seed =
  std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );

std::vector< GolBool > initStateSmall ( widthSmall *heightSmall );
std::vector< GolBool > initStateBig ( widthBig *heightBig );

bool initialized = false;

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
  {
    if ( !initialized )
    {
      std::default_random_engine gen( static_cast< unsigned > ( seed ) );
      std::bernoulli_distribution dist;

      auto genFunction = [ &gen, &dist ]( )
                         {
                           return dist( gen );
                         };

      std::generate(
                    std::begin( initStateSmall ),
                    std::end( initStateSmall ),
                    genFunction
                    );

      std::generate(
                    std::begin( initStateBig ),
                    std::end( initStateBig ),
                    genFunction
                    );

      initialized = true;
    }
  }


  /////////////////////////////////////////////////////////////////
  /// \brief ~CpuMultiThreadUnitTests
  /////////////////////////////////////////////////////////////////
  virtual
  ~CpuMultiThreadUnitTests( )
  {}


  static
  void
  iterateLow( gol::GameOfLife &game )
  {
    for ( auto i = 0; i < lowIterations; ++i )
    {
      game.propogateState( );
    }
  }


  static
  void
  iterateHigh( gol::GameOfLife &game )
  {
    for ( auto i = 0; i < highIterations; ++i )
    {
      game.propogateState( );
    }
  }


};


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/// \brief InitializingTest
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
TEST_F( CpuMultiThreadUnitTests, InitializingTest )
{}


/////////////////////////////////////////////////////////////////
/// \brief TestSmallCpuSingleThread
/////////////////////////////////////////////////////////////////
TEST_F( CpuMultiThreadUnitTests, TestSmallCpuSingleThread )
{
  gol::GameOfLifeCpu game( initStateSmall, widthSmall, heightSmall );
  CpuMultiThreadUnitTests::iterateLow( game );
}


/////////////////////////////////////////////////////////////////
/// \brief TestSmallCpuMultiThreads
/////////////////////////////////////////////////////////////////
TEST_F( CpuMultiThreadUnitTests, TestSmallCpuMultiThreads )
{
  gol::GameOfLifeCpu game( initStateSmall, widthSmall, heightSmall, true );
  CpuMultiThreadUnitTests::iterateLow( game );
}



/////////////////////////////////////////////////////////////////
/// \brief TestSmallCpuSingleThreadEqualsMultiThread
/////////////////////////////////////////////////////////////////
TEST_F( CpuMultiThreadUnitTests, TestSmallCpuSingleThreadEqualsMultiThread )
{
  gol::GameOfLifeCpu game1( initStateSmall, widthSmall, heightSmall );
  CpuMultiThreadUnitTests::iterateLow( game1 );

  gol::GameOfLifeCpu game2( initStateSmall, widthSmall, heightSmall, true );
  CpuMultiThreadUnitTests::iterateLow( game2 );

  auto &state1 = game1.getState();
  auto &state2 = game2.getState();

  EXPECT_EQ(state1.size(), state2.size());

  for (unsigned i = 0; i < state1.size(); ++i)
  {
    EXPECT_EQ(state1, state2);
  }
}


/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
// Big
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////
/// \brief TestBigCpuSingleThread
/////////////////////////////////////////////////////////////////
TEST_F( CpuMultiThreadUnitTests, TestBigCpuSingleThread )
{
  gol::GameOfLifeCpu game( initStateBig, widthBig, heightBig );
  CpuMultiThreadUnitTests::iterateHigh( game );
}

/////////////////////////////////////////////////////////////////
/// \brief TestBigCpuMultiThreads
/////////////////////////////////////////////////////////////////
TEST_F( CpuMultiThreadUnitTests, TestBigCpuMultiThreads )
{
  gol::GameOfLifeCpu game( initStateBig, widthBig, heightBig, true );
  CpuMultiThreadUnitTests::iterateHigh( game );
}

/////////////////////////////////////////////////////////////////
/// \brief TestBigCpuSingleThreadEqualsMultiThread
/////////////////////////////////////////////////////////////////
TEST_F( CpuMultiThreadUnitTests, TestBigCpuSingleThreadEqualsMultiThread )
{
  gol::GameOfLifeCpu game1( initStateBig, widthBig, heightBig );
  CpuMultiThreadUnitTests::iterateHigh( game1 );

  gol::GameOfLifeCpu game2( initStateBig, widthBig, heightBig, true );
  CpuMultiThreadUnitTests::iterateHigh( game2 );

  auto &state1 = game1.getState();
  auto &state2 = game2.getState();

  EXPECT_EQ(state1.size(), state2.size());

  for (unsigned i = 0; i < state1.size(); ++i)
  {
    EXPECT_EQ(state1, state2);
  }
}

} // namespace

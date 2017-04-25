// AllTimingUnitTests.cpp

#include "gpu/GameOfLifeThrust.hpp"
#include "gpu/GameOfLifeCuda.hpp"
#include "cpu/GameOfLifeCpu.hpp"

#include "gmock/gmock.h"

#include <random>
#include <vector>
#include <chrono>


namespace
{

constexpr unsigned long lowIterations  = 100ul;
constexpr unsigned long highIterations = 1000ul;

constexpr std::vector< GolBool >::size_type widthSmall  = 2000;
constexpr std::vector< GolBool >::size_type heightSmall = 1500;

constexpr std::vector< GolBool >::size_type widthBig  = 5000;
constexpr std::vector< GolBool >::size_type heightBig = 3000;

constexpr std::vector< GolBool >::size_type widthMassive  = 10000;
constexpr std::vector< GolBool >::size_type heightMassive = 7500;

const unsigned long long seed =
  std::chrono::high_resolution_clock::now( ).time_since_epoch( ).count( );

std::vector< GolBool > initStateSmall ( widthSmall *heightSmall );
std::vector< GolBool > initStateBig ( widthBig *heightBig );
std::vector< GolBool > initStateMassive ( widthMassive *heightMassive );

bool initialized = false;

const CudaManager cuda;


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
  {
    if ( !initialized )
    {
      std::default_random_engine gen( seed );
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

      std::generate(
                    std::begin( initStateMassive ),
                    std::end( initStateMassive ),
                    genFunction
                    );

      initialized = true;
    }
  }


  /////////////////////////////////////////////////////////////////
  /// \brief ~AllTimingUnitTests
  /////////////////////////////////////////////////////////////////
  virtual
  ~AllTimingUnitTests( )
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
/// \brief InitializingTest
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, InitializingTest )
{}


/////////////////////////////////////////////////////////////////
/// \brief TestSmallCpuSingleThread
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, TestSmallCpuSingleThread )
{
  gol::GameOfLifeCpu game( initStateSmall, widthSmall, heightSmall );
  AllTimingUnitTests::iterateLow( game );
}


/////////////////////////////////////////////////////////////////
/// \brief TestSmallCpuMultiThreads
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, TestSmallCpuMultiThreads )
{
  gol::GameOfLifeCpu game( initStateSmall, widthSmall, heightSmall, true );
  AllTimingUnitTests::iterateLow( game );
}


/////////////////////////////////////////////////////////////////
/// \brief PrepGpuSmall
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, PrepGpuSmall )
{
  gol::GameOfLifeCuda game( initStateSmall, widthSmall, heightSmall );
}


/////////////////////////////////////////////////////////////////
/// \brief TestSmallGpuCudaKernel
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, TestSmallGpuCudaKernel )
{
  gol::GameOfLifeCuda game( initStateSmall, widthSmall, heightSmall );
  AllTimingUnitTests::iterateLow( game );
}


/////////////////////////////////////////////////////////////////
/// \brief TestSmallGpuThrustKernel
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, TestSmallGpuThrustKernel )
{
  gol::GameOfLifeThrust game( initStateSmall, widthSmall, heightSmall );
  AllTimingUnitTests::iterateLow( game );
}


/////////////////////////////////////////////////////////////////
/// \brief TestBigCpuMultiThreads
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, TestBigCpuMultiThreads )
{
  gol::GameOfLifeCpu game( initStateBig, widthBig, heightBig, true );
  AllTimingUnitTests::iterateHigh( game );
}


/////////////////////////////////////////////////////////////////
/// \brief PrepGpuBig
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, PrepGpuBig )
{
  gol::GameOfLifeCuda game( initStateBig, widthBig, heightBig );
}


/////////////////////////////////////////////////////////////////
/// \brief TestBigGpuCudaKernel
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, TestBigGpuCudaKernel )
{
  gol::GameOfLifeCuda game( initStateBig, widthBig, heightBig );
  AllTimingUnitTests::iterateHigh( game );
}


/////////////////////////////////////////////////////////////////
/// \brief TestBigGpuThrustKernel
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, TestBigGpuThrustKernel )
{
  gol::GameOfLifeThrust game( initStateBig, widthBig, heightBig );
  AllTimingUnitTests::iterateHigh( game );
}


/////////////////////////////////////////////////////////////////
/// \brief PrepGpuMassive
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, PrepGpuMassive )
{
  gol::GameOfLifeCuda game( initStateMassive, widthMassive, heightMassive );
}


/////////////////////////////////////////////////////////////////
/// \brief TestMassiveGpuCudaKernel
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, TestMassiveGpuCudaKernel )
{
  gol::GameOfLifeCuda game( initStateMassive, widthMassive, heightMassive );
  AllTimingUnitTests::iterateHigh( game );
}


/////////////////////////////////////////////////////////////////
/// \brief TestMassiveGpuThrustKernel
/////////////////////////////////////////////////////////////////
TEST_F( AllTimingUnitTests, TestMassiveGpuThrustKernel )
{
  gol::GameOfLifeThrust game( initStateMassive, widthMassive, heightMassive );
  AllTimingUnitTests::iterateHigh( game );
}


} // namespace

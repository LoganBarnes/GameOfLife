// GameOfLifeThrust.cpp
#include "GameOfLife.hpp"

#include <memory>
#include "gpu/CudaManager.hpp"


namespace gol
{


class GameOfLifeThrust : public GameOfLife
{
public:

  explicit
  GameOfLifeThrust(
                   std::vector< GolBool > initState,
                   SizeType               width,
                   SizeType               height
                   );

  ~GameOfLifeThrust( );

  virtual
  void propogateState ( ) final;

  virtual
  const std::vector< GolBool > &getState ( ) final;


private:

  CudaManager cuda_;

  // This allows us to keep all cuda related 
  // syntax and headers in the source files
  struct MemberVars;
  std::shared_ptr< MemberVars > m_;

};



} // namespace gol

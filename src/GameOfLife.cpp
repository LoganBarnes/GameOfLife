// GameOfLife.cpp

namespace gol
{

///
/// \brief GameOfLife::GameOfLife
/// \param initState
/// \param width
///
GameOfLife::GameOfLife(
                       std::vector< char >            initState,
                       std::vector< char >::size_type width
                       )
  : upImpl_( new GameOfLife::GameOfLifeImpl( initState, width ) )
{}

///
/// \brief GameOfLife::~GameOfLife
///
GameOfLife::~GameOfLife( ) {}

///
/// \brief GameOfLife::propogateState
///
void
GameOfLife::propogateState( ) { upImpl_->propogateState( ); }

///
/// \brief GameOfLife::getState
/// \return
///
const std::vector< char >&
GameOfLife::getState( ) { return upImpl_->getState( ); }

///
/// \brief GameOfLife::getWidth
/// \return
///
std::vector< char >::size_type
GameOfLife::getWidth( ) { return upImpl_->getWidth( ); }

///
/// \brief GameOfLife::getHeight
/// \return
///
std::vector< char >::size_type
GameOfLife::getHeight( ) { return upImpl_->getHeight( ); }



///
/// \brief operator <<
/// \param os
/// \param g
/// \return
///
std::ostream&
operator<<(
           std::ostream     &os,
           const GameOfLife &g
           )
{
  g._printState( os );
  return os;
}



///
/// \brief GameOfLife::_printState
/// \param os
///
void
GameOfLife::_printState( std::ostream &os ) const
{
  typedef std::vector< char >::size_type SizeType;

  const std::vector< char > &state = upImpl_->getState( );

  SizeType width  = upImpl_->getWidth( );
  SizeType height = upImpl_->getHeight( );

  SizeType index = 0;

  for ( SizeType x = 0; x < width + 2; ++x )
  {
    os << '=';
  }

  os << '\n';

  for ( SizeType y = 0; y < height; ++y )
  {
    os << '|';

    for ( SizeType x = 0; x < width; ++x )
    {
      os << ( state[ index ] ? '0' : ' ' );
      ++index;
    }

    os << "|\n";
  }

  for ( SizeType x = 0; x < width + 2; ++x )
  {
    os << '=';
  }

  os << '\n';
} // GameOfLife::_printState



} // namespace gol

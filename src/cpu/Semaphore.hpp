// Semaphore.hpp
#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>

///
/// \brief The Semaphore class
///        Stack Overflow: C++0x has no semaphores? How to synchronize threds?
///
class Semaphore
{

public:

  ///
  /// \brief Semaphore
  /// \param count
  ///
  Semaphore( unsigned int count = 0 )
    : count_( count )
  {}

  ///
  /// \brief notify
  ///
  void
  notify( )
  {
    std::unique_lock< std::mutex > lock( mutex_ );
    ++count_;
    condition_.notify_one( );
  }


  ///
  /// \brief wait
  ///
  void
  wait( )
  {
    std::unique_lock< std::mutex > lock( mutex_ );

    while ( !count_ )
    {
      condition_.wait( lock );
    }

    --count_;
  }


private:

  std::mutex mutex_;
  std::condition_variable condition_;
  unsigned int count_;

};

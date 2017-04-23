cmake_minimum_required ( VERSION 3.7.2 )
project ( GameOfLife )

option( STRICT_FLAGS "Compile with strict error and warning flags" OFF )

# project dirs
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src )

include( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DownloadProject.cmake )
set( CMAKE_MACOSX_RPATH TRUE )


# compile flags
if ( NOT MSVC AND STRICT_FLAGS )
  set( PROJECT_CPP_FLAGS "${PROJECT_CPP_FLAGS} -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wno-unused" )
  set( PROJECT_CPP_FLAGS "${PROJECT_CPP_FLAGS} -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-declarations -Wundef -Werror" )
  set( PROJECT_CPP_FLAGS "${PROJECT_CPP_FLAGS} -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls" )
  set( PROJECT_CPP_FLAGS "${PROJECT_CPP_FLAGS} -Wshadow -Wsign-conversion -Wsign-promo -Wstrict-overflow=5 -Wswitch-default " )
endif( )

if ( NOT MSVC )
  set( PROJECT_CPP_FLAGS "-fPIC" )
endif( )


###################################################
###                  GOL LIB                    ###
###################################################

add_library (
             gol
             ${SRC_DIR}/cpu/GameOfLifeCpu.hpp
             ${SRC_DIR}/cpu/GameOfLifeCpu.cpp
             )

# target_link_libraries ( gol glfw )
# add_dependencies      ( gol glfw )

# target_include_directories( gol SYSTEM PUBLIC ${glfw_SOURCE_DIR}/include )
target_include_directories( gol        PUBLIC ${SRC_DIR}                 )

set_target_properties(
                      gol PROPERTIES
                      CXX_STANDARD          14
                      CXX_STANDARD_REQUIRED ON
                      )

if ( PROJECT_CPP_FLAGS )
  set_target_properties( gol PROPERTIES COMPILE_FLAGS ${PROJECT_CPP_FLAGS} )
endif( )



###################################################
###                    CUDA                     ###
###################################################
find_package( CUDA )

if ( ${CUDA_FOUND} )

  include_directories( SYSTEM ${SRC_DIR}/gpu/cuda_helpers )
  include_directories(        ${SRC_DIR}                  )

  ###
  ### set nvcc options
  ###
  if( UNIX OR APPLE )
    list( APPEND PROJECT_NVCC_FLAGS -Xcompiler -fPIC )
  endif( )

  if( APPLE )
    list( APPEND PROJECT_NVCC_FLAGS -Xcompiler -fno-strict-aliasing )
  endif( )

  ###
  ### Auto detect compute architecture
  ###
  if ( NOT ${CMAKE_VERSION} VERSION_LESS 3.7.1 AND NOT CUDA_ARCH_FLAGS )
    cuda_select_nvcc_arch_flags( CUDA_ARCH_FLAGS Auto )
    set( CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS} CACHE STRING "gencode architecture flags for nvcc" )
  endif( )

  list( APPEND PROJECT_NVCC_FLAGS ${CUDA_ARCH_FLAGS} )


  ###
  ### build CUDA library
  ###
  cuda_add_library(
                   golCuda

                   ${SRC_DIR}/gpu/GameOfLifeThrust.hpp
                   ${SRC_DIR}/gpu/GameOfLifeThrust.cu

                   OPTIONS --std=c++11 --expt-relaxed-constexpr ${PROJECT_NVCC_FLAGS}
                   DEBUG -g
                   RELEASE -use_fast_math
                   )

  set_target_properties(
                        golCuda PROPERTIES
                        CXX_STANDARD          14
                        CXX_STANDARD_REQUIRED ON
                        )

  if ( PROJECT_CPP_FLAGS )
    set_target_properties( golCuda PROPERTIES COMPILE_FLAGS ${PROJECT_CPP_FLAGS} )
  endif( )

  target_link_libraries( gol golCuda )
  add_dependencies     ( gol golCuda )

  list(
       APPEND GPU_TESTS
       ${SRC_DIR}/testing/CudaUnitTests.cpp
       ${SRC_DIR}/testing/ThrustUnitTests.cpp
       ${SRC_DIR}/testing/AllTimingUnitTests.cpp
       )

endif ( ${CUDA_FOUND} )




###################################################
###                  GOL EXEC                   ###
###################################################

###
### NCurses terminal functionality to render
### game states on the command line
###
### We only build the interactive executable if
### the ncurses library is installed on the system
###
set( CURSES_NEED_NCURSES TRUE )
find_package( Curses )


if ( ${Curses_FOUND} )

  add_executable(
                 gameOfLife
                 ${SRC_DIR}/app/GameOfLifeApp.hpp
                 ${SRC_DIR}/app/GameOfLifeApp.cpp
                 ${SRC_DIR}/app/GameOfLifeMain.cpp
                 )

  target_link_libraries ( gameOfLife gol ${CURSES_LIBRARY} )
  add_dependencies      ( gameOfLife gol                   )

  target_include_directories( gameOfLife SYSTEM PUBLIC ${CURSES_INCLUDE_DIR} )
  target_include_directories( gameOfLife        PUBLIC ${SRC_DIR}            )

  set_target_properties(
                        gameOfLife PROPERTIES
                        CXX_STANDARD          14
                        CXX_STANDARD_REQUIRED ON
                        )

  if ( PROJECT_CPP_FLAGS )
    set_target_properties( gameOfLife PROPERTIES COMPILE_FLAGS ${PROJECT_CPP_FLAGS} )
  endif( )

endif ( ${Curses_FOUND} )






###################################################
###                  TESTING                    ###
###################################################


###################################
###   add gmock functionality   ###
###################################

# Download and unpack googletest at configure time
download_project(
                 PROJ                googletest
                 GIT_REPOSITORY      https://github.com/google/googletest.git
                 GIT_TAG             master
                 UPDATE_DISCONNECTED 1
                 QUIET
                 )

# Prevent overriding the parent project's compiler/linker settings on Windows
set( gtest_force_shared_crt  ON CACHE BOOL "" FORCE )

# Add googlemock directly to the project build
add_subdirectory( ${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} )


add_executable(
               testGameOfLife
               ${SRC_DIR}/testing/CpuUnitTests.cpp
               ${SRC_DIR}/testing/CpuMultiThreadUnitTests.cpp
               ${GPU_TESTS}
               )

target_link_libraries ( testGameOfLife gol gmock_main )
add_dependencies      ( testGameOfLife gol gmock_main )

target_include_directories( testGameOfLife PUBLIC ${SRC_DIR} )
target_include_directories(
                           testGameOfLife SYSTEM PUBLIC
                           ${gtest_SOURCE_DIR}/include
                           ${gmock_SOURCE_DIR}/include
                           )

set_target_properties(
                      testGameOfLife PROPERTIES
                      CXX_STANDARD          14
                      CXX_STANDARD_REQUIRED ON
                      )

if ( PROJECT_CPP_FLAGS )
  set_target_properties( testGameOfLife PROPERTIES COMPILE_FLAGS ${PROJECT_CPP_FLAGS} )
endif( )

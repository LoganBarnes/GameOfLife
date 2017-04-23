Game of Life
============

[![Build Status](https://travis-ci.org/LoganBarnes/GameOfLife.svg?branch=master)](https://travis-ci.org/LoganBarnes/GameOfLife)

Download
--------
Clone the repo:

```bash
git clone https://gitlab.com/LoganBarnes/GameOfLife.git
```


Build
-----

### Bash & CMake

```bash
# from project root dir
mkdir build
cd build
cmake ..
make -j12
```

### Pure CMake

```bash
# from project root dir
cmake -E make_directory build
cmake -E chdir build cmake ..
cmake -E chdir build cmake --build . -- -j12
```

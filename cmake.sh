#!/bin/bash

CC=icc #gcc
CXX=icpc #g++
FC=ifortran

BUILD_TYPE=Release
INSTALL_DIR=$HOME/.xacc
BUILD_TESTS=TRUE
BUILD_EXAMPLES=TRUE

cmake \
  -DCMAKE_C_COMPILER=${CC} \
  -DCMAKE_CXX_COMPILER=${CXX} \
  -DCMAKE_Fortran_COMPILER=${FC} \
  -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
  -DXACC_BUILD_TESTS=${BUILD_TESTS} \
  -DXACC_BUILD_EXAMPLES=${BUILD_EXAMPLES} \
  ..
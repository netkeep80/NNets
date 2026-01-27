mkdir build
cd build
cmake ..
cmake --build . --config=Release
ctest -C Release
@echo off
rem ctest -C Release --verbose
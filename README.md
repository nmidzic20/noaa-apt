// from project folder (windows)

// configure cmake

cmake -S . -B build `  -DCMAKE_BUILD_TYPE=Release`
-DCMAKE_TOOLCHAIN_FILE="C:/Users/Korisnik/FOI/Diplomski/software/cmake/dev/vcpkg/scripts/buildsystems/vcpkg.cmake" `
-DVCPKG_TARGET_TRIPLET=x64-windows

// build

cmake --build build --config Release

// run

.\build\Release\noaa_apt.exe "C:\path\to\input.wav" "C:\path\to\out.png" 1200

.\build\Release\noaa_apt.exe ".\test.wav" ".\out.png" 1200

For FTT, install from within vcpkg:

.\vcpkg install fftw3[core]:x64-windows

Add to CMakeLists.txt:

find_package(FFTW3 REQUIRED)
target_link_libraries(noaa_apt PRIVATE SndFile::sndfile FFTW3::fftw3)

Then configure cmake again before building (commands at the top).

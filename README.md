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

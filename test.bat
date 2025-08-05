@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" arm64
set "LIB=%LIB%;C:\Program Files (x86)\Windows Kits\10\lib\10.0.22621.0\um\arm64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.22621.0\ucrt\arm64"
cargo test --release %*
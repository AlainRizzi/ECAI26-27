@echo off
rem Runs a command in an arm64 Linux container, with this folder mounted as /lab1.
rem Requires Docker Desktop to be running. Examples:
rem   .\arm make test
rem   .\arm make vector_add OPT=-O3
rem   .\arm ./src/vector_add/benchmark
docker run --rm --platform linux/arm64 -v "%~dp0.:/lab1" -w /lab1 gcc:14 %*

@echo off
rem Lets "make ..." work from cmd in this folder by running it through arm.cmd.
call "%~dp0arm.cmd" make %*

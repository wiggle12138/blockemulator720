REM BlockEmulator UTF-8 版本启动脚本（更新版）
REM 设置UTF-8环境
chcp 65001 >nul 2>&1

start cmd /k blockEmulator_Windows_UTF8.exe -n 0 -N 4 -s 0 -S 2 & 

start cmd /k blockEmulator_Windows_UTF8.exe -n 1 -N 4 -s 0 -S 2 & 

start cmd /k blockEmulator_Windows_UTF8.exe -n 2 -N 4 -s 0 -S 2 & 

start cmd /k blockEmulator_Windows_UTF8.exe -n 3 -N 4 -s 0 -S 2 & 

start cmd /k blockEmulator_Windows_UTF8.exe -n 0 -N 4 -s 1 -S 2 & 

start cmd /k blockEmulator_Windows_UTF8.exe -n 1 -N 4 -s 1 -S 2 & 

start cmd /k blockEmulator_Windows_UTF8.exe -n 2 -N 4 -s 1 -S 2 & 

start cmd /k blockEmulator_Windows_UTF8.exe -n 3 -N 4 -s 1 -S 2 & 

start cmd /k blockEmulator_Windows_UTF8.exe -c -N 4 -S 2 & 


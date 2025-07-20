@echo off
setlocal enabledelayedexpansion

REM =============================================
REM 增强版Docker部署脚本 (Windows)
REM 版本: 2.0
REM 修复ANSI颜色问题并增强功能
REM =============================================

@REM REM 禁用ANSI颜色代码（解决显示问题）
@REM set "RED="
@REM set "GREEN="
@REM set "YELLOW="
@REM set "BLUE="
@REM set "NC="

REM 替代颜色方案：使用文本标签
set "INFO_TAG=[INFO]"
set "SUCCESS_TAG=[SUCCESS]"
set "WARNING_TAG=[WARNING]"
set "ERROR_TAG=[ERROR]"

REM 打印消息函数
:print_info
echo %INFO_TAG% %~1
goto :eof

:print_success
echo %SUCCESS_TAG% %~1
goto :eof

:print_warning
echo %WARNING_TAG% %~1
goto :eof

:print_error
echo %ERROR_TAG% %~1
goto :eof

REM 检查Docker是否安装
:check_docker
call :print_info "检查Docker环境..."
docker --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker未安装，请先安装Docker Desktop"
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker服务未启动，请启动Docker Desktop"
    exit /b 1
)

call :print_success "Docker检查通过"
goto :eof

REM 增强版文件检查
:check_files
call :print_info "检查必要文件..."

set "missing_files=0"

REM 检查可执行文件
set "exe_file="
if exist "..\blockEmulator" (
    set "exe_file=..\blockEmulator"
    call :print_info "找到可执行文件: blockEmulator"
) else if exist "..\Precompile.exe" (
    set "exe_file=..\Precompile.exe"
    call :print_info "找到可执行文件: Precompile.exe"
) else (
    call :print_error "未找到可执行文件"
    set /a "missing_files+=1"
)

REM 检查配置文件
if not exist "..\paramsConfig.json" (
    call :print_error "缺少配置文件: paramsConfig.json"
    set /a "missing_files+=1"
) else (
    call :print_info "找到配置文件: paramsConfig.json"
)

if not exist "..\ipTable.json" (
    call :print_error "缺少配置文件: ipTable.json"
    set /a "missing_files+=1"
) else (
    call :print_info "找到配置文件: ipTable.json"
)

REM 检查数据集文件
if not exist "..\selectedTxs_300K.csv" (
    call :print_warning "数据集文件 selectedTxs_300K.csv 不存在，系统将使用模拟数据"
) else (
    call :print_info "找到数据集文件: selectedTxs_300K.csv"
)

if !missing_files! gtr 0 (
    call :print_error "缺少必要文件，无法继续"
    exit /b 1
)

goto :eof

REM 显示帮助信息
:show_help
echo.
echo Docker部署脚本 (Windows版本 v2.0)
echo.
echo 用法: %~nx0 [命令]
echo.
echo 命令:
echo   build     构建Docker镜像
echo   start     启动所有节点
echo   stop      停止所有节点
echo   restart   重启所有节点
echo   status    查看服务状态和健康检查
echo   logs      查看所有节点日志
echo   logs [节点名] 查看指定节点日志
echo   exec [节点名] 进入指定容器
echo   cleanup   清理所有Docker资源
echo   health    检查容器健康状态
echo   help      显示此帮助信息
echo.
echo 节点名称:
echo   shard0-node0  shard0-node1  shard1-node0  shard1-node1
echo.
echo 示例:
echo   %~nx0 build
echo   %~nx0 start
echo   %~nx0 logs shard0-node0
echo   %~nx0 exec shard0-node0
goto :eof

REM 构建镜像（添加进度指示）
:build_images
call :print_info "开始构建Docker镜像..."
call :print_info "这可能需要几分钟，请耐心等待..."

REM 添加超时控制
set "start_time=%time%"
call docker-compose build

if errorlevel 1 (
    call :print_error "镜像构建失败"
    exit /b 1
)

REM 计算构建时间
call :get_elapsed_time "%start_time%" build_time
call :print_success "Docker镜像构建完成 (耗时: !build_time!)"
goto :eof

REM 启动服务（添加健康检查）
:start_services
call :print_info "启动区块链节点服务..."
docker-compose up -d

REM 添加健康检查等待
call :print_info "等待节点启动..."
set "all_healthy=0"
set "retry_count=0"

:health_check_loop
set /a "retry_count+=1"
if !retry_count! gtr 10 (
    call :print_warning "部分节点启动较慢，请稍后手动检查"
    goto :start_complete
)

set "healthy_count=0"
for %%n in (shard0-node0 shard0-node1 shard1-node0 shard1-node1) do (
    docker inspect -f "{{.State.Health.Status}}" %%n | find "healthy" >nul
    if !errorlevel! == 0 set /a "healthy_count+=1"
)

if !healthy_count! equ 4 (
    set "all_healthy=1"
    goto :start_complete
)

REM 显示进度
call :print_info "节点健康检查: !healthy_count!/4 (尝试 !retry_count!/10)"
timeout /t 5 >nul
goto :health_check_loop

:start_complete
if !all_healthy! equ 1 (
    call :print_success "所有节点已启动并健康运行"
) else (
    call :print_warning "节点已启动，但部分节点健康检查未通过"
)
goto :eof

REM 停止服务
:stop_services
call :print_info "停止区块链节点服务..."
docker-compose down
call :print_success "所有节点已停止"
goto :eof

REM 重启服务
:restart_services
call :print_info "重启区块链节点服务..."
docker-compose restart
call :print_success "所有节点已重启"
goto :eof

REM 增强版服务状态检查
:status_services
call :print_info "容器状态概览:"
docker-compose ps

echo.
call :print_info "容器资源使用情况:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

echo.
call :print_info "容器健康状态:"
for %%n in (shard0-node0 shard0-node1 shard1-node0 shard1-node1) do (
    docker inspect -f "{{.Name}} - Status: {{.State.Status}}, Health: {{.State.Health.Status}}" %%n
)
goto :eof

REM 查看日志（添加时间戳）
:view_logs
if "%~1"=="" (
    call :print_info "查看所有节点日志 (按Ctrl+C退出)..."
    docker-compose logs -f --timestamps
) else (
    call :print_info "查看节点 %~1 的日志 (按Ctrl+C退出)..."
    docker-compose logs -f --timestamps %~1
)
goto :eof

REM 进入容器
:enter_container
if "%~1"=="" (
    call :print_error "请指定容器名称，例如: shard0-node0"
    exit /b 1
)
call :print_info "进入容器 %~1..."
docker-compose exec %~1 /bin/sh
goto :eof

REM 清理资源
:cleanup
call :print_warning "清理所有Docker资源..."
docker-compose down -v
call :print_success "清理完成"
goto :eof

REM 新增：容器健康检查
:health_check
call :print_info "容器健康状态检查:"
set "all_healthy=1"
for %%n in (shard0-node0 shard0-node1 shard1-node0 shard1-node1) do (
    docker inspect -f "{{.Name}} - {{.State.Health.Status}}" %%n
    docker inspect -f "{{.State.Health.Status}}" %%n | find "healthy" >nul
    if !errorlevel! neq 0 set "all_healthy=0"
)

if !all_healthy! equ 1 (
    call :print_success "所有容器健康状态正常"
) else (
    call :print_warning "部分容器健康状态异常"
)
goto :eof

REM 计算耗时函数
:get_elapsed_time
setlocal
set "start=%~1"
set "end=%time%"

REM 时间计算逻辑
set /a "start_h=1%start:~0,2%-100, start_m=1%start:~3,2%-100, start_s=1%start:~6,2%-100"
set /a "end_h=1%end:~0,2%-100, end_m=1%end:~3,2%-100, end_s=1%end:~6,2%-100"

set /a "total_sec=((end_h-start_h)*3600)+((end_m-start_m)*60)+(end_s-start_s)"
set /a "min=total_sec/60, sec=total_sec%%60"

endlocal & set "%~2=%min%分%sec%秒"
goto :eof

REM 主函数
:main
set "command=%~1"

REM 显示标题
echo.
echo ========================================
echo  区块链节点部署系统 - Windows版本 v2.0
echo ========================================
echo.

REM 检查Docker环境
call :check_docker
if errorlevel 1 exit /b 1

REM 检查必要文件
call :check_files
if errorlevel 1 exit /b 1

REM 根据命令执行相应操作
if "%command%"=="build" (
    call :build_images
) else if "%command%"=="start" (
    call :start_services
) else if "%command%"=="stop" (
    call :stop_services
) else if "%command%"=="restart" (
    call :restart_services
) else if "%command%"=="status" (
    call :status_services
) else if "%command%"=="logs" (
    call :view_logs %~2
) else if "%command%"=="exec" (
    call :enter_container %~2
) else if "%command%"=="cleanup" (
    call :cleanup
) else if "%command%"=="health" (
    call :health_check
) else if "%command%"=="" (
    call :show_help
) else if "%command%"=="help" (
    call :show_help
) else if "%command%"=="--help" (
    call :show_help
) else if "%command%"=="-h" (
    call :show_help
) else (
    call :print_error "未知命令: %command%"
    call :show_help
    exit /b 1
)

goto :eof

REM 执行主函数
call :main %*
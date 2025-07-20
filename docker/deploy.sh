#!/bin/bash

# Docker部署脚本
# 用于创建、部署和加载Docker运行环境

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker服务未启动，请启动Docker"
        exit 1
    fi
    
    print_success "Docker检查通过"
}

# 检查必要文件
check_files() {
    print_info "检查必要文件..."
    
    # 检查可执行文件
    local exe_file=""
    if [ -f "../blockEmulator" ]; then
        exe_file="../blockEmulator"
        print_success "找到可执行文件: blockEmulator"
    elif [ -f "../Precompile.exe" ]; then
        exe_file="../Precompile.exe"
        print_success "找到可执行文件: Precompile.exe"
    else
        print_error "未找到可执行文件，请先编译项目"
        print_info "请运行: go build -o blockEmulator ."
        exit 1
    fi
    
    # 检查配置文件
    if [ ! -f "../paramsConfig.json" ]; then
        print_error "配置文件 paramsConfig.json 不存在"
        exit 1
    fi
    print_success "找到配置文件: paramsConfig.json"
    
    if [ ! -f "../ipTable.json" ]; then
        print_error "配置文件 ipTable.json 不存在"
        exit 1
    fi
    print_success "找到配置文件: ipTable.json"
    
    # 检查数据集文件
    if [ ! -f "../selectedTxs_300K.csv" ]; then
        print_warning "数据集文件 selectedTxs_300K.csv 不存在，系统将使用模拟数据"
    else
        print_success "找到数据集文件: selectedTxs_300K.csv"
    fi
}

# 显示帮助信息
show_help() {
    echo "Docker部署脚本"
    echo ""
    echo "用法: $0 [命令]"
    echo ""
    echo "命令:"
    echo "  build     构建Docker镜像"
    echo "  start     启动所有节点"
    echo "  stop      停止所有节点"
    echo "  restart   重启所有节点"
    echo "  status    查看服务状态"
    echo "  logs      查看所有节点日志"
    echo "  logs [节点名] 查看指定节点日志"
    echo "  exec [节点名] 进入指定容器"
    echo "  cleanup   清理所有Docker资源"
    echo "  help      显示此帮助信息"
    echo ""
    echo "节点名称:"
    echo "  shard0-node0  shard0-node1  shard1-node0  shard1-node1"
    echo ""
    echo "示例:"
    echo "  $0 build"
    echo "  $0 start"
    echo "  $0 logs shard0-node0"
    echo "  $0 exec shard0-node0"
}

# 构建镜像
build_images() {
    print_info "开始构建Docker镜像..."
    
    # 构建镜像
    docker-compose build
    
    print_success "Docker镜像构建完成"
}

# 启动服务
start_services() {
    print_info "启动区块链节点服务..."
    
    # 启动所有服务
    docker-compose up -d
    
    print_success "所有节点已启动"
}

# 停止服务
stop_services() {
    print_info "停止区块链节点服务..."
    
    # 停止所有服务
    docker-compose down
    
    print_success "所有节点已停止"
}

# 重启服务
restart_services() {
    print_info "重启区块链节点服务..."
    
    # 重启所有服务
    docker-compose restart
    
    print_success "所有节点已重启"
}

# 查看服务状态
status_services() {
    print_info "查看服务状态..."
    
    # 显示服务状态
    docker-compose ps
    
    # 显示资源使用情况
    print_info "容器资源使用情况："
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# 查看日志
view_logs() {
    local service_name=$1
    
    if [ -z "$service_name" ]; then
        print_info "查看所有节点日志..."
        docker-compose logs -f
    else
        print_info "查看节点 $service_name 的日志..."
        docker-compose logs -f $service_name
    fi
}

# 进入容器
enter_container() {
    local service_name=$1
    
    if [ -z "$service_name" ]; then
        print_error "请指定容器名称，例如: shard0-node0"
        exit 1
    fi
    
    print_info "进入容器 $service_name..."
    docker-compose exec $service_name /bin/sh
}

# 清理资源
cleanup() {
    print_warning "清理所有Docker资源..."
    
    # 停止并删除容器
    docker-compose down -v
    
    # 删除镜像
    docker rmi $(docker images -q block-emulator-main_shard0-node0) 2>/dev/null || true
    docker rmi $(docker images -q block-emulator-main_shard0-node1) 2>/dev/null || true
    docker rmi $(docker images -q block-emulator-main_shard1-node0) 2>/dev/null || true
    docker rmi $(docker images -q block-emulator-main_shard1-node1) 2>/dev/null || true
    
    print_success "清理完成"
}

# 主函数
main() {
    local command=$1
    
    # 检查Docker环境
    check_docker
    
    # 检查必要文件
    check_files
    
    case $command in
        "build")
            build_images
            ;;
        "start")
            start_services
            ;;
        "stop")
            stop_services
            ;;
        "restart")
            restart_services
            ;;
        "status")
            status_services
            ;;
        "logs")
            view_logs $2
            ;;
        "exec")
            enter_container $2
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"--help"|"-h"|"")
            show_help
            ;;
        *)
            print_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 
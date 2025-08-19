#!/bin/bash

# Define variables
CONTAINER_NAME="forum"

# 定义错误处理函数
handle_error() {
    echo "❌ 命令执行失败，错误发生在行: $1"
    exit 1
}

# Step 1: 停止容器（容器不存在不是严重错误，可忽略）
echo "🔄 停止容器 $CONTAINER_NAME..."
docker stop "$CONTAINER_NAME" || echo "⚠️ 容器 $CONTAINER_NAME 不存在或已停止"

# Step 2: 删除容器
echo "🗑️ 删除容器 $CONTAINER_NAME..."
docker rm "$CONTAINER_NAME" 2>/dev/null || echo "⚠️ 删除容器失败或容器不存在"

# Step 3: 运行新容器（关键步骤，失败则“catch”）
echo "🚀 运行新容器..."
if ! docker run --name "$CONTAINER_NAME" -p 9999:80 -d postmill-populated-exposed-withimg; then
    handle_error $LINENO
fi

# Step 4: 等待服务启动
echo "⏳ 等待服务启动..."
sleep 15

echo "✅ 部署成功！"


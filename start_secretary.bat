@echo off
:: 1. 切换到当前脚本所在的目录 (确保路径正确)
cd /d "%~dp0"

:: 2. 激活 uv 创建的虚拟环境
call .venv\Scripts\activate

:: 3. 启动 Streamlit 应用
:: --server.headless true 是为了防止它在后台询问邮箱
echo 正在启动 Theon 的私人秘书...
streamlit run app.py
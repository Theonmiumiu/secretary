@echo off
:: 切换到当前脚本目录
cd /d "%~dp0"

:: 直接使用虚拟环境里的 python.exe 来运行 streamlit
:: 这种写法不需要 activate，移动文件夹后也能用
".venv\Scripts\python.exe" -m streamlit run app.py
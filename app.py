import streamlit as st
import datetime
import pandas as pd
from pathlib import Path
import langchain_core.load
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# ================= 导入核心类和工具 =================
from main import Secretary
from src.LLMs.models import (
    get_former_logs, get_date_delta, get_time,
    update_dashboard, add_dashboard, remove_dashboard,
    stream_wrapper, _get_dashboard_df
)
from src.LLMs.prompts import record_system_prompt

# 建立工具字典，方便调用
tool_dic = {tool.name: tool for tool in
            [get_former_logs, get_date_delta, get_time, update_dashboard, add_dashboard, remove_dashboard]}

st.set_page_config(page_title="Theon's Secretary", page_icon="📝", layout="wide")

# ================= 1. 核心状态初始化 =================
if "secretary" not in st.session_state:
    st.session_state.secretary = Secretary()
    # 每次启动应用时检查并生成周报（只有周一且未生成过才会真正执行）
    st.session_state.secretary.weekly_report()

sec = st.session_state.secretary
date_str = sec.date.strftime('%Y_%m_%d')
today_file_path = sec.chat_logs_dir / f"context_{date_str}.json"

# 加载当天的历史记录到 session_state
if "history" not in st.session_state:
    if today_file_path.exists():
        with open(today_file_path, 'r', encoding='utf-8') as f:
            st.session_state.history = langchain_core.load.loads(f.read())
    else:
        # ======= 模拟 main.py 里的每日首次初始化 =======
        with st.spinner(f"🌞 {date_str} {sec.weekday}！正在初始化今日工作..."):
            df, _ = _get_dashboard_df()
            df_md = df.to_markdown()

            if sec.valid_dashboard_dates_list:
                history = [SystemMessage(content=record_system_prompt)]
                instruction = f'今天是{date_str}，{sec.weekday}，我们最近最新的任务计划面板是<dashboard>，'
                history += [HumanMessage(content=instruction),
                            HumanMessage(content=f"<dashboard>\n{df_md}\n</dashboard>")]
            else:
                initial_instruction = "用户之前还没有跟你提起过他的任务，因此没有<dashboard>，请询问他最近有什么计划，然后做好记录"
                history = [SystemMessage(content=record_system_prompt), HumanMessage(content=initial_instruction)]

            try:
                res = stream_wrapper(sec.model, history)
                history.append(res)
                with open(today_file_path, 'w', encoding='utf-8') as f:
                    f.write(langchain_core.load.dumps(history, pretty=True, ensure_ascii=False))
                st.session_state.history = history
            except Exception as e:
                st.error(f"初始化失败: {e}")
                st.stop()

# ================= 2. 侧边栏：历史档案 =================
with st.sidebar:
    st.header("🗄️ 档案室")
    view_mode = st.radio("选择查看内容:", ["当前对话", "往期周报", "每日日志"], index=0)
    selected_content = None

    if view_mode == "往期周报":
        st.subheader("📊 周报列表")
        if sec.weekly_reports_dir.exists():
            report_files = sorted(sec.weekly_reports_dir.glob("周报_*.txt"), key=lambda f: f.stat().st_mtime,
                                  reverse=True)
            if report_files:
                selected_file = st.selectbox("选择周报", report_files, format_func=lambda x: x.stem)
                if selected_file:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        selected_content = f.read()
            else:
                st.info("暂无周报")

    elif view_mode == "每日日志":
        st.subheader("📅 日志列表")
        if sec.chat_logs_dir.exists():
            log_files = sorted(sec.chat_logs_dir.glob("context_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            if log_files:
                selected_log = st.selectbox("选择日志", log_files, format_func=lambda x: x.stem.replace('context_', ''))
                if selected_log:
                    with open(selected_log, 'r', encoding='utf-8') as f:
                        log_data = langchain_core.load.loads(f.read())
                        formatted_log = ""
                        for msg in log_data:
                            if isinstance(msg, HumanMessage):
                                content = msg.content.split('\n今天是')[0]
                                formatted_log += f"**User**: {content}\n\n"
                            elif isinstance(msg, AIMessage) and msg.content:
                                formatted_log += f"**Secretary**: {msg.content}\n\n---\n\n"
                        selected_content = formatted_log
            else:
                st.info("暂无日志")

# ================= 3. 核心界面布局 =================
if view_mode == "当前对话":
    left_col, right_col = st.columns([0.65, 0.35], gap="large")

    # ---------------- 左侧栏：主聊天区 ----------------
    with left_col:
        st.title("📝 Theon's Secretary")
        st.caption(f"当前时间: {date_str} {sec.weekday}")

        # 1. 渲染聊天历史
        for msg in st.session_state.history:
            # 过滤不需要在 UI 显示的系统信息和工具调用信息
            if isinstance(msg, SystemMessage):
                continue
            if isinstance(msg, ToolMessage):
                with st.expander(f"🛠️ 工具执行完毕 (ID: {msg.tool_call_id[:6]}...)"):
                    st.caption(msg.content)
                continue
            if isinstance(msg, AIMessage) and not msg.content:
                continue  # 这是纯纯的工具调用指令头，不用展示

            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content
            if content:
                # 去除紧箍咒显示
                display_content = content.split('\n今天是')[0]
                with st.chat_message(role):
                    st.markdown(display_content)

        # 2. 处理聊天输入与 Agent 执行流
        if prompt := st.chat_input("有什么我可以帮忙的？"):
            # 界面展示
            with st.chat_message("user"):
                st.markdown(prompt)

            # 加上紧箍咒
            augmented_prompt = prompt
            if any(i in prompt for i in ['日志', '计划', '日程', '安排', '规划', '任务']):
                augmented_prompt = prompt + f'\n今天是{sec.date}，回忆我们一开始对你的输出要求，严格按照[临期任务]、[学业任务]、[实习任务]、[习惯养成]、[聊天]五个部分回复。'

            history = st.session_state.history
            history.append(HumanMessage(content=augmented_prompt))

            # 【核心安全机制】快照长度
            initial_history_len = len(history)

            with st.chat_message("assistant"):
                with st.spinner("Secretary 正在思考与执行..."):
                    try:
                        # 注入面板数据
                        df, _ = _get_dashboard_df()
                        df_md = df.to_markdown()

                        agent_reply = stream_wrapper(sec.model_with_tools, history[:-1] + [
                            HumanMessage(content=f'<dashboard>{df_md}</dashboard>'), history[-1]])
                        history.append(agent_reply)

                        consecutive_tool_errors = 0

                        # ============ 完美的工具调用循环 ============
                        while agent_reply.tool_calls:
                            for call in agent_reply.tool_calls:
                                call_id = call.get('id')
                                call_name = call.get('name')
                                call_args = call.get('args')

                                try:
                                    if call_name == 'get_former_logs':
                                        call_args['chat_logs_dir'] = str(sec.chat_logs_dir)
                                        result = get_former_logs.invoke(call_args)
                                    elif call_name in tool_dic:
                                        result = tool_dic[call_name].invoke(call_args)
                                    else:
                                        raise ValueError(f"不存在的工具: {call_name}")

                                    consecutive_tool_errors = 0
                                except Exception as tool_e:
                                    consecutive_tool_errors += 1
                                    if consecutive_tool_errors <= sec.max_retry:
                                        result = f"工具执行失败，请检查参数后重试: {tool_e}"
                                    else:
                                        result = f"FATAL ERROR: 工具连续失败 {sec.max_retry} 次！请立即停止调用工具，并向用户道歉和汇报错误：{tool_e}"

                                tool_msg = ToolMessage(content=str(result), tool_call_id=call_id)
                                history.append(tool_msg)

                            # 获取执行工具后的最新面板并塞入提示词继续推理
                            df, _ = _get_dashboard_df()
                            df_md = df.to_markdown()
                            agent_reply = stream_wrapper(sec.model_with_tools, history[:-1] + [
                                HumanMessage(content=f'<dashboard>{df_md}</dashboard>'), history[-1]])
                            history.append(agent_reply)

                        st.markdown(agent_reply.content)

                        # 保存对话并刷新界面
                        with open(today_file_path, 'w', encoding='utf-8') as f:
                            f.write(langchain_core.load.dumps(history, pretty=True, ensure_ascii=False))

                        st.session_state.history = history
                        st.rerun()

                    except Exception as e:
                        st.error("与大模型交互时发生异常（网络或API报错），已取消本次操作，请重试。")
                        # 【回滚机制触发】
                        st.session_state.history = history[:initial_history_len]

    # ---------------- 右侧栏：计划面板区 ----------------
    with right_col:
        st.subheader("📋 实时计划面板")
        df, _ = _get_dashboard_df()

        if not df.empty:
            df.fillna('', inplace=True)
            df['deadline_dt'] = pd.to_datetime(df['deadline'], format="%Y_%m_%d %H_%M", errors='coerce')
            df = df.sort_values(by=['deadline_dt'], na_position='last')

            aspect_mapping = {
                'urgent': '🚨 临期任务 (Urgent)',
                'academic': '📚 学业任务 (Academic)',
                'internship': '💼 实习任务 (Internship)',
                'everyday': '🔄 日常习惯 (Everyday)'
            }

            for aspect_key, title in aspect_mapping.items():
                aspect_df = df[df['aspect'] == aspect_key]
                if not aspect_df.empty:
                    st.markdown(f"#### {title}")
                    for _, row in aspect_df.iterrows():
                        with st.container(border=True):
                            dl_str = row['deadline'] if str(row['deadline']).strip() else "无明确截止"
                            st.markdown(f"**{row['task']}**")
                            st.caption(f"⏳ `{dl_str}`")
                            if row['description']:
                                st.markdown(f"<small style='color: gray;'>{row['description']}</small>",
                                            unsafe_allow_html=True)
        else:
            st.info("面板干干净净！告诉左侧的秘书帮你添加任务吧。")

else:
    # 历史查看模式
    st.title("🗄️ 档案查阅")
    if selected_content:
        st.info(f"当前视图：{view_mode}")
        st.markdown(selected_content)
    else:
        st.write("👈 请在左侧侧边栏选择要查看的文件")
import streamlit as st
import pandas as pd
import langchain_core.load
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.utils.logger import logger

# ================= 导入核心类和工具 (与 main.py 闭环对齐) =================
from main import Secretary
from src.LLMs.models import (
    get_former_logs, get_date_delta, get_time,
    update_dashboard, add_dashboard, remove_dashboard,
    _get_dashboard_df
)
# 🌟 修复点：导入 conclude_system_prompt 用于复习提醒
from src.LLMs.prompts import record_system_prompt, conclude_system_prompt

# 建立工具字典，方便调用
tool_dic = {tool.name: tool for tool in
            [get_former_logs, get_date_delta, get_time, update_dashboard, add_dashboard, remove_dashboard]}

st.set_page_config(page_title="Miumiu", page_icon="📝", layout="wide")

# ================= 1. 核心状态初始化 =================
logger.info(">>> 正在加载/刷新 Streamlit 前端页面...")

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
    logger.info("初始化会话状态: is_processing = False")

if "secretary" not in st.session_state:
    logger.info("首次启动应用，正在实例化 Secretary 核心类...")
    st.session_state.secretary = Secretary()
    logger.info("尝试检查并生成周报...")
    st.session_state.secretary.weekly_report()

sec = st.session_state.secretary
date_str = sec.date.strftime('%Y_%m_%d')
today_file_path = sec.chat_logs_dir / f"context_{date_str}.json"

# 加载当天的历史记录到 session_state
if "history" not in st.session_state:
    if today_file_path.exists():
        logger.info(f"发现今日聊天记录文件，正在从 {today_file_path} 加载历史上下文...")
        with open(today_file_path, 'r', encoding='utf-8') as f:
            st.session_state.history = langchain_core.load.loads(f.read())
        logger.info(f"历史记录加载完成，共 {len(st.session_state.history)} 条消息。")
    else:
        logger.info("未发现今日聊天记录，开始执行每日首次全局初始化...")
        with st.spinner(f"🌞 {date_str} {sec.weekday}！正在初始化今日日程并生成复习提醒..."):
            df, _ = _get_dashboard_df()
            df_md = df.to_markdown()

            time_anchor = f"今天是{date_str}，{sec.weekday}。"

            if sec.valid_dashboard_dates_list:
                logger.info("检测到历史面板数据，正在将面板注入初始上下文...")
                history = [SystemMessage(content=record_system_prompt)]
                instruction = time_anchor + '如下是我们最近最新的任务计划面板<dashboard>，请检查其中是否有任务即将在3日内到期，若是，请调用<dashboard>管理工具，将它的aspect应被归类为urgent'
                history += [HumanMessage(content=f"{instruction}\n\n<dashboard>\n{df_md}\n</dashboard>")]
            else:
                logger.info("未检测到任何历史面板数据，进入全新用户引导流程...")
                initial_instruction = time_anchor + "用户之前还没有跟你提起过他的任务，因此没有<dashboard>，请询问他最近有什么计划，然后做好记录"
                history = [SystemMessage(content=record_system_prompt), HumanMessage(content=initial_instruction)]

            try:
                initial_history_len = len(history)
                logger.info("正在调用大模型进行初始化推理 (Model Invoke)...")
                res = sec.model_with_tools.invoke(history)
                logger.info(f"模型响应 - content: {res.content[:100] if res.content else '(无content)'}, tool_calls数量: {len(res.tool_calls) if res.tool_calls else 0}")
                history.append(res)

                # 🔧 工具调用循环 (与用户输入处理逻辑一致)
                while res.tool_calls:
                    logger.info(f"检测到 {len(res.tool_calls)} 个工具调用请求")
                    for call in res.tool_calls:
                        call_id = call.get('id')
                        call_name = call.get('name')
                        call_args = call.get('args')
                        logger.info(f"🛠️ 执行工具: [{call_name}] | 参数: {call_args}")

                        try:
                            if call_name == 'get_former_logs':
                                call_args['chat_logs_dir'] = str(sec.chat_logs_dir)
                                result = get_former_logs.invoke(call_args)
                            elif call_name in tool_dic:
                                result = tool_dic[call_name].invoke(call_args)
                            else:
                                raise ValueError(f"不存在的工具: {call_name}")
                            logger.info(f"✅ 工具 [{call_name}] 执行成功")
                        except Exception as tool_e:
                            logger.error(f"❌ 工具 [{call_name}] 执行失败: {tool_e}")
                            result = f"工具执行失败: {tool_e}"

                        tool_msg = ToolMessage(content=str(result), tool_call_id=call_id)
                        history.append(tool_msg)

                    logger.info("工具执行完毕，获取最新dashboard并继续推理...")
                    df, _ = _get_dashboard_df()
                    df_md = df.to_markdown()
                    res = sec.model_with_tools.invoke(
                        history + [AIMessage(content=f'面板已更新：\n<dashboard>\n{df_md}\n</dashboard>')]
                    )
                    logger.info(f"模型响应 - content: {res.content[:100] if res.content else '(无content)'}, tool_calls数量: {len(res.tool_calls) if res.tool_calls else 0}")
                    history.append(res)

                # 生成友好的打招呼消息
                if not res.content:
                    logger.info("模型没有生成content，使用纯聊天模型生成打招呼消息...")
                    df, _ = _get_dashboard_df()
                    df_md = df.to_markdown()
                    greeting = sec.model.invoke(
                        history + [AIMessage(content=f'工作已完成，当前面板：\n<dashboard>\n{df_md}\n</dashboard>\n现在该跟老大打招呼并汇报情况了')]
                    )
                    logger.info(f"生成打招呼消息: {greeting.content[:100]}...")
                    history.append(greeting)

                # ========================================================
                # 🌟 修复点：老老实实补回你的 3, 7, 14, 30 天复习提醒机制
                # ========================================================
                logger.info("开始生成每日复习提醒...")
                reviews_parts = ["💡 **【Miumiu的复习提醒】**\n"]
                for i in [3, 7, 14, 30]:
                    former_log = get_former_logs.func(sec.chat_logs_dir, date_str, i)
                    context = [SystemMessage(content=conclude_system_prompt),
                               HumanMessage(content=f'<former_log>{former_log}</former_log>')]
                    logger.info(f'正在归纳 {i} 天前的复习提醒...')
                    review_res = sec.review_model.invoke(context)
                    reviews_parts.append(f"**{i}天前**：\n{review_res.content}\n")
                # ========================================================

                logger.info(f"初始化推理完成，正在保存至 {today_file_path}")
                with open(today_file_path, 'w', encoding='utf-8') as f:
                    f.write(langchain_core.load.dumps(history, pretty=True, ensure_ascii=False))
                st.session_state.history = history
                logger.info("每日首次初始化全部完成！")
            except Exception as e:
                logger.error(f"初始化今日工作失败: {e}", exc_info=True)
                st.error(f"初始化失败 (网络或API错误): {e}")
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
                selected_file = st.selectbox("选择周报日期", report_files, format_func=lambda x: x.stem)
                if selected_file:
                    logger.info(f"用户在侧边栏查阅了周报: {selected_file.name}")
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        selected_content = f.read()
            else:
                st.info("暂无生成的周报")
        else:
            st.info("暂无周报目录")

    elif view_mode == "每日日志":
        st.subheader("📅 日志列表")
        if sec.chat_logs_dir.exists():
            log_files = sorted(sec.chat_logs_dir.glob("context_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            if log_files:
                selected_log = st.selectbox("选择日志日期", log_files,
                                            format_func=lambda x: x.stem.replace('context_', ''))
                if selected_log:
                    logger.info(f"用户在侧边栏查阅了往期日志: {selected_log.name}")
                    with open(selected_log, 'r', encoding='utf-8') as f:
                        log_data = langchain_core.load.loads(f.read())
                        formatted_log = ""
                        for msg in log_data:
                            if isinstance(msg, HumanMessage):
                                content = msg.content.split('\n今天是')[0]
                                formatted_log += f"**User**: {content}\n\n"
                            elif isinstance(msg, AIMessage) and msg.content:
                                formatted_log += f"**Secretary🐱**: {msg.content}\n\n---\n\n"
                        selected_content = formatted_log
            else:
                st.info("暂无日志文件")

# ================= 3. 核心界面布局 =================
if view_mode == "当前对话":
    left_col, right_col = st.columns([0.65, 0.35], gap="large")

    # ---------------- 左侧栏：主聊天区 ----------------
    with left_col:
        st.title("📝 Theon's Secretary")
        st.caption(f"当前时间: {date_str} {sec.weekday}")

        processing_message_placeholder = st.empty()
        chat_container = st.container(height=650, border=False)

        with chat_container:
            for msg in st.session_state.history:
                if isinstance(msg, SystemMessage):
                    continue
                if isinstance(msg, ToolMessage):
                    with st.expander(f"🛠️ 工具执行完毕 (ID: {msg.tool_call_id[:6]}...)"):
                        st.caption(msg.content)
                    continue
                if isinstance(msg, AIMessage) and not msg.content:
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content
                if content:
                    display_content = content.split('\n今天是')[0]
                    display_avatar = "🐱" if role == "assistant" else "😼"
                    with st.chat_message(role, avatar=display_avatar):
                        st.markdown(display_content)

        # 2. 处理聊天输入与 Agent 执行流
        if (prompt := st.chat_input("有什么我可以帮忙的？")) and not st.session_state.is_processing:
            logger.info(f"========== 收到新用户指令: {prompt} ==========")

            with st.chat_message("user", avatar="😼"):
                st.markdown(prompt)

            history = st.session_state.history
            history.append(HumanMessage(content=prompt))

            initial_history_len = len(history)

            with processing_message_placeholder.chat_message("assistant", avatar="🐱"):
                with st.spinner("Secretary 正在思考与执行..."):
                    try:
                        st.session_state.is_processing = True
                        logger.info("锁定前端输入 (is_processing = True)")

                        logger.info("正在获取最新任务面板 <dashboard> 以注入上下文...")
                        df, _ = _get_dashboard_df()
                        df_md = df.to_markdown()

                        logger.info("==> [1/3] 第一轮大模型推理开始 (判断是否需要调用工具)...")
                        agent_reply = sec.model_with_tools.invoke(
                            history[:-1] + [HumanMessage(content=f'<dashboard>{df_md}</dashboard>'), history[-1]])
                        history.append(agent_reply)

                        consecutive_tool_errors = 0

                        # ============ 完美的工具调用循环 ============
                        while agent_reply.tool_calls:
                            logger.info(f"大模型决定调用工具，共检测到 {len(agent_reply.tool_calls)} 个工具请求。")

                            for call in agent_reply.tool_calls:
                                call_id = call.get('id')
                                call_name = call.get('name')
                                call_args = call.get('args')

                                logger.info(f"🛠️ 开始执行工具: [{call_name}] | 参数: {call_args}")

                                try:
                                    if call_name == 'get_former_logs':
                                        call_args['chat_logs_dir'] = str(sec.chat_logs_dir)
                                        result = get_former_logs.invoke(call_args)
                                    elif call_name in tool_dic:
                                        result = tool_dic[call_name].invoke(call_args)
                                    else:
                                        logger.warning(f"⚠️ 大模型尝试调用了不存在的工具: {call_name}")
                                        raise ValueError(f"不存在的工具: {call_name}")

                                    logger.info(f"✅ 工具 [{call_name}] 执行成功！返回结果: {result}")
                                    consecutive_tool_errors = 0
                                except Exception as tool_e:
                                    consecutive_tool_errors += 1
                                    logger.error(
                                        f"❌ 工具 [{call_name}] 执行异常: {tool_e} (连续失败 {consecutive_tool_errors} 次)")

                                    if consecutive_tool_errors <= sec.max_retry:
                                        result = f"工具执行失败，请检查参数后重试: {tool_e}"
                                        logger.info(
                                            f"已将失败信息返回给大模型，引导其第 {consecutive_tool_errors} 次重试。")
                                    else:
                                        result = f"FATAL ERROR: 工具连续失败 {sec.max_retry} 次！请立即停止调用工具，并向用户道歉和汇报错误：{tool_e}"
                                        logger.error("超过最大重试次数，已强行中断工具循环。")

                                tool_msg = ToolMessage(content=str(result), tool_call_id=call_id)
                                history.append(tool_msg)

                            logger.info("正在获取工具执行后的最新 <dashboard>...")
                            df, _ = _get_dashboard_df()
                            df_md = df.to_markdown()

                            logger.info("==> [2/3] 将工具结果提交给大模型，进行下一轮推理...")
                            # 提交面板
                            agent_reply = sec.model_with_tools.invoke(
                                history + [AIMessage(
                                    content=f'面板已更新，最新状态如下供参考：\n<dashboard>\n{df_md}\n</dashboard>')]
                            )

                        logger.info("==> [3/3] 大模型推理闭环，生成最终回复。")

                        # 🌟 融入了你的“工作完成后聊天”逻辑
                        chat = sec.model.invoke(
                            history + [
                                AIMessage(content=f'面板已经成功更新为\n<dashboard>\n{df_md}\n</dashboard>\n'),
                                AIMessage(
                                    content=f'我现在需要检查一下我的工作是否完成，如果确认完成，就该跟老大聊聊天了')
                            ]
                        )
                        history.append(chat)

                        logger.info("正在将今日完整历史记录落盘写入 JSON 文件...")
                        with open(today_file_path, 'w', encoding='utf-8') as f:
                            f.write(langchain_core.load.dumps(history, pretty=True, ensure_ascii=False))
                        logger.info(f"文件写入成功: {today_file_path}")

                        st.session_state.history = history
                        st.session_state.is_processing = False
                        logger.info("解锁前端输入 (is_processing = False)，触发 UI 重绘 (rerun)。")
                        logger.info("========== 本轮交互处理完毕 ==========\n")
                        st.rerun()

                    except Exception as e:
                        logger.error(f"💥 与大模型交互时发生全局灾难性异常: {e}", exc_info=True)
                        st.error("与大模型交互时发生异常（网络或API报错），已取消本次操作，请重试。")

                        logger.warning(f"执行状态回滚：将上下文截断至发生错误前 (长度 {initial_history_len})")
                        st.session_state.history = history[:initial_history_len]
                        st.session_state.is_processing = False
                        st.rerun()

    # ---------------- 右侧栏：计划面板区 ----------------
    with right_col:
        st.subheader("📋 实时计划面板")
        logger.info("--> 正在渲染右侧计划面板区...")

        dashboard_container = st.container(height=700, border=False)

        with dashboard_container:
            df, _ = _get_dashboard_df()

            if not df.empty:
                logger.info(f"成功读取到面板数据，共 {len(df)} 个任务，正在进行分类渲染。")
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
                logger.info("面板数据为空，渲染空白提示。")
                st.info("面板干干净净！告诉左侧的秘书帮你添加任务吧。")

else:
    st.title("🗄️ 档案查阅")
    if selected_content:
        st.info(f"现在视图：{view_mode}")
        st.markdown(selected_content)
    else:
        st.write("👈 请在左侧侧边栏选择要查看的文件")
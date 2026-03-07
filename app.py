import streamlit as st
import datetime
import json5
from pathlib import Path
from src.LLMs.secretary import Secretary
from src.LLMs.reporter import Reporter


# ================= 1. 工具函数 (复用 main.py 逻辑) =================

def dic2list(today_context_dic: dict):
    """把字典转为元组列表，适配 LLM 接口"""
    today_context_lst = []
    # 按索引排序，确保对话顺序正确
    indices = sorted(
        [int(k.replace('user_prompt', '')) for k in today_context_dic.keys() if k.startswith('user_prompt')])
    for i in indices:
        u_key = f'user_prompt{i}'
        a_key = f'agent_reply{i}'
        if u_key in today_context_dic:
            tup = (today_context_dic.get(u_key, ''), today_context_dic.get(a_key, ''))
            today_context_lst.append(tup)
    return today_context_lst


def early_days_time_str(now, delta: int):
    earlier_date = now - datetime.timedelta(delta)
    return earlier_date.strftime("%Y_%m_%d")


def fetch_dic(date):
    """查找指定日期的日志"""
    logs_dir_path = Path(__file__).parent / 'src' / 'chat_logs'
    if not logs_dir_path.exists():
        return {}
    for file in logs_dir_path.rglob("*"):
        if date in file.stem:
            with open(file, 'r', encoding='utf-8') as f:
                return json5.load(f)
    return {}


def get_review_content(secretary, date_now, days_delta):
    """获取复习内容"""
    target_date = early_days_time_str(date_now, days_delta)
    dic = fetch_dic(target_date)
    if dic:
        tup_lst = dic2list(dic)
        tup_lst.pop(1)
        return secretary.schedule(tup_lst, '请告诉我，我现在完成了哪些任务？分点简要罗列，不要输出任何无关的废话')
    else:
        return "无记录"


# ================= 2. 周报生成逻辑 =================

def check_and_generate_weekly_report():
    """检查是否是周一，如果是且未生成过，则生成周报"""
    date_now = datetime.datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekdays[date_now.weekday()]
    date_string = date_now.strftime("%Y_%m_%d")

    reports_dir = Path(__file__).parent / "src" / "weekly_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    current_report_path = reports_dir / f"周报_{date_string}.txt"

    # 逻辑：今天是周一 且 对应的文件还不存在
    if weekday == '周一' and not current_report_path.exists():
        with st.spinner('📊 检测到今天是周一，正在为您生成上周工作总结...'):
            try:
                reporter = Reporter()
                # 获取过去7天 (1-7)
                last_week_days = [date_now - datetime.timedelta(i) for i in range(1, 8)]
                last_week_days_logs = []

                for day in last_week_days:
                    day_str = day.strftime('%Y_%m_%d')
                    log_path = Path(__file__).parent / "src" / "chat_logs" / f"context_{day_str}.json"

                    if log_path.exists():
                        with open(log_path, 'r', encoding='utf-8') as f:
                            context = json5.load(f)
                        # 将字典转字符串存入列表
                        last_week_days_logs.append(f"【日期: {day_str}】\n{json5.dumps(context, ensure_ascii=False)}")

                if last_week_days_logs:
                    full_context = '\n\n'.join(last_week_days_logs)
                    report_content = reporter.report(full_context)

                    with open(current_report_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)

                    return report_content
                else:
                    return "上周无日志记录，跳过生成。"
            except Exception as e:
                st.error(f"周报生成失败: {e}")
    return None


# ================= 3. 核心初始化逻辑 =================

def init_app_state():
    """初始化应用状态，包含每日初始化和周报检查"""

    # 1. 实例化 Secretary
    if "secretary" not in st.session_state:
        st.session_state.secretary = Secretary()

    # 2. 路径准备
    logs_dir_path = Path(__file__).parent / "src" / 'chat_logs'
    logs_dir_path.mkdir(parents=True, exist_ok=True)
    logs_time_dic_path = logs_dir_path / "logs_time_dic.json"

    if not logs_time_dic_path.exists():
        with open(logs_time_dic_path, 'w', encoding='utf-8') as f:
            json5.dump({}, f)

    with open(logs_time_dic_path, 'r', encoding='utf-8') as f:
        logs_time_dic = json5.load(f, encoding='utf-8')

    date_now = datetime.datetime.now()
    date_string = date_now.strftime("%Y_%m_%d")

    weeks = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weeks[date_now.weekday()]

    today_file_path = logs_dir_path / f"context_{date_string}.json"

    # --- A. 周报检查 (独立于每日初始化) ---
    new_report = check_and_generate_weekly_report()
    if new_report:
        st.session_state.new_weekly_report = new_report

    # --- B. 每日初始化流程 ---
    if date_string not in logs_time_dic.keys():
        with st.spinner(f'🌞 {date_string} {weekday}！正在初始化日程...'):
            try:
                secretary = st.session_state.secretary

                # 1. 昨天回顾
                yesterday_date = early_days_time_str(date_now, 1)
                yesterday_lag = secretary.schedule(dic2list(fetch_dic(yesterday_date)),
                                                   '请告诉我，我还没完成的任务有哪些？')

                user_prompt0 = f'今天是{date_string}，{weekday}，这是我们昨天的日程对话：\n\n\n{yesterday_lag}\n\n\n请整理一下，我们今天需要接着做的事情有哪些？整理成日程表'
                agent_reply0 = secretary.schedule([], user_prompt0)

                today_context_dic = {"user_prompt0": user_prompt0, "agent_reply0": agent_reply0}

                # 临时保存
                with open(today_file_path, 'w', encoding='utf-8') as f:
                    json5.dump(today_context_dic, f, ensure_ascii=False, indent=4)

                # 2. 历史复习
                reviews_parts = ["[复习提醒] : "]
                for days, label in [(3, "三天前"), (7, "七天前"), (14, "十四天前"), (30, "一个月前")]:
                    content = get_review_content(secretary, date_now, days)
                    reviews_parts.append(f"{label} : \n\n{content}")
                reviews = "\n\n\n".join(reviews_parts)

                today_context_dic['user_prompt1'] = f'我们今天要按计划复习的内容是什么？'
                today_context_dic['agent_reply1'] = reviews

                # 3. 事务提交
                logs_time_dic[date_string] = f"context_{date_string}.json"

                with open(logs_time_dic_path, 'w', encoding='utf-8') as f:
                    json5.dump(logs_time_dic, f, ensure_ascii=False, indent=4)

                with open(today_file_path, 'w', encoding='utf-8') as f:
                    json5.dump(today_context_dic, f, ensure_ascii=False, indent=4)

                st.session_state.today_context_dic = today_context_dic

            except Exception as e:
                st.error(f"初始化失败 (网络或API错误): {e}")
                st.stop()

    # --- C. 读取已有记录 ---
    else:
        filename = logs_time_dic.get(date_string, '')
        today_file_path = logs_dir_path / filename
        if today_file_path.exists():
            with open(today_file_path, 'r', encoding='utf-8') as f:
                st.session_state.today_context_dic = json5.load(f)
        else:
            st.session_state.today_context_dic = {}

    st.session_state.today_file_path = today_file_path
    st.session_state.date_string = date_string


# ================= 4. 界面渲染 (UI) =================

st.set_page_config(page_title="Theon's Secretary", page_icon="📝", layout="wide")

# --- 侧边栏：历史档案 ---
with st.sidebar:
    st.header("🗄️ 档案室")

    view_mode = st.radio("选择查看内容:", ["当前对话", "往期周报", "每日日志"], index=0)

    selected_content = None

    if view_mode == "往期周报":
        st.subheader("📊 周报列表")
        reports_dir = Path(__file__).parent / "src" / "weekly_reports"
        if reports_dir.exists():
            report_files = sorted(reports_dir.glob("周报_*.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
            if report_files:
                selected_file = st.selectbox("选择周报日期", report_files, format_func=lambda x: x.stem)
                if selected_file:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        selected_content = f.read()
            else:
                st.info("暂无生成的周报")
        else:
            st.info("暂无周报目录")

    elif view_mode == "每日日志":
        st.subheader("📅 日志列表")
        logs_dir = Path(__file__).parent / "src" / "chat_logs"
        if logs_dir.exists():
            # 排除 logs_time_dic
            log_files = sorted(logs_dir.glob("context_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            if log_files:
                selected_log = st.selectbox("选择日志日期", log_files,
                                            format_func=lambda x: x.stem.replace('context_', ''))
                if selected_log:
                    with open(selected_log, 'r', encoding='utf-8') as f:
                        log_data = json5.load(f)
                        # 格式化展示
                        formatted_log = ""
                        indices = sorted(
                            [int(k.replace('user_prompt', '')) for k in log_data.keys() if k.startswith('user_prompt')])
                        for i in indices:
                            u = log_data.get(f'user_prompt{i}')
                            a = log_data.get(f'agent_reply{i}')
                            if u: formatted_log += f"**User**: {u}\n\n"
                            if a: formatted_log += f"**Secretary**: {a}\n\n---\n\n"
                        selected_content = formatted_log
            else:
                st.info("暂无日志文件")

# --- 主界面 ---

now_display = datetime.datetime.now()
weeks_display = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
weekday_display = weeks_display[now_display.weekday()]

st.title("📝 Theon's Secretary")
st.caption(f"当前时间: {now_display.strftime('%Y-%m-%d')} {weekday_display}")

# 新周报弹窗提示
if "new_weekly_report" in st.session_state:
    with st.expander("🎉 最新周报已生成！点击查看", expanded=True):
        st.markdown(st.session_state.new_weekly_report)
    del st.session_state.new_weekly_report

# 内容展示区域
if view_mode == "当前对话":
    # 确保初始化
    if "today_context_dic" not in st.session_state:
        init_app_state()

    # 渲染聊天
    if "today_context_dic" in st.session_state:
        context = st.session_state.today_context_dic
        indices = sorted([int(k.replace('user_prompt', '')) for k in context.keys() if k.startswith('user_prompt')])

        chat_container = st.container()
        with chat_container:
            for i in indices:
                u_p = context.get(f'user_prompt{i}')
                a_r = context.get(f'agent_reply{i}')
                if u_p:
                    with st.chat_message("user"):
                        st.markdown(u_p)
                if a_r:
                    with st.chat_message("assistant"):
                        st.markdown(a_r)

    # 输入框
    if prompt := st.chat_input("请输入任务..."):
        # 界面显示：只显示你输入的原始内容，保持清爽
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取当前索引
        context = st.session_state.today_context_dic
        current_indices = [int(k.replace('user_prompt', '')) for k in context.keys() if k.startswith('user_prompt')]
        new_ind = max(current_indices) + 1 if current_indices else 0

        # 【核心修改】后台逻辑：加上格式提醒的“紧箍咒”
        # 这段话会被存入字典和文件，也会发给 LLM，但刚才已经在界面上展示了简洁版
        augmented_prompt = prompt
        if any(i in prompt for i in ['日志', '计划', '日程', '安排', '规划', '任务']):
            augmented_prompt = prompt + f'\n今天是{now_display.strftime("%Y-%m-%d")}回忆我们一开始对你的输出要求，严格按照[临期任务]、[学业任务]、[实习任务]、[习惯养成]、[聊天]五个部分回复。'

        st.session_state.today_context_dic[f'user_prompt{new_ind}'] = augmented_prompt

        # 调用 LLM
        with st.chat_message("assistant"):
            with st.spinner("Secretary 正在思考..."):
                try:
                    # 这里的 history_lst 会包含刚刚存入的 augmented_prompt
                    history_lst = dic2list(st.session_state.today_context_dic)

                    # 发送带有“紧箍咒”的 prompt 给大模型
                    response = st.session_state.secretary.schedule(history_lst, augmented_prompt)

                    if response is None: response = "模型无响应"

                    st.markdown(response)
                    st.session_state.today_context_dic[f'agent_reply{new_ind}'] = response

                    # 保存文件
                    with open(st.session_state.today_file_path, 'w', encoding='utf-8') as f:
                        json5.dump(st.session_state.today_context_dic, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    st.error(f"Error: {e}")

else:
    # 历史查看模式
    if selected_content:
        st.info(f"正在查看：{view_mode}")
        st.markdown(selected_content)
    else:
        st.write("👈 请在左侧侧边栏选择要查看的文件")
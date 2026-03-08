#models.py
import datetime
import langchain_core.load
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolArg
from src.get_env import config
from typing import Annotated, List, Literal, Optional
from pathlib import Path
from langchain_core.messages import BaseMessage, BaseMessageChunk
import json5
import pandas as pd
from ..utils.logger import logger
import re

base_model = ChatOpenAI(
    base_url=config.LLM_API_URL,
    api_key=config.LLM_API_KEY,
    model=config.LLM_MODEL,
    max_tokens=config.MAX_TOKENS,
    max_retries=config.MAX_RETRY,
    temperature=config.TEMPERATURE,
    top_p=config.TOP_P,
    frequency_penalty=config.FREQUENCY_PENALTY
)

# 打印上下文和思考信息的包装器
def stream_wrapper(model, input_prompts, log_folder: Path = Path("logs_thinking")):
    """
    流式输出包装器 - 兼容 Tool Call 版本
    """
    if not log_folder.exists():
        log_folder.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = log_folder / f"reasoning_{timestamp}.md"

    print(f"\n{'=' * 20} MODEL STREAM START {'=' * 20}")
    print(f"--- [系统] 提示词与思考过程同步至: {log_file_path} ---\n")

    is_reasoning = False
    has_printed_content_start = False

    # 【核心修改 1】：用于自动合并所有 Chunk 的终极对象
    final_chunk: BaseMessageChunk = None

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"# Context\n**Time:** {datetime.datetime.now()}\n\n")
        # 获取上下文
        for msg in input_prompts:
            if hasattr(msg, 'dict'):
                # 如果是标准的 LangChain Message 对象
                msg_str = json5.dumps(msg.dict(), indent=2, ensure_ascii=False)
            elif isinstance(msg, tuple):
                # 如果是简写的元组 ("role", "content")
                msg_str = f"[{msg[0]}]: {msg[1]}"
            elif isinstance(msg, dict):
                # 如果是纯字典
                msg_str = json5.dumps(msg, indent=2, ensure_ascii=False)
            else:
                # 兜底：纯字符串或其他未知类型
                msg_str = str(msg)

            f.write(msg_str + "\n\n")
        f.write(f"\n\n# Thinking Process Log\n**Time:** {datetime.datetime.now()}\n\n")

        for chunk in model.stream(input_prompts):

            # 【核心修改 2】：利用 LangChain 的底层魔法，自动累加所有内容和 Tool Calls
            if final_chunk is None:
                final_chunk = chunk
            else:
                final_chunk += chunk

                # --- 提取推理逻辑 (保持你的逻辑不变) ---
            reasoning_chunk = ""
            if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
                reasoning_chunk = chunk.additional_kwargs['reasoning_content']
            elif hasattr(chunk, 'response_metadata') and 'reasoning_content' in chunk.response_metadata:
                reasoning_chunk = chunk.response_metadata['reasoning_content']

            # --- 打印思考内容 ---
            if reasoning_chunk:
                if not is_reasoning:
                    print(f"\033[90m<think>\n", end="", flush=True)
                    f.write("### <think>\n")
                    is_reasoning = True

                print(f"\033[90m{reasoning_chunk}\033[0m", end="", flush=True)
                f.write(reasoning_chunk)
                f.flush()

            # --- 打印正文内容 ---
            content_chunk = chunk.content
            if content_chunk:
                if is_reasoning and not has_printed_content_start:
                    print(f"\n</think>\033[0m\n", end="", flush=True)
                    print(f"\n--- [思考结束，生成回答] ---\n")
                    f.write("\n\n### </think>\n\n---\n\n### Final Answer\n")
                    is_reasoning = False
                    has_printed_content_start = True

                if not has_printed_content_start and not is_reasoning:
                    f.write("### Final Answer\n")
                    has_printed_content_start = True

                print(content_chunk, end="", flush=True)
                f.write(content_chunk)
                f.flush()

    print(f"\n\n{'=' * 20} MODEL STREAM END {'=' * 20}\n")

    # 【核心修改 3】：直接返回这个包含了 Content 和完整 Tool_calls 的复合 Chunk
    # LangGraph 完全兼容 MessageChunk 类型
    return final_chunk

@tool()
def get_time() -> str:
    """
    可以获取当前的星期和时间信息，格式为"%A %Y_%m_%d %H_%M_%S"
    :return:格式为"%Y_%m_%d %H_%M_%S"的当前时间
    """
    return datetime.datetime.now().strftime("%A %Y_%m_%d %H_%M_%S")


@tool()
def get_date_delta(
        date: Annotated[str, "基准日的日期字符串，应该是'%Y_%m_%d'的格式"],
        former_date: Annotated[str, "基准日期之前的某天的日期字符串，应该是'%Y_%m_%d'的格式"]
) -> int:
    """
    可以获取基准日和之前某一天之间相差的具体自然天数
    :return: 返回整数，代表相差多少天
    """
    date = datetime.datetime.strptime(date, '%Y_%m_%d')
    former_date = datetime.datetime.strptime(former_date, '%Y_%m_%d')
    delta = date-former_date
    return delta.days


@tool()
def get_former_logs(
        chat_logs_dir: Annotated[str, InjectedToolArg, "存储计划日志的文件夹路径"],
        date: Annotated[str, "基准日的日期字符串，应该是'%Y_%m_%d'的格式"],
        delta: Annotated[int, "需要几天前的计划日志就填写几"]
) -> str:
    """
    用于获取基准日date前delta天的全天计划日志，以及Theon与你的对话，包含了任务完成、调整、取消等细节信息
    :return: 基准日date前delta天的详细计划日志
    """
    former_date = datetime.datetime.strptime(date, '%Y_%m_%d') - datetime.timedelta(delta)
    if not isinstance(chat_logs_dir, Path):
        chat_logs_dir = Path(chat_logs_dir)
    log_path = chat_logs_dir / f"context_{former_date.strftime('%Y_%m_%d')}.json"
    if log_path.exists():
        with open(log_path, 'r', encoding='utf-8') as f:
            history_str = f.read()
        history: List[BaseMessage] = langchain_core.load.loads(history_str)
        result_list = []
        for i in history:
            result_list.append(f"{i.type}:\n{i.content}\n")

        return "\n".join(result_list)
    else:
        return "没有当天的计划日志文件，这说明当天没有进行工作"


# 这个东西真的非常非常适合做成数据库，这是下一步的优化方向之一
def _get_dashboard_df() -> tuple[pd.DataFrame, Path]:
    """
    通用内部函数：负责定位文件夹、加载当天的面板，或从过去的面板继承，或初始化新面板。
    返回当前的 DataFrame 和当天应该保存的 Path。
    """
    dashboard_dir_path = Path(__file__).parent.parent / 'dashboards'
    dashboard_dir_path.mkdir(parents=True, exist_ok=True)
    file_name = f"dashboard_{datetime.datetime.now().strftime('%Y_%m_%d')}.csv"
    dashboard_path = dashboard_dir_path / file_name

    # 1. 如果当天文件已存在，直接读取
    if dashboard_path.exists():
        return pd.read_csv(dashboard_path), dashboard_path

    # 2. 如果当天不存在，尝试寻找过去的面板继承
    pattern = r"\d{4}_\d{2}_\d{2}"
    valid_df_dates = []
    # 稍微严谨点，只找 csv 文件
    for file in dashboard_dir_path.glob('*.csv'):
        matches = re.findall(pattern, file.stem)
        if matches:
            valid_df_dates.append(datetime.datetime.strptime(matches[0], "%Y_%m_%d"))

    if valid_df_dates:
        valid_df_dates.sort()
        latest_file_name = f'dashboard_{valid_df_dates[-1].strftime("%Y_%m_%d")}.csv'
        inherent_df_path = dashboard_dir_path / latest_file_name
        return pd.read_csv(inherent_df_path), dashboard_path

    # 3. 如果什么都没找到，返回一个全新带表头的空 DF
    empty_df = pd.DataFrame(columns=['aspect', 'deadline', 'task', 'description'])
    return empty_df, dashboard_path


@tool()
def add_dashboard(
        aspect: Annotated[Literal["urgent", "academic", "internship", "everyday"], "任务类型"],
        deadline: Annotated[str, "截止时间，格式'%Y_%m_%d %H_%M'。everyday任务填''"],
        task: Annotated[str, "任务的简洁名称"],
        description: Annotated[str, "任务详细信息"]
):
    """
    当用户透露他增加了新任务时，调用此函数向<dashboards>增加条目。一次只能操作一个。
    """
    try:
        cur_df, dashboard_path = _get_dashboard_df()

        # 检查是否已经存在同名任务，防止重复添加
        if task in cur_df['task'].values:
            return f"添加失败：面板中已存在名为 '{task}' 的任务。如果需要修改请调用 update_dashboard。"

        new_row_df = pd.DataFrame([{
            'aspect': aspect,
            'deadline': deadline,
            'task': task,
            'description': description
        }])

        result_df = pd.concat([cur_df, new_row_df], ignore_index=True)
        result_df.to_csv(dashboard_path, index=False)
        return f"任务 '{task}' 已成功增添至面板。"
    except Exception as e:
        logger.error(f'add_dashboard 失败：\n{e}')
        return f'工具调用失败，报错信息：{e}'


@tool()
def remove_dashboard(
        task: Annotated[str, "你想要删除的任务的任务名称"]
):
    """
    当用户任务已完成或不再需要时调用，从<dashboards>中删除该任务。一次只能操作一个。
    """
    try:
        cur_df, dashboard_path = _get_dashboard_df()

        # 必须检查任务是否存在！
        if task not in cur_df['task'].values:
            return f"删除失败：未找到名为 '{task}' 的任务，请检查任务名称是否拼写正确。"

        idx = cur_df[cur_df['task'] == task].index
        result_df = cur_df.drop(idx)

        result_df.to_csv(dashboard_path, index=False)
        return f"任务 '{task}' 已成功从面板中删除。"
    except Exception as e:
        logger.error(f'remove_dashboard 失败：\n{e}')
        return f'工具调用失败，报错信息：{e}'


@tool()
def update_dashboard(
        task: Annotated[str, "需要更新信息的任务的名称"],
        aspect: Annotated[Optional[Literal["urgent", "academic", "internship", "everyday"]], "可选，更新后的任务类型"] = None,
        deadline: Annotated[Optional[str], "可选，更新后的截止时间"] = None,
        description: Annotated[Optional[str], "可选，更新后的详细信息"] = None
):
    """
    当用户任务信息发生变化时调用，更新<dashboards>中的特定字段。一次只能操作一个。
    """
    try:
        cur_df, dashboard_path = _get_dashboard_df()

        # 必须检查任务是否存在！
        if task not in cur_df['task'].values:
            return f"更新失败：未找到名为 '{task}' 的任务，请检查任务名称是否拼写准确。"

        # 使用 .loc 进行安全的字段更新
        if aspect:
            cur_df.loc[cur_df['task'] == task, 'aspect'] = aspect
        if deadline:
            cur_df.loc[cur_df['task'] == task, 'deadline'] = deadline
        if description:
            cur_df.loc[cur_df['task'] == task, 'description'] = description

        cur_df.to_csv(dashboard_path, index=False)
        return f"任务 '{task}' 的信息已成功更新。"
    except Exception as e:
        logger.error(f'update_dashboard 失败：\n{e}')
        return f'工具调用失败，报错信息：{e}'
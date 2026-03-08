import datetime
import langchain_core.load
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolArg
from src.get_env import config
from typing import Annotated, List
from pathlib import Path
from langchain_core.messages import BaseMessage, BaseMessageChunk
import json5

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
    可以获取当前的时间，格式为"%Y_%m_%d %H_%M_%S"
    :return:格式为"%Y_%m_%d %H_%M_%S"的当前时间
    """
    return datetime.datetime.now().strftime("%Y_%m_%d %H_%M_%S")


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
        return "没有当天的计划日志文件，这可能说明当天没有进行工作"



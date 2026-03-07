import datetime
import langchain_core.load
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolArg
from src.get_env import config
from typing import Annotated, List
from pathlib import Path
from langchain_core.messages import BaseMessage

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
        chat_logs_dir: Annotated[InjectedToolArg, "存储计划日志的文件夹路径"],
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



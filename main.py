#main.py
from pathlib import Path
import datetime
import langchain_core.load
import pandas as pd
from src.LLMs.models import base_model, review_model, get_former_logs, get_date_delta, get_time, update_dashboard, add_dashboard, remove_dashboard, stream_wrapper, _get_dashboard_df
from langchain_openai import ChatOpenAI
from src.LLMs.prompts import record_system_prompt, report_system_prompt, conclude_system_prompt
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import re
from typing import List
from src.utils.logger import logger


# 补丁
# ==============================================================================
# 🐒 Monkey Patch: 让 LangChain 能够识别 SiliconFlow 的 reasoning_content
# ==============================================================================
from langchain_openai.chat_models import base as langchain_openai_base

# 保存原始函数引用
_original_convert_delta = langchain_openai_base._convert_delta_to_message_chunk


def _patched_convert_delta_to_message_chunk(
        _dict, default_class
):
    # 先调用原始逻辑拿到基础 chunk
    chunk = _original_convert_delta(_dict, default_class)

    # 【核心修改】检查是否有 reasoning_content，如果有，塞进 additional_kwargs
    if "reasoning_content" in _dict:
        chunk.additional_kwargs["reasoning_content"] = _dict["reasoning_content"]

    return chunk


# 替换掉库里的函数
langchain_openai_base._convert_delta_to_message_chunk = _patched_convert_delta_to_message_chunk
# ==============================================================================


class Secretary:
    def __init__(self,
                 model: ChatOpenAI = base_model,
                 review_model: ChatOpenAI = review_model,
                 model_with_tools: ChatOpenAI = base_model.bind_tools([get_former_logs, get_date_delta, get_time, update_dashboard, add_dashboard, remove_dashboard]),
                 chat_logs_dir: Path = Path(__file__).parent / "src" / "chat_logs",
                 dashboard_dir: Path = Path(__file__).parent / "src" / "dashboards",
                 weekly_reports_dir: Path = Path(__file__).parent / "src" / "weekly_reports",
                 max_retry: int = 3
                 ):
        self.now: datetime.datetime = datetime.datetime.now()
        self.date = self._get_day()
        self.weekday = self._get_weekday()
        self.chat_logs_dir: Path = chat_logs_dir
        self.dashboards_dir: Path = dashboard_dir
        self.weekly_reports_dir: Path = weekly_reports_dir
        self.model = model
        self.review_model = review_model
        self.model_with_tools = model_with_tools
        self.chat_logs_dir.mkdir(parents=True, exist_ok=True)
        self.weekly_reports_dir.mkdir(parents=True, exist_ok=True)
        # 文件夹中所有文件的日记都在里面了
        self.valid_dashboard_dates_list = self._get_valid_dashboard_dates_list()
        self.max_retry = max_retry

    def _get_day(self):
        return self.now.date()

    def _get_weekday(self):
        weeks = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday = weeks[self.now.weekday()]
        return weekday

    def _get_valid_dashboard_dates_list(self) -> list:
        pattern = r"\d{4}_\d{2}_\d{2}"
        file_date_list: List[datetime.datetime.date] = []
        for file in self.dashboards_dir.glob("*"):
            matches = re.findall(pattern, file.stem)
            if matches:
                file_date_list.append(datetime.datetime.strptime(matches[0],"%Y_%m_%d").date())
            else:
                continue
        file_date_list.sort()
        return file_date_list

    def weekly_report(self):
        """
        方便自动生成周报
        :return: 生成周报文件
        """
        flag1 = self.weekday == '周一'
        flag2 = not any(self.date.strftime('%Y_%m_%d') in file.stem for file in self.weekly_reports_dir.glob('*'))
        if flag1 and flag2:
            week_report_path = self.weekly_reports_dir / f"周报_{self.date.strftime('%Y_%m_%d')}.txt"
            last_week_days_logs = []
            # 按照一周的自然顺序，打标签，给出全部上下文
            weekdays = ['Mon','Tues','Wed','Thurs','Fri','Sat','Sun']
            for i in range(7,0,-1):
                weekday = weekdays[7-i]
                last_week_days_logs.append(f"<{weekday}>\n{get_former_logs.func(self.chat_logs_dir, self.date.strftime('%Y_%m_%d'), i)}\n</{weekday}>")
            tmp = '\n\n'.join(last_week_days_logs)
            last_week_logs_context = f"<logs>\n{tmp}\n</logs>"
            # 判断是否有上周的dashboard
            flag = self.date-datetime.timedelta(7) <= self.valid_dashboard_dates_list[-1] < self.date
            if flag:
                dashboard_last_week_name = f"dashboard_{datetime.datetime.strftime(self.valid_dashboard_dates_list[-1], '%Y_%m_%d')}.csv"
                last_week_df = pd.read_csv(self.dashboards_dir / dashboard_last_week_name)
                df_md = last_week_df.to_markdown()
            else:
                df_md = f"未能获取上周最终的<dashboard>，可能意味着上周没有进行任务管理"
            # 目前是输入了当天的整个上下文。为了了解到具体的计划变更，这么做是相对合理的
            context_list = [
                SystemMessage(report_system_prompt),
                HumanMessage(last_week_logs_context),
                HumanMessage(f"<dashboard>{df_md}</dashboard>")
            ]
            # 这个周报现在就保存了content内容
            report = stream_wrapper(self.model, context_list).content
            with open(week_report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(report)


    def record(self):
        """
        记录每天的计划
        :return:
        """
        if not self.date in self.valid_dashboard_dates_list:
            # 初始化
            today_file_path = self.chat_logs_dir / f"context_{self.date.strftime('%Y_%m_%d')}.json"
             # 如果之前有任务记录，也就是不是第一次使用
            if self.valid_dashboard_dates_list:
                latest_date = self.valid_dashboard_dates_list[-1]
                logger.info(f'识别出最近一天使用 Secretary 在 {latest_date} ，正在尝试加载最近一天的计划面板')
                # 使用工具函数里定义的初始化来初始化，偷懒行为
                df, _ = _get_dashboard_df()
                df_md = df.to_markdown()
                logger.info(f'最近一天的计划面板加载成功')
                history = [SystemMessage(record_system_prompt)]
                instruction = f'今天是{self.date.strftime("%Y_%m_%d")}，{self.weekday}，我们最近最新的任务计划面板是<dashboard>，'
                history += [HumanMessage(instruction), HumanMessage(f"<dashboard>\n{df_md}\n</dashboard>")]

            # 如果是第一次使用
            else:
                logger.info(f'第一次使用Secretary，正在进行全局初始化')
                # 第一次
                initial_instruction = "用户之前还没有跟你提起过他的任务，因此没有<dashboard>，请询问他最近有什么计划，然后做好记录"
                history = [SystemMessage(record_system_prompt), HumanMessage(initial_instruction)]
            logger.info(f'模型初始化今天任务中...')
            res = stream_wrapper(self.model, history)
            history.append(res)
            with open(today_file_path, 'w', encoding='utf-8') as f:
                history_str = langchain_core.load.dumps(history,pretty=True,ensure_ascii=False)
                f.write(history_str)
            print(f'[进行中] : \n{res.content}')

            print(f'[复习提醒] : \n')
            for i in [3,7,14,30]:
                former_log = get_former_logs.func(self.chat_logs_dir, self.date.strftime('%Y_%m_%d'), i)
                context = [SystemMessage(conclude_system_prompt), HumanMessage(f'<former_log>{former_log}</former_log>')]
                logger.info(f'模型归纳复习提醒中...')
                print(stream_wrapper(self.model, context).content)
        else:
            # 上下文简单地管理在json文件里
            today_file_path = self.chat_logs_dir / f"context_{self.date.strftime('%Y_%m_%d')}.json"
            try:
                with open(today_file_path, 'r', encoding='utf-8') as f:
                    history_str = f.read()
                    history = langchain_core.load.loads(history_str)
            except Exception as e:
                logger.error(f'读取当天上下文时出现未知错误：{e}')
                history = [SystemMessage(record_system_prompt)]

        tool_dic = {}
        for tool in [get_former_logs, get_date_delta, get_time, update_dashboard, add_dashboard, remove_dashboard]:
            tool_dic[tool.name] = tool

        while True:
            user_prompt = input(f'您的需求？\n')
            if not user_prompt.strip():
                continue

            # 记录本轮对话开始前的 history 长度，方便崩溃回滚
            initial_history_len = len(history)

            history.append(HumanMessage(user_prompt))
            df, _ = _get_dashboard_df()
            df_md = df.to_markdown()

            try:
                agent_reply = stream_wrapper(
                    self.model_with_tools,
                    history[:-1] + [HumanMessage(f'<dashboard>{df_md}</dashboard>'),history[-1]]
                )
                history.append(agent_reply)

                consecutive_tool_errors = 0
                while agent_reply.tool_calls:
                    for call in agent_reply.tool_calls:
                        call_id = call.get('id')
                        call_name = call.get('name')
                        call_args = call.get('args')
                        logger.info(f'模型调用工具 {call_name} ')

                        try:
                            if call_name == 'get_former_logs':
                                call_args['chat_logs_dir'] = str(self.chat_logs_dir)
                                result = get_former_logs.invoke(call_args)
                            elif call_name in tool_dic:
                                result = tool_dic[call_name].invoke(call_args)
                            else:
                                logger.warning(f'未知工具调用: {call_name}')
                                # 强制进入 except
                                raise ValueError(f"不存在的工具: {call_name}")

                            # 工具执行成功，清零计数器
                            consecutive_tool_errors = 0

                        except Exception as tool_e:
                            logger.error(f'工具 {call_name} 执行期间发生错误: {tool_e}')
                            consecutive_tool_errors += 1

                            if consecutive_tool_errors <= self.max_retry:
                                result = f"工具执行失败，请检查参数后重试: {tool_e}"
                                logger.warning(f'大模型工具调用失败，已引导其进行第 {consecutive_tool_errors} 次重试')
                            else:
                                # 要求汇报
                                result = f"\n<FATAL ERROR>工具连续失败 {self.max_retry} 次！请立即停止调用工具，并向用户道歉和汇报错误：{tool_e}\n</FATAL ERROR>"
                                logger.error(f'经过 {self.max_retry} 次重试仍然失败，已强制拦截')

                        tool_msg = ToolMessage(content=str(result), tool_call_id=call_id)
                        history.append(tool_msg)

                    logger.info(f'模型获取工具结果，继续思考中...')
                    agent_reply = stream_wrapper(
                        self.model_with_tools,
                        history + [HumanMessage(f'<dashboard>{df_md}</dashboard>')]
                    )
                    history.append(agent_reply)

                print(f"\n\n{agent_reply.content}\n\n")
                # chat = stream_wrapper(base_model, history + [HumanMessage(f'请检查你的工作是否完成，另外也别光顾着工作，别忘了跟我聊天，给我点反馈')])
                # history.append(chat)
                # 写入文件
                with open(today_file_path, 'w', encoding='utf-8') as f:
                    history_str = langchain_core.load.dumps(history, pretty=True, ensure_ascii=False)
                    f.write(history_str)

            except Exception as e:
                logger.error(f"与大模型交互时发生全局异常: {e}")
                print("\n秘书的大脑刚才短路了一下（网络异常或API报错），已取消本次操作，请重试。")
                # 直接将历史记录切回本轮对话开始前的状态
                history = history[:initial_history_len]
                continue

            # 检查日期，是否应该新建log了
            date_update = datetime.datetime.now().date()
            if date_update != self.date:
                print(f'今日已经结束，新的一天开始了')
                break

def main():
    # 不会在新的一天自动初始化，不过先阶段我的使用场景下问题不大，是下一个优化点
    secretary = Secretary()
    secretary.record()


if __name__ == '__main__':
    main()
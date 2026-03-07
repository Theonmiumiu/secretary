from pathlib import Path
import datetime
import langchain_core.load
from src.LLMs.models import base_model, get_former_logs, get_date_delta, get_time
from langchain_openai import ChatOpenAI
from src.LLMs.prompts import record_system_prompt, report_system_prompt, conclude_system_prompt
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
import re
from typing import List


class Secretary:
    def __init__(self,
                 model: ChatOpenAI = base_model,
                 model_with_tools: ChatOpenAI = base_model.bind_tools([get_former_logs, get_date_delta, get_time]),
                 chat_logs_dir: Path = Path(__file__).parent / "src" / "chat_logs",
                 weekly_reports_dir: Path = Path(__file__).parent / "src" / "weekly_reports"
                 ):
        self.now: datetime.datetime = datetime.datetime.now()
        self.date = self._get_day()
        self.weekday = self._get_weekday()
        self.chat_logs_dir: Path = chat_logs_dir
        self.weekly_reports_dir: Path = weekly_reports_dir
        self.model = model
        self.model_with_tools = model_with_tools
        self.chat_logs_dir.mkdir(parents=True, exist_ok=True)
        self.weekly_reports_dir.mkdir(parents=True, exist_ok=True)
        self.valid_logs_dates_list = self._get_valid_logs_dates_list()


    def _get_day(self):
        return self.now.date()

    def _get_weekday(self):
        weeks = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        weekday = weeks[self.now.weekday()]
        return weekday

    def _early_days_time_str(self, delta: int):
        """
        获得几天前的日期的字符串
        :param delta: 几天前？
        :return: 返回这一天的日期字符串
        """
        earlier_date = self.date - datetime.timedelta(delta)
        earlier_date = earlier_date.strftime("%Y_%m_%d")
        return earlier_date

    def _get_valid_logs_dates_list(self) -> list:
        pattern = r"\b\d{4}_\d{2}_\d{2}\b"
        file_date_list: List[datetime.datetime] = []
        for file in self.chat_logs_dir.glob("*"):
            matches = re.findall(pattern, file.stem)
            if matches:
                file_date_list.append(datetime.datetime.strptime(matches[0],"%Y_%m_%d").date())
            else:
                continue
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
            for i in range(1,8):
                last_week_days_logs.append(get_former_logs.func(self.chat_logs_dir, self.date.strftime('%Y_%m_%d'), i))
            last_week_logs_context = '\n\n'.join(last_week_days_logs)
            context_list = [
                SystemMessage(report_system_prompt),
                HumanMessage(last_week_logs_context)
            ]
            report = self.model.invoke(context_list)
            with open(week_report_path, 'w', encoding='utf-8') as f:
                f.write(report.content)
            # 完整一点的话最好还要引入一下logger
            print(report)


    def record(self):
        """
        记录每天的计划
        :return:
        """
        if not self.date in self.valid_logs_dates_list:
            # 初始化
            today_file_path = self.chat_logs_dir / f"context_{self.date.strftime('%Y_%m_%d')}.json"
            # 如果之前有任务记录，也就是不是第一次使用
            if self.valid_logs_dates_list:
                self.valid_logs_dates_list.sort()
                latest_date = self.valid_logs_dates_list[-1]
                latest_date_file_path = self.chat_logs_dir / f"context_{latest_date.strftime('%Y_%m_%d')}.json"
                with open(latest_date_file_path, 'r', encoding='utf-8') as f:
                    history_str = f.read()
                yesterday_msg_list = langchain_core.load.loads(history_str)
                yesterday_msg_list.reverse()
                history = [SystemMessage(record_system_prompt)]
                for msg in yesterday_msg_list:
                    if isinstance(msg, AIMessage):
                        instruction = f'今天是{self.date.strftime("%Y_%m_%d")}，{self.weekday}，我将提供给你<昨天的日程对话>，请整理一下，我们今天需要接着做的事情有哪些？整理成日程表'
                        history += [HumanMessage(instruction), HumanMessage(f"<昨天的日程对话>\n{msg.content}\n</昨天的日程对话>)")]
                        break
                    else:
                        continue
            # 如果是第一次使用
            else:
                # 第一次
                initial_instruction = "用户之前还没有跟你提起过他的任务，请询问他最近有什么计划，然后做好记录"
                history = [SystemMessage(record_system_prompt), HumanMessage(initial_instruction)]
            res = self.model.invoke(history)
            history.append(res)
            with open(today_file_path, 'w', encoding='utf-8') as f:
                history_str = langchain_core.load.dumps(history,pretty=True)
                f.write(history_str)
            print(f'[进行中] : \n{res}')

            print(f'[复习提醒] : \n')
            for i in [3,7,14,30]:
                former_log = get_former_logs.func(self.chat_logs_dir, self.date.strftime('%Y_%m_%d'), i)
                context = [SystemMessage(conclude_system_prompt), HumanMessage(former_log)]
                print(self.model.invoke(context).content)
        else:
            # 上下文简单地管理在json文件里
            today_file_path = self.chat_logs_dir / f"context_{self.date.strftime('%Y_%m_%d')}.json"
            try:
                with open(today_file_path, 'r', encoding='utf-8') as f:
                    history_str = f.read()
                    history = langchain_core.load.loads(history_str)
            except Exception as e:
                print(f'读取当天上下文时出现未知错误：{e}')
                history = [SystemMessage(record_system_prompt)]

        while True:
            user_prompt = input(f'您的需求？')
            if any(i in user_prompt for i in ['日志','计划','日程','安排','规划','任务']):
                user_prompt = user_prompt + f'\n今天是{self.date}，回忆我们一开始对你的输出要求，严格按照[临期任务]、[学业任务]、[实习任务]、[习惯养成]、[聊天]五个部分回复。'
            history.append(HumanMessage(user_prompt))
            agent_reply = self.model_with_tools.invoke(history)
            history.append(agent_reply)
            while agent_reply.tool_calls:
                for call in agent_reply.tool_calls:
                    call_id = call.get('id')
                    call_name = call.get('name')
                    call_args = call.get('args')
                    if call_name == 'get_time':
                        result = get_time.invoke(call_args)
                    elif call_name == 'get_former_logs':
                        call_args['chat_logs_dir'] = self.chat_logs_dir
                        result = get_former_logs.invoke(call_args)
                    elif call_name == 'get_date_delta':
                        result = get_date_delta.invoke(call_args)
                    else:
                        result = f'未知工具调用'
                    tool_msg = ToolMessage(content = str(result), tool_call_id = call_id)
                    history.append(tool_msg)
                agent_reply = self.model_with_tools.invoke(history)
                history.append(agent_reply)
            print(f"\n\n{agent_reply.content}\n\n")
            with open(today_file_path, 'w', encoding='utf-8') as f:
                history_str = langchain_core.load.dumps(history,pretty=True)
                f.write(history_str)

            # 检查日期，是否应该新建log了
            date_update = datetime.datetime.now().date()
            if date_update != self.date:
                print(f'今日已经结束，新的一天开始了')
                break

def main():
    secretary = Secretary()
    secretary.record()


if __name__ == '__main__':
    main()
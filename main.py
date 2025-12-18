from pathlib import Path
import datetime
import json5
from src.LLMs.secretary import Secretary
from src.LLMs.reporter import Reporter


def dic2list(today_context_dic: dict):
    """
    方便把聊天上下文的dic转为list
    :param today_context_dic: 聊天上下文的dic
    :return: 聊天上下文的lst
    """
    today_context_lst = []
    for i in range(int(len(today_context_dic) / 2)):
        tup = (today_context_dic.get(f'user_prompt{i}', 'Warning : 没找到上下文'),
               today_context_dic.get(f'agent_reply{i}', 'Warning : 没找到上下文'))
        today_context_lst.append(tup)
    return today_context_lst


def early_days_time_str(now, delta: int):
    """
    获得几天前的日期的字符串
    :param now: 当天日期的datetime对象
    :param delta: 几天前？
    :return: 返回这一天的日期字符串
    """
    earlier_date = now - datetime.timedelta(delta)
    earlier_date = earlier_date.strftime("%Y_%m_%d")
    return earlier_date


def fetch_dic(date):
    """
    方便获得某一天的日程记录
    :param date: 日期，格式按照"%Y_%m_%d"
    :return: 日程记录的dic
    """
    logs_dir_path = Path(__file__).parent / 'src' / 'logs'
    for file in logs_dir_path.rglob("*"):
        if date in file.stem:
            print(f'已找到{date}的日程记录')
            file_path = logs_dir_path / f'context_{date}.json'
            with open(file_path, 'r', encoding='utf-8') as f:
                dic = json5.load(f)
            return dic
    print(f'未找到{date}的日程记录')
    return {}


def get_review_content(secretary, date_now, days_delta):
    """
    获取指定天数前的复习内容
    :param secretary: Secretary 实例
    :param date_now: 当前时间对象
    :param days_delta: 回溯天数
    :return: 复习内容的字符串（如果无记录则返回"无记录"）
    """
    target_date = early_days_time_str(date_now, days_delta)
    dic = fetch_dic(target_date)
    if dic:
        return secretary.schedule(dic2list(dic), '请告诉我，我现在完成了哪些任务？分点简要罗列，不要输出任何无关的废话')
    else:
        print(f'暂时没有{days_delta}天前的任务日程')
        return "无记录"

def weekly_report():
    """
    方便自动生成周报
    :return: 生成周报文件
    """
    reporter = Reporter()
    date_now = datetime.datetime.now()
    date_string = date_now.strftime("%Y_%m_%d")
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weekdays[date_now.weekday()]

    if weekday == '周一':
        weekly_reports_dir_path = Path(__file__).parent / "src" / "weekly_reports"
        if not any(date_string in file.stem for file in weekly_reports_dir_path.rglob('*')):
            week_report_path = weekly_reports_dir_path / f"周报_{date_string}.txt"
            last_week_days = [date_now - datetime.timedelta(i) for i in range(1,8)]
            last_week_days_logs = []
            for i in last_week_days:
                log_path = Path(__file__).parent / "src" / "logs" / f"context_{i.strftime('%Y_%m_%d')}.json"
                if log_path.exists():
                    with open(log_path, 'r', encoding='utf-8') as f:
                        context = json5.load(f)
                last_week_days_logs.append(str(context))
            last_week_logs_context = '\n\n'.join(last_week_days_logs)
            report = reporter.report(last_week_logs_context)
            with open(week_report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(report)


def processor():
    secretary = Secretary()
    # 说白了这只是个上下文管理系统而已
    logs_dir_path = Path(__file__).parent / "src" / 'logs'
    # 应该有一个json文件专门用来存放各个文件的日期，就是这个logs_time_dic了
    logs_time_dic_path = logs_dir_path / "logs_time_dic.json"
    with open(logs_time_dic_path, 'r', encoding='utf-8') as f:
        logs_time_dic = json5.load(f, encoding='utf-8')

    date_now = datetime.datetime.now()
    date_string = date_now.strftime("%Y_%m_%d")
    weeks = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday = weeks[date_now.weekday()]

    # 自动总结周报
    weekly_report()

    if date_string not in logs_time_dic.keys():
        # 初始化
        today_file_path = logs_dir_path / f"context_{date_string}.json"
        yesterday_date = early_days_time_str(date_now, 1)
        yesterday_lag = secretary.schedule(dic2list(fetch_dic(yesterday_date)), '请告诉我，我还没完成的任务有哪些？')
        ind = 2

        user_prompt0 = f'今天是{date_string}，{weekday}，这是我们昨天的日程对话：\n{yesterday_lag}\n\n请整理一下，我们今天需要接着做的事情有哪些？整理成日程表'
        today_context_dic = {"user_prompt0": user_prompt0}
        agent_reply0 = secretary.schedule([], user_prompt0)
        today_context_dic['agent_reply0'] = agent_reply0
        with open(today_file_path, 'w', encoding='utf-8') as f:
            json5.dump(today_context_dic, f, ensure_ascii = False)
        print(f'[进行中] : \n{agent_reply0}')


        # 整理需要定期复习的内容
        reviews_parts = ["[复习提醒] : "]
        # 定义复习周期和对应的标签
        review_periods = [
            (3, "三天前"),
            (7, "七天前"),
            (14, "十四天前"),
            (30, "一个月前")
        ]

        for days, label in review_periods:
            content = get_review_content(secretary, date_now, days)
            reviews_parts.append(f"{label} : \n{content}")

        reviews = "\n".join(reviews_parts) + "\n"
        print(reviews)

        today_context_dic['user_prompt1'] = f'我们今天要按计划复习的内容是什么？'
        today_context_dic['agent_reply1'] = reviews

        logs_time_dic[date_string] = f"context_{date_string}.json"
        with open(logs_time_dic_path, 'w', encoding='utf-8') as f:
            json5.dump(logs_time_dic, f, ensure_ascii=False)

    else:
        today_file_path = logs_dir_path / logs_time_dic.get(date_string, '')
        try:
            with open(today_file_path, 'r', encoding='utf-8') as f:
                today_context_dic = json5.load(f)
            ind = int(len(today_context_dic) / 2)

        except Exception as e:
            print(f'读取当天上下文时出现未知错误：{e}')

    while True:
        user_prompt = input(f'您的需求？')
        user_prompt = user_prompt + '\n回忆我们一开始对你的输出要求，严格按照[临期任务]、[学业任务]、[实习任务]、[习惯养成]、[聊天]五个部分回复。'
        today_context_dic[f'user_prompt{ind}'] = user_prompt
        today_context_lst = dic2list(today_context_dic)
        agent_reply = secretary.schedule(today_context_lst, user_prompt)
        today_context_dic[f'agent_reply{ind}'] = agent_reply
        print(f"\n\n{agent_reply}\n\n")
        with open(today_file_path, 'w', encoding='utf-8') as f:
            json5.dump(today_context_dic, f, indent=4, ensure_ascii = False)
        ind += 1
        # 检查日期，是否应该新建log了
        date_update = datetime.datetime.now().strftime("%Y_%m_%d")
        if date_update != date_string:
            print(f'今日已经结束，新的一天开始了')
            with open(today_file_path, 'w', encoding='utf-8') as f:
                json5.dump(today_context_dic, f, indent=4, ensure_ascii = False)
            break


def main():
    processor()


if __name__ == '__main__':
    main()
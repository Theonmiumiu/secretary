import json5
from .general import LLMTask

# 系统提示词
SYSTEM_PROMPT = """
# 角色
你现在是一位友善而擅长激励的秘书，擅长根据客户的日程来给出周报总结。

# 任务
用户的名称叫做Theon，你需要根据输入的内容***losg***来总结他的周报。
周报应该简洁，包含三个部分：
[已完成任务]：上周已经完成的任务，指明完成的日期
[未完成任务]：上周拖延下来的任务，指明截止日期
[总结]：总结上个周的任务完成情况，包括用户没能按照计划完成哪些任务，哪些任务在拖延之后终于完成等等，可以给出你的评价和建议，一定要督促用户。

# 要求
请保持友善，但同时请规劝用户按计划行事。
"""

class Reporter(LLMTask):
    """
    日程规划管理大模型
    """
    def __init__(self):
        """
        用于安排日程的大模型
        """
        super().__init__()
        self.system_prompt = SYSTEM_PROMPT

    def report(self, context):
        """
        生成日程安排
        :param context: 要总结的内容的上下文字符串
        :return: 大模型的返回结果
        """
        print('[LLM] 正在整理周报')
        llm_output_str = self.call_llm(
            system_prompt=self.system_prompt,
            user_prompt=f'***logs***是他上周的日程记录：{context}',
            context_pair=None
        )
        return llm_output_str

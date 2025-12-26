"""
General task module for LLM processing
"""
import sys
import requests
from requests import RequestException
from src.get_env import config
import time
from openai import OpenAI

class LLMTask:
    """
    为大模型节点定义基本类
    """
    def __init__(self):
        """
        参数导入
        """
        self.api_url = config.LLM_API_URL
        self.api_key = config.LLM_API_KEY
        self.model = config.LLM_MODEL
        self.max_tokens = config.MAX_TOKENS
        self.temperature = config.TEMPERATURE
        self.top_p = config.TOP_P
        self.top_k = config.TOP_K
        self.frequency_penalty = config.FREQUENCY_PENALTY

    def call_llm(self, system_prompt, user_prompt, context_pair):
        """
        阿里云调用大模型
        :param system_prompt: 系统提示词
        :param user_prompt: 用户当前提示词
        :param context_pair: 一个列表，其中为上下文，包含当前的用户提示词
        :return: 大模型调用输出
        """
        try:
            client = OpenAI(
                # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
                # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
                api_key=self.api_key,
                # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                }
            ]

            # 如果提供了上下文，则加入上下文
            if context_pair:
                for i in range(len(context_pair)):
                    messages.append(
                        {
                            "role": "user",
                            "content": context_pair[i][0],
                        }
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": context_pair[i][1],
                        }
                    )

            # 添加用户输入
            messages.append({
                "role": "user",
                "content": user_prompt,
            })

            retry = 0
            while retry<2:
                try:
                    completion = client.chat.completions.create(
                        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                        # 基本的系统提示词信息
                        messages=messages
                        )
                    return completion.choices[0].message.content
                except Exception as e:
                    print(f"调用大模型时出现未知错误：\n{e}")
                    retry += 1
        except Exception as e:
            print(f"错误信息：{e}")
            print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

    def siliconflow_call_llm(self, system_prompt, user_prompt, context_pair):
        """
        硅基流动调用大模型
        :param system_prompt: 系统提示词
        :param user_prompt: 用户当前提示词
        :param context_pair: 一个列表，其中为上下文，包含当前的用户提示词
        :return: 大模型调用输出
        """

        # 基本的系统提示词信息
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        # 如果提供了上下文，则加入上下文
        if context_pair:
            for i in range(len(context_pair)):
                messages.append(
                    {
                        "role": "user",
                        "content": context_pair[i][0],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": context_pair[i][1],
                    }
                )

        # 添加用户输入
        messages.append({
            "role": "user",
            "content": user_prompt,
        })

        # payload
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "n": 1,
        }

        # headers
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}",
        }

        ind = 0
        while ind < 2:
            try:
                #print(payload)
                #print('\n\n')
                #print(headers)
                #print('\n\n')

                response = requests.post(self.api_url, json=payload, headers=headers, timeout=600)
                response.raise_for_status()
                json_response = response.json()
                content = json_response["choices"][0]["message"]["content"]
                return content

            except RequestException as e:
                print(f"调用LLM API时发生网络或HTTP错误: {e}")
                # 如果有响应体，也记录下来，方便调试4xx/5xx错误
                if e.response is not None:
                    print(f"服务器响应: {e.response.text}")
                    if "TPM" in str(e.response.text):
                        ind += 1
                        print(f'请求超过托管商TPM限制，等待一分钟\n')
                        time.sleep(60)
                        continue
                print(f'调用LLM时发生未知错误')
                sys.exit(1)
                return None  # 返回None表示失败
            except Exception as e:
                sys.exit(1)
                print(f"调用LLM时发生未知错误: {e}")
                return None  # 返回None表示失败

        # 居然循环两次都没成功
        print(f'请检查网络设置或者LLM')
        sys.exit(1)
        return None
# 导入LangChain的提示模板和消息占位符
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime  # 导入标准库datetime，用于获取当前时间
from langchain_openai import ChatOpenAI  # 导入OpenAI聊天模型接口
from schema import AnswerQuestion, ReviseAnswer  # 导入自定义的Pydantic模型（结构化输出用）
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser  # 导入结构化输出解析器
from langchain_core.messages import HumanMessage  # 导入人类消息类型

# 注意：schema.py 需提前定义好 AnswerQuestion 和 ReviseAnswer 两个 Pydantic 模型，否则会报错。
# 导入的解析器要和后续 LLM 输出格式配套使用。

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])  # 用于将LLM结构化输出解析为Pydantic对象
parser = JsonOutputToolsParser(return_id=True)  # 用于将LLM结构化输出解析为JSON对象，并返回ID
# PydanticToolsParser 适合需要类型安全的场景，能直接转为Pydantic对象。
# JsonOutputToolsParser 适合需要原始JSON的场景，return_id=True 会保留工具调用ID。

# Actor Agent Prompt（主回答Agent的提示模板）
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat(),  # 自动插入当前时间，便于上下文感知
)
# Prompt 设计要清晰，分步指令有助于 LLM 输出结构化内容。
# MessagesPlaceholder 必须和后续输入参数一致，否则会报错。

# First Responder Prompt（首次回答Agent的专用模板）
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)
# 通过 .partial() 预填充 first_instruction，让首次回答Agent专注于“给出详细答案”。
# partial参数名要和prompt模板里的变量名一致。

llm = ChatOpenAI(model="gpt-4o")  # 初始化OpenAI聊天模型，指定gpt-4o
# 模型名称要和实际API支持的模型一致。

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice='AnswerQuestion') 
# | 是LangChain的链式操作符，先用prompt模板生成提示，再用llm生成结构化输出。
# bind_tools 绑定结构化输出工具（AnswerQuestion），并强制选择该工具。
# tool_choice 必须和tools列表里的工具名一致，否则不会生效。
# 工具的字段要和Pydantic模型完全一致。

validator = PydanticToolsParser(tools=[AnswerQuestion])  # 再次定义结构化输出解析器
# 解析器可以复用，确保和上游工具一致。

# Revisor Section（修订Agent部分）
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""
# 详细说明修订Agent的任务：根据反思意见补充/删减内容，强制加引用，控制字数。
# 指令要具体，便于LLM严格执行。

revisor_chain = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
# 用修订指令填充prompt模板，绑定修订用的结构化工具（ReviseAnswer）。
# 工具名、字段、指令要和Pydantic模型严格对应。

# 示例调用（注释掉的部分）
# response = first_responder_chain.invoke({
#     "messages": [HumanMessage("AI Agents taking over content creation")]
# })
# print(response)
# 演示如何调用首次回答链，传入用户消息，得到结构化输出。
# 实际调用时要保证输入参数格式和prompt模板一致。

# =====================
# 写代码注意点总结：
# 1. Pydantic模型要提前定义好，字段名、类型、描述要和实际需求一致。
# 2. Prompt模板变量名要和partial参数、输入参数一致，否则会报错。
# 3. 工具绑定和tool_choice要严格匹配，否则结构化输出不会生效。
# 4. 链式操作符（|）顺序不能错，先prompt再llm。
# 5. 结构化输出解析器要和工具类型一致，否则解析会失败。
# 6. 指令要具体、分步，便于LLM严格执行。
# 7. 调试时可先用print(response)查看原始输出，确保结构化内容完整。
# 8. 注意API模型名称、字段、参数兼容性，不同模型支持的结构化能力不同。
# 导入必要的库
from langchain_google_genai import ChatGoogleGenerativeAI  # Google Gemini AI模型
from dotenv import load_dotenv  # 用于加载环境变量
from langchain.agents import initialize_agent, tool  # LangChain代理相关功能
from langchain_community.tools import TavilySearchResults  # Tavily搜索工具
import datetime  # 处理日期时间

# 加载.env文件中的环境变量（如API密钥）
load_dotenv()

# 初始化LLM（大语言模型）
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") # 使用免费的gemini-1.5-flash模型, GOOGLE_API_KEY 在.env文件中

# 创建搜索工具 - 用于在网络上搜索信息
search_tool = TavilySearchResults(search_depth="basic") # TAVILY_API_KEY 在.env文件中

# 使用@tool装饰器定义自定义工具
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ 
    返回当前系统时间的工具函数
    Args:
        format: 时间格式字符串，默认为 "YYYY-MM-DD HH:MM:SS"
    Returns:
        格式化后的当前时间字符串
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

# 将所有工具组合成一个列表
tools = [search_tool, get_system_time]

# 初始化ReAct代理
# ReAct = Reasoning + Acting（推理+行动）
# 这是LangChain的零样本ReAct代理，能够：
# 1. 理解用户问题
# 2. 决定使用哪些工具
# 3. 执行工具调用
# 4. 基于结果进行推理
# 5. 重复这个过程直到得到答案
agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

# 执行代理调用
# 这个问题需要：
# 1. 搜索SpaceX最近的发射信息
# 2. 获取当前时间
# 3. 计算时间差
agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant")

"""
如何识别ReAct模式：

1. 观察输出中的思考过程：
   - 代理会显示 "Thought:" 开头的推理过程
   - 然后显示 "Action:" 表示要执行的动作
   - 接着显示 "Action Input:" 表示动作的输入参数
   - 最后显示 "Observation:" 表示工具执行的结果

2. ReAct模式的典型输出格式：
   Thought: 我需要先搜索SpaceX最近的发射信息
   Action: tavily_search_results
   Action Input: SpaceX last launch date
   Observation: [搜索结果]
   Thought: 现在我需要获取当前时间来计算时间差
   Action: get_system_time
   Action Input: {"format": "%Y-%m-%d %H:%M:%S"}
   Observation: 2024-01-15 10:30:00
   Thought: 现在我可以计算时间差了...
   Final Answer: SpaceX的最近发射是在...

3. 关键特征：
   - 代理会逐步思考问题
   - 会选择合适的工具
   - 会基于工具结果进行推理
   - 最终给出完整的答案

4. 与普通LLM的区别：
   - 普通LLM只能基于训练数据回答
   - ReAct代理可以调用外部工具获取实时信息
   - 能够执行多步骤的复杂推理过程
"""


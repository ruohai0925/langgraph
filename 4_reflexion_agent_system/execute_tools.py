# 导入必要的库和模块
import json  # 用于JSON数据的序列化和反序列化
from typing import List, Dict, Any  # 类型提示，提高代码可读性和类型安全
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, HumanMessage  # LangChain消息类型
from langchain_community.tools import TavilySearchResults  # Tavily搜索工具，用于网络搜索

# 创建Tavily搜索工具实例
# max_results=2 限制每次搜索返回最多2个结果，平衡信息量和处理效率
tavily_tool = TavilySearchResults(max_results=2)

# 核心函数：执行工具调用中的搜索查询
# 参数：state - 消息历史列表，包含对话的完整上下文
# 返回：工具消息列表，包含搜索结果
def execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    # 获取最后一条AI消息，这通常是包含工具调用的消息
    # 需要类型检查确保最后一条消息是AIMessage类型
    if not isinstance(state[-1], AIMessage):
        return []
    last_ai_message: AIMessage = state[-1]
    
    # 检查AI消息是否包含工具调用
    # 如果没有工具调用或工具调用为空，直接返回空列表
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []
    
    # 处理AnswerQuestion或ReviseAnswer工具调用，提取搜索查询
    # tool_messages列表用于存储所有生成的工具消息
    tool_messages = []
    
    # 遍历AI消息中的所有工具调用
    for tool_call in last_ai_message.tool_calls:
        # 只处理AnswerQuestion或ReviseAnswer类型的工具调用
        # 这两种工具都包含search_queries字段
        if tool_call["name"] in ["AnswerQuestion", "ReviseAnswer"]:
            # 获取工具调用的唯一ID，用于关联工具消息
            call_id = tool_call["id"]
            # 从工具调用参数中提取搜索查询列表
            # 如果search_queries不存在，默认为空列表
            search_queries = tool_call["args"].get("search_queries", [])
            
            # 使用Tavily工具执行每个搜索查询
            # query_results字典用于存储每个查询的搜索结果
            query_results = {}
            for query in search_queries:
                # 调用Tavily搜索工具执行查询
                result = tavily_tool.invoke(query)
                # 将查询和结果配对存储
                query_results[query] = result
            
            # 创建工具消息，包含搜索结果
            # 将query_results转换为JSON字符串作为消息内容
            # tool_call_id关联到原始的工具调用
            tool_messages.append(
                ToolMessage(
                    content=json.dumps(query_results),  # 搜索结果序列化为JSON
                    tool_call_id=call_id  # 关联到原始工具调用
                )
            )
    
    # 返回所有生成的工具消息
    return tool_messages

# 示例用法：演示如何使用execute_tools函数
# 创建测试状态，模拟真实的对话流程
test_state = [
    # 用户消息：提出关于小企业如何利用AI发展的问题
    HumanMessage(
        content="Write about how small business can leverage AI to grow"
    ),
    # AI消息：包含工具调用，生成搜索查询
    AIMessage(
        content="",  # 内容为空，因为这是工具调用消息
        tool_calls=[
            {
                "name": "AnswerQuestion",  # 工具名称
                "args": {
                    'answer': '',  # 答案字段（此时为空）
                    'search_queries': [
                            'AI tools for small business',  # 搜索查询1：AI工具
                            'AI in small business marketing',  # 搜索查询2：AI营销
                            'AI automation for small business'  # 搜索查询3：AI自动化
                    ], 
                    'reflection': {
                        'missing': '',  # 缺失内容反思（此时为空）
                        'superfluous': ''  # 冗余内容反思（此时为空）
                    }
                },
                "id": "call_KpYHichFFEmLitHFvFhKy1Ra",  # 工具调用唯一ID
            }
        ],
    )
]

# 执行工具调用
results = execute_tools(test_state)

# 将结果输出到文件
if results:
    # 确保content是字符串类型再进行JSON解析
    content = results[0].content
    if isinstance(content, str):
        parsed_content = json.loads(content)
        
        # 将结果写入JSON文件
        with open('search_results.json', 'w', encoding='utf-8') as f:
            json.dump(parsed_content, f, ensure_ascii=False, indent=2)
        
        print("搜索结果已保存到 search_results.json 文件")
        print("文件内容预览:")
        print(json.dumps(parsed_content, ensure_ascii=False, indent=2)[:500] + "...")
    else:
        print("Content is not a string:", content)
else:
    print("没有找到搜索结果")
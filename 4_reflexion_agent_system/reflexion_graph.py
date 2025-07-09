# 导入必要的库
from typing import List

# 导入LangChain的消息类型，用于处理AI对话中的消息
from langchain_core.messages import BaseMessage, ToolMessage
# 导入LangGraph的核心组件，用于构建AI工作流图
from langgraph.graph import END, MessageGraph

# 导入自定义的链（chain），这些是AI处理任务的具体组件
from chains import revisor_chain, first_responder_chain
# 导入工具执行函数，用于执行具体的操作
from execute_tools import execute_tools

# 创建一个消息图（MessageGraph），这是LangGraph的核心数据结构
# 它定义了AI代理（agent）如何在不同节点之间流转和处理消息
graph = MessageGraph()

# 设置最大迭代次数，防止无限循环
MAX_ITERATIONS = 2

# 向图中添加三个节点，每个节点代表AI工作流中的一个处理步骤：

# 1. "draft" 节点：初始响应者链，负责生成初步的回答
graph.add_node("draft", first_responder_chain)

# 2. "execute_tools" 节点：工具执行器，负责调用外部工具或API
graph.add_node("execute_tools", execute_tools)

# 3. "revisor" 节点：修订者链，负责检查和改进之前的回答
graph.add_node("revisor", revisor_chain)

# 添加边（edges），定义节点之间的执行顺序：
# draft -> execute_tools -> revisor
graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

# 定义条件边函数，决定revisor节点之后应该执行什么
def event_loop(state: List[BaseMessage]) -> str:
    """
    事件循环函数：决定工作流是否继续迭代还是结束
    
    参数:
        state: 当前状态，包含所有历史消息的列表
    
    返回:
        str: 下一个要执行的节点名称，或END表示结束
    """
    # 计算工具消息的数量，这代表已经执行了多少次工具调用
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    
    # 如果迭代次数超过最大限制，则结束工作流
    if num_iterations > MAX_ITERATIONS:
        return END
    
    # 否则继续执行工具调用
    return "execute_tools"

# 添加条件边，从revisor节点根据event_loop函数的返回值决定下一步
graph.add_conditional_edges("revisor", 
    event_loop,
    {
        "execute_tools": "execute_tools",  # 如果 event_loop 返回 "execute_tools", 则跳转到 execute_tools 节点
        END: END           # 如果 event_loop 返回 END, 则结束图的执行
    }
    )

# 设置工作流的入口点，从draft节点开始执行
graph.set_entry_point("draft")

# 编译图，生成可执行的应用程序
app = graph.compile()

# 打印图的可视化表示（Mermaid格式），用于调试和文档
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()          # 打印ASCII格式的图

# 测试工作流：输入一个问题，让AI代理处理
response = app.invoke(
    "Write about how small business can leverage AI to grow"
)

# 打印最终结果
# 获取最后一个响应中的工具调用结果
print(response[-1].tool_calls[0]["args"]["answer"])
# # 打印完整的响应对象，用于调试
# print(response, "response")

# 可视化
# LangGraph 在可视化（比如 Mermaid 图、ASCII 图、UI 调试工具）时，会把所有流程判断节点（即你传入的函数）也显示出来，方便你理解流程的分支和判断逻辑。
# 所以，你传入的函数名，会显示在图上。

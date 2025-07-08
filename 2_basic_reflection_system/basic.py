from typing import List, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain
import json

load_dotenv()

REFLECT = "reflect"
GENERATE = "generate"
graph = MessageGraph()

def generate_node(state): # state 是当前状态，包含所有历史消息
    """
    生成节点函数
    这是图中的一个节点，负责生成内容
    
    Args:
        state: 当前状态，包含所有历史消息
        
    Returns:
        生成链的输出结果
    """
    return generation_chain.invoke({
        "messages": state
    })


def reflect_node(state):
    """
    反思节点函数
    这个节点负责对生成的内容进行反思和改进
    
    Args:
        state: 当前状态，包含所有历史消息
        
    Returns:
        反思结果，包装成HumanMessage格式
    """
    print("当前反思输入：", state)
    response = reflection_chain.invoke({"messages": state})
    return [HumanMessage(content=response.content)] # 返回反思结果，包装成HumanMessage格式，让AI以为这是用户输入的


# 向图中添加节点
graph.add_node(GENERATE, generate_node)  # 添加生成节点
graph.add_node(REFLECT, reflect_node)    # 添加反思节点

# 设置图的入口点
graph.set_entry_point(GENERATE)


def should_continue(state):
    """
    条件边函数
    决定图执行的下一个步骤
    
    Args:
        state: 当前状态
        
    Returns:
        END: 如果消息数量超过4条，结束执行
        REFLECT: 否则继续到反思节点
    """
    print("len(state):", len(state))
    if (len(state) > 2):
        return END  # 结束执行
    return REFLECT  # 继续到反思节点


# 添加条件边 - 从GENERATE节点根据should_continue函数的返回值决定下一步
# graph.add_conditional_edges(GENERATE, should_continue)

# 新代码：添加了路径映射字典，这样就可以画出conditional edges了
graph.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        REFLECT: REFLECT,  # 如果 should_continue 返回 "reflect", 则跳转到 REFLECT 节点
        END: END           # 如果 should_continue 返回 END, 则结束图的执行
    }
)

# 添加普通边 - 从REFLECT节点直接连接到GENERATE节点
graph.add_edge(REFLECT, GENERATE)

# 编译图 - 将图结构编译成可执行的应用程序
app = graph.compile()

# 可视化图结构
print(app.get_graph().draw_mermaid())  # 生成Mermaid格式的图
app.get_graph().print_ascii()          # 打印ASCII格式的图

# 执行图 - 使用HumanMessage作为输入开始图的执行
response = app.invoke(HumanMessage(content="AI Agents taking over content creation"))

def message_to_dict(msg):
    # 如果是自定义消息对象，转成dict
    if hasattr(msg, 'to_dict'):
        return msg.to_dict()
    elif hasattr(msg, '__dict__'):
        return msg.__dict__
    else:
        return str(msg)

# 如果 response 是单个对象，先转成列表
if not isinstance(response, list):
    response = [response]

serializable_response = [message_to_dict(msg) for msg in response]

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(serializable_response, f, ensure_ascii=False, indent=2)

"""
注意事项：

1. 依赖文件：
   - 需要chains.py文件，其中定义了generation_chain和reflection_chain
   - 需要.env文件，包含必要的API密钥

2. 图执行流程：
   GENERATE → REFLECT → GENERATE → REFLECT → ... → END
   
3. 终止条件：
   - 当消息数量超过6条时自动终止
   - 这防止了无限循环

4. 消息传递：
   - 使用LangChain的Message类型进行消息传递
   - 状态在节点间自动传递

5. 节点功能：
   - generate_node: 负责内容生成
   - reflect_node: 负责内容反思和改进
   
6. 图结构：
   - 使用MessageGraph处理基于消息的图
   - 支持条件边和普通边
   - 可以可视化图结构

7. 执行模式：
   - 编译后执行，提高效率
   - 支持异步执行（如果需要）

8. 调试和监控：
   - 可以打印图结构进行调试
   - 支持Mermaid和ASCII格式的可视化

9. 性能考虑：
   - 每次循环都会调用LLM，注意API成本
   - 可以根据需要调整终止条件

10. 扩展性：
    - 可以轻松添加更多节点
    - 可以修改条件逻辑
    - 可以集成不同的LLM和工具
"""


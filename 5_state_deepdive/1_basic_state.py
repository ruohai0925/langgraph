# 导入类型注解工具TypedDict，用于定义状态的数据结构
from typing import TypedDict
# 导入LangGraph的END（结束标志）和StateGraph（状态图）
from langgraph.graph import END, StateGraph

# 定义一个简单的状态结构，只有一个整数count
class SimpleState(TypedDict):
    count: int

# 定义一个节点函数：每次调用让count加1
def increment(state: SimpleState) -> SimpleState: 
    return {
        "count": state["count"] + 1
    }

# 定义条件判断函数：决定流程是否继续
# 如果count小于5，返回"continue"，否则返回"stop"
def should_continue(state):
    if(state["count"] < 5): 
        return "continue"
    else: 
        return "stop"
    
# 创建一个状态图，指定状态类型为SimpleState
graph = StateGraph(SimpleState)

# 添加一个节点，名字叫"increment"，对应的处理函数是increment
graph.add_node("increment", increment)

# 设置流程的入口点，从"increment"节点开始
graph.set_entry_point("increment")

# 添加条件边：
# 每次执行完increment节点后，调用should_continue判断下一步
# - 如果返回"continue"，就回到increment节点（循环）
# - 如果返回"stop"，就结束流程
graph.add_conditional_edges(
    "increment", 
    should_continue, 
    {
        "continue": "increment", 
        "stop": END
    }
)

# 编译图，生成可执行的应用
graph.compile()
app = graph.compile()

# 初始化状态，count从0开始
state = {
    "count": 0
}

# 执行工作流，直到流程结束
result = app.invoke(state)
# 打印最终结果（count应该是5）
print(result)

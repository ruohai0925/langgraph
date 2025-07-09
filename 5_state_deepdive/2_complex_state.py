# 导入类型注解工具TypedDict、List、Annotated
from typing import TypedDict, List, Annotated
# 导入LangGraph的END（结束标志）和StateGraph（状态图）
from langgraph.graph import END, StateGraph
# 导入operator模块，用于后续的注解
import operator

# 定义一个更复杂的状态结构，包含三个字段：
# - count: 当前计数
# - sum: 累加和，带有operator.add注解（可用于自动聚合）
# - history: 计数历史列表，带有operator.concat注解（可用于自动拼接）
class SimpleState(TypedDict):
    count: int
    sum: Annotated[int, operator.add]
    history: Annotated[List[int], operator.concat]

# 节点函数：每次让count加1，并更新sum和history
# 注意：这里只是演示，sum和history的聚合并未用到注解的自动聚合特性
# 实际上，sum和history的更新逻辑需要你自己实现

def increment(state: SimpleState) -> SimpleState: 
    new_count = state["count"] + 1
    return {
        "count": new_count, 
        "sum": new_count,  # 这里只是简单赋值，未做累加
        "history": [new_count]  # 这里只是新建一个列表，未做历史拼接
    }

# 条件判断函数：决定流程是否继续
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
app = graph.compile()

# 初始化状态，count为0，sum为0，history为空列表
state = {
    "count": 0, 
    "sum": 0, 
    "history": []
}

# 执行工作流，直到流程结束
result = app.invoke(state)
# 打印最终结果
print(result)

# 重要知识点：
# 1. Annotated类型+operator.add/operator.concat的用法，理论上可以让LangGraph自动聚合状态（如自动累加sum、拼接history），但本例中未用到自动聚合特性。
# 2. 如果想让sum/history自动累加/拼接，需要用LangGraph的高级聚合机制，或在节点函数里手动实现。
# 3. 这种状态结构适合需要追踪多种信息的复杂流程。
# 4. 每次increment只会把sum/history重置为当前值/单元素列表，实际聚合需手动实现。

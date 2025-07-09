# 导入Pydantic库，用于创建数据验证和序列化模型
from pydantic import BaseModel, Field
# 导入类型提示，用于定义列表类型
from typing import List


class Reflection(BaseModel):
    """
    反思模型：用于AI对自身答案的批判性分析
    包含两个关键维度：缺失内容和冗余内容
    """
    # 缺失内容批判：指出答案中缺少的重要信息或观点
    missing: str = Field(description="Critique of what is missing.")
    # 冗余内容批判：指出答案中不必要的、重复的或无关的内容
    superfluous: str = Field(description="Critique of what is superfluous") 

class AnswerQuestion(BaseModel):
    """
    回答问题模型：定义AI回答问题的结构化输出格式
    包含答案、搜索查询和反思三个核心组件
    """

    # 核心答案：约250字的详细回答，确保简洁而全面
    answer: str = Field(
        description="~250 word detailed answer to the question.")
    
    # 搜索查询列表：1-3个用于改进答案的搜索关键词
    # 这些查询基于对当前答案的批判，用于寻找补充信息
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )
    
    # 反思组件：使用Reflection模型对初始答案进行批判性分析
    # 帮助识别答案的不足和改进方向
    reflection: Reflection = Field(
        description="Your reflection on the initial answer.")
    
class ReviseAnswer(AnswerQuestion): # 修订答案模型, 继承AnswerQuestion模型, 在原有基础上增加了引用列表，确保修订的可信度和可验证性
    """
    修订答案模型：继承AnswerQuestion，用于AI修订和改进答案
    在原有基础上增加了引用列表，确保修订的可信度和可验证性
    """

    # 引用列表：为修订后的答案提供支撑的参考文献或来源
    # 确保答案的可信度，便于用户验证信息的准确性
    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )
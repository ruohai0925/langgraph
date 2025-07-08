from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# 定义生成提示模板
# 这个模板用于指导AI生成Twitter帖子
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",  # 系统角色提示
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
            # 翻译：你是一个Twitter科技影响者助手，负责撰写优秀的Twitter帖子。
            # 为用户的要求生成最好的Twitter帖子。
            # 如果用户提供批评，请用你之前尝试的修订版本来回应。
        ),
        MessagesPlaceholder(variable_name="messages"),  # 消息占位符，用于插入对话历史
    ]
)

# 定义反思提示模板
# 这个模板用于指导AI对生成的Twitter帖子进行批判性分析
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",  # 系统角色提示
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
            # 翻译：你是一个病毒式传播的Twitter影响者，正在评估一条推文。
            # 为用户推文生成批评和建议。
            # 始终提供详细的建议，包括长度、病毒性、风格等要求。
        ),
        MessagesPlaceholder(variable_name="messages"),  # 消息占位符，用于插入对话历史
    ]
)

# 初始化LLM（大语言模型）
# 使用OpenAI的GPT-4o模型作为推理引擎
llm = ChatOpenAI(model="gpt-4o")

# 创建生成链
# 将生成提示模板与LLM组合，形成可执行的生成链
# 这个链负责根据用户输入和对话历史生成Twitter帖子
generation_chain = generation_prompt | llm

# 创建反思链
# 将反思提示模板与LLM组合，形成可执行的反思链
# 这个链负责对生成的Twitter帖子进行批判性分析和改进建议
reflection_chain = reflection_prompt | llm

"""
代码功能说明：

1. 提示模板设计：
   - generation_prompt: 专门用于生成Twitter帖子的提示模板
   - reflection_prompt: 专门用于反思和改进的提示模板
   - 两个模板都使用MessagesPlaceholder来保持对话上下文

2. 角色定义：
   - 生成链扮演"Twitter科技影响者助手"角色
   - 反思链扮演"病毒式传播的Twitter影响者"角色
   - 这种角色分离确保了专业性和客观性

3. 链的组合：
   - 使用管道操作符 "|" 将提示模板与LLM组合
   - 形成可重用的处理链
   - 支持链式调用和组合

4. 消息传递：
   - MessagesPlaceholder确保对话历史在链间传递
   - 支持多轮对话和上下文保持

"""

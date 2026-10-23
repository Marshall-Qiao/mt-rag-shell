from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, chat

prompt_q_and_a = chat.ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
        You are an expert financial analyst. Use you knowledge base to answer questions about audited financial statements. Please note: 
        1. Always answer questions in CHINESE 
        2. If you can't answer from the given context, please state that you don't know and explain why you think the given context is not relevant.
        3. If you can find relevant information from the given context, please include brief references at the end of your answer.
        """
        ),
        HumanMessagePromptTemplate.from_template(
            """
              Context: {context}
              Question: {input}
            """
        ),
    ]
)

prompt_no_docs_found = chat.ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
        You are an expert financial analyst. Use you knowledge base to answer questions about audited financial statements. Please note: 
        1. Always respond in CHINESE.
        2. Provide a thorough, accurate, and informative answer.
        3. Indicate that the information is based on general knowledge
        4. Mention that the information may not be up-to-date
        5. Maintain a friendly and helpful tone throughout your response.
        """
        ),
        HumanMessagePromptTemplate.from_template(
            """
              Context: {context}
              Question: {input}
            """
        ),
    ]
)

prompt_text_completion = chat.ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
        You are an assistant that helps users to write articles based on given context.
        And based on current input, you are going to complete the sentence or paragraph.
        This article should be in Chinese.
        """
        ),
        HumanMessagePromptTemplate.from_template(
            """
              context: {context}
              input: {input}
            """
        ),
    ]
)


prompt_doc_grader = chat.ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are a grader assessing relevance of retrieved document(s) to a user question. \n 
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        ),
        HumanMessagePromptTemplate.from_template(
            """
              Here is the retrieved document(s): \n\n {context} \n\n
              Here is the user question: {question} \n
            """
        ),
    ]
)

prompt_question_timeliness = chat.ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
        You are a financial and investment assistant with expertise in evaluating the time sensitivity of questions. Your task is to analyze the given text and determine its relevance level on a scale from 0 to 4, based on how time-sensitive the question is for investors and market participants.
        - Score 4: **Very High Timeliness** – The question relates to immediate market dynamics, breaking news, recent earnings reports, or same-day market movements, requiring information from within the past 48 hours. Examples: “Why did the market go up today?” “What caused a particular stock to surge yesterday?”
        - Score 3: **High Timeliness** – The question focuses on significant recent events or data, such as a company’s earnings report, industry news, or economic developments within the past two weeks. Questions of this type typically require information from within the last 14 days. Examples: “Why did the market react strongly to the recent jobs report?” “What’s your take on ASML’s earnings report from last week?”
        - Score 2: **Moderate Timeliness** – The question concerns trends in the economy, policy, or industry, where timeliness matters but is not urgent, generally requiring data from within the past three months. This includes questions on broader market trends or policy expectations. Examples: “How should I position my assets in a rate-cutting environment?” “Why has Tesla’s stock been performing a certain way recently?”
        - Score 1: **Low Timeliness** – The question addresses relatively stable topics that could use information from within the past year. Such questions may involve long-term investment trends or company fundamentals. Examples: “Is SNOW a more reliable long-term investment than PLTR?” “Is TLT suitable for recession hedging at this point?”
        - Score 0: **No Timeliness Requirement** – The question does not relate to any time-sensitive event, often covering general investment knowledge or strategy. This type of question is not time-bound and can be answered with information from any period. Examples: “What is Buffett’s investment philosophy?” “How can I manage my investment mindset?”

        """
        ),
        HumanMessagePromptTemplate.from_template(
            """
              Here is the user question: {question} \n
            """
        ),
    ]
)

prompt_question_timeliness = chat.ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""
        You are a financial and investment assistant with expertise in evaluating the time sensitivity of questions. Your task is to analyze the given text and determine its relevance level on a scale from 0 to 4, based on how time-sensitive the question is for investors and market participants.
        - Score 4: **Very High Timeliness** – The question relates to immediate market dynamics, breaking news, recent earnings reports, or same-day market movements, requiring information from within the past 48 hours. Examples: “Why did the market go up today?” “What caused a particular stock to surge yesterday?”
        - Score 3: **High Timeliness** – The question focuses on significant recent events or data, such as a company’s earnings report, industry news, or economic developments within the past two weeks. Questions of this type typically require information from within the last 14 days. Examples: “Why did the market react strongly to the recent jobs report?” “What’s your take on ASML’s earnings report from last week?”
        - Score 2: **Moderate Timeliness** – The question concerns trends in the economy, policy, or industry, where timeliness matters but is not urgent, generally requiring data from within the past three months. This includes questions on broader market trends or policy expectations. Examples: “How should I position my assets in a rate-cutting environment?” “Why has Tesla’s stock been performing a certain way recently?”
        - Score 1: **Low Timeliness** – The question addresses relatively stable topics that could use information from within the past year. Such questions may involve long-term investment trends or company fundamentals. Examples: “Is SNOW a more reliable long-term investment than PLTR?” “Is TLT suitable for recession hedging at this point?”
        - Score 0: **No Timeliness Requirement** – The question does not relate to any time-sensitive event, often covering general investment knowledge or strategy. This type of question is not time-bound and can be answered with information from any period. Examples: “What is Buffett’s investment philosophy?” “How can I manage my investment mindset?”

        """
        ),
        HumanMessagePromptTemplate.from_template(
            """
              Here is the user question: {question} \n
            """
        ),
    ]
)


def get_prompt(prompt: str):
    if prompt == "Q_A":
        return prompt_q_and_a
    elif prompt == "TEXT_COMPLETION":
        return prompt_text_completion

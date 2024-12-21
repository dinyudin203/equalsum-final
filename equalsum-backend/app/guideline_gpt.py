import re
import json
import os
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# 이 위에 키를 os.environ["OPENAI_API_KEY"] ="" 형식으로 불러오셔야합니다.
model = ChatOpenAI(model="gpt-4", temperature=0.5)

def text_to_json(text):
    if text:
      cleaned_text = re.sub(r"\n", " ", text.strip())  
      cleaned_text = re.sub(r"\"", "", cleaned_text) 
      cleaned_text = re.sub(r",", "", cleaned_text) 
      cleaned_text = re.sub(r" +", " ", cleaned_text) 
      result = {"Guideline": cleaned_text.strip()}
    else:
      result = {"Guideline": None}
    return result
def key_result_query(company_name, department_name, objective, context):
    def generate_key_query_prompt(company_name, department_name, objective, context):
        prompt_template = f"""
            # Guideline
            ## Background
            You are a Korean OKR expert.
            Read the content of the {context} of the {company_name} company and familiarize yourself with the company's goals.
            After that, you will establish Key Results for a department named {department_name}.
            The department's OKRs must be subordinate goals that contribute to achieving the company's goals as outlined in {context}.
            The Objective of the department is '{objective}'.

            ## Instruction
            1. Create **3 critical and detailed questions** that will help the department generate effective Key Results aligned with the Objective.
            2. For each question, provide **follow-up questions** related to the following evaluation criteria:
                - **Connectivity**: How well the Key Result is connected to the Objective.
                - **Measurability**: Whether the Key Result can be quantitatively measured (e.g., subjects and criteria for measurement).
                - **Directivity**: Whether the Key Result represents tangible results (output/outcome) rather than activities or inputs.
            3. Ensure the follow-up questions are tailored to the specific context of each main question, avoiding generic or repetitive phrasing.
            4. Should include some specific examples in the follow-up questions.

            ## Example Format
            The output must be in the following format:
            Question: "Nth critical question to generate Key Results?",
            Connectivity: "Follow-up question related to Connectivity?",
            Measurability: "Follow-up question related to Measurability?",
            Directivity: "Follow-up question related to Directivity?"

            ## Output Rules
            - Output must be in Korean.
            - Do not include any additional comments or explanations.
            - Ensure all questions are well-aligned with the department's Objective and follow the above format exactly.

            # Output
            Please generate the output in the specified format with 3 questions.
        """
        return prompt_template

    # LangChain ChatOpenAI 모델 호출
    key_query_prompt = generate_key_query_prompt(company_name, department_name, objective, context)
    key_query_response = model([
        SystemMessage(content="You are an OKR AI assistant."),
        HumanMessage(content=key_query_prompt)
    ])

    # 응답 처리
    response = key_query_response.content
    json_response = text_to_json(response)

    return json_response

# # 테스트 실행
# company_name = "CJ씨푸드"
# department_name = "우리맛연구팀"
# objective = "간편한 한끼 식사로 균형잡힌 영양을 제공한다"
# context = """
# 곡류, 고기나 생선류, 채소류, 과일류, 유제품류, 유지류 등 총 6가지의 식품으로 짜인 밥상이 균형 잡힌 식단이라고 볼 수 있습니다.
# 이는 각 식품군이 우리 몸에 필요한 다양한 영양소를 제공하기 때문입니다.
# 특히 간편한 한 끼 식사를 위해서는 모든 영양소를 고루 섭취하는 균형 잡힌 식단이 중요합니다.
# 균형 잡힌 식사는 단순히 맛있게 먹는 것을 넘어 건강을 유지하고 질병을 예방하는 데 중요한 역할을 합니다.
# 따라서 식료품을 선택할 때는 단순히 맛이나 가격뿐만 아니라 영양적인 균형도 고려해야 합니다.
# """
# result = key_result_query(company_name, department_name, objective, context)
# print(result)

def objective_query(company_name, department_name, upper_objective, context):
    def generate_o_query_prompt(company_name, department_name, upper_objective, context):
      prompt_template = f"""
        # Guideline
        ## Background
        You are a Korean OKR expert.
        Read the {context} content of the {company_name} company and familiarize yourself with the company's goals.
        After that, you will establish Objectives for a department named {department_name}.
        The department's OKRs must be subordinate goals that contribute to achieving the company's goals as outlined in {context}.
        The Upper-Objective of the department is "{upper_objective}".

        ## Instruction
        1. Create **3 critical and detailed questions** that will help the department generate effective Objectives aligned with the Upper-Objective.
        2. For each question, provide **follow-up questions** related to the following evaluation criteria:
            - **Align**: How well the Objective is connected to the Upper-Objective, specifying how the connection can be demonstrated.
            - **Customer Value**: What specific customer problem the Objective solves, and how it creates value for the customer, with actionable and practical considerations.
        3. Ensure the follow-up questions are tailored to the specific context of each main question, avoiding generic or repetitive phrasing.
        4. Should include some specific examples in the follow-up questions.

        ## Output Format
        The output must be in the following format:
        Question: "Nth critical question to generate Objectives?",
        Align: "Follow-up question related to Align?",
        Customer Value: "Follow-up question related to Customer Value?"

        ## Output Rules
        - Output must be in Korean.
        - Do not include any additional comments or explanations.
        - Ensure all questions are well-aligned with the department's Upper-Objective and follow the above format exactly.

        # Output
        Please generate the output in the specified format with 3 questions.
      """
      return prompt_template

    # Generate Prompt
    o_query_prompt = generate_o_query_prompt(company_name, department_name, upper_objective, context)

    # LangChain ChatOpenAI 모델 호출
    o_query_response = model([
        SystemMessage(content="You are an OKR AI assistant."),
        HumanMessage(content=o_query_prompt)
    ])

    # 응답 처리
    response = o_query_response.content
    print(response)
    json_response = text_to_json(response)

    return json_response

# # 테스트 실행
# upper_objective = "우리 집밥이 좀 더 쉬워지고, 맛있어지고 건강해지도록 한다."
# print(objective_query(company_name, department_name, upper_objective, context))


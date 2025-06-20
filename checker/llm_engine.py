import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def run_check(code: str) -> str:
    # 1. 룰셋 로드
    rule_path = os.path.join("data", "java_rule_set_essential.xlsx")
    df = pd.read_excel(rule_path)

    # 2. 주요 컬럼만 필터링 후 NaN 제거
    rule_df = df[["PRIORITY", "RULE_NAME", "RULE_DESC"]].dropna()

    # 3. 각 규칙을 텍스트로 변환
    rule_text = "\n".join(
        f"- 우선순위: [{row.PRIORITY}], 규칙명: [{row.RULE_NAME}], 설명: {row.RULE_DESC}"
        for _, row in rule_df.iterrows()
    )

    # 4. 사용자 프롬프트 불러오기
    with open("prompt/commit_prompt", "r", encoding="utf-8") as f:
        user_query = f.read()

    # 5. 전체 프롬프트 구성
    full_prompt = f"""
다음은 Java 룰셋입니다:

{rule_text}

{user_query}

[검사 대상 Java 코드]
{code}
    """

    # 6. OpenAI API 호출
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 Java 룰셋 검사 전문가입니다."},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0,
    )

    return response.choices[0].message.content.strip()
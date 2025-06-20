import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from checker.llm_engine import run_check
from dotenv import load_dotenv
load_dotenv()

# 1. 현재 Git에서 커밋 대기 중인 Java 파일 불러오기
def get_staged_java_files():
    import subprocess
    result = subprocess.run(["git", "diff", "--cached", "--name-only"], stdout=subprocess.PIPE)
    files = result.stdout.decode("utf-8").splitlines()
    return [f for f in files if f.endswith(".java")]

# 2. 전체 Java 코드 로딩
def load_code(files):
    contents = ""
    for file in files:
        with open(file, 'r') as f:
            contents += f"\n\n// File: " + file + "\n" + f.read()
    return contents

if __name__ == "__main__":
    java_files = get_staged_java_files()
    if not java_files:
        print("✅ 검사 결과: Java 파일 없음 — 커밋 허용")
        sys.exit(0)

    code = load_code(java_files)
    result = run_check(code)  # => 이 함수는 GPT에 룰 검사 요청
    print(result)

    # 룰 위반 여부 판별
    if "✅ 검사 결과: 위반된 규칙이 없습니다" in result:
        sys.exit(0)  # commit 허용
    else:
        # 검사 결과 저장 (Slack 전송에서 활용)
        with open(".check_result.txt", "w") as f:
            f.write(result)
        sys.exit(1)  # commit 중단
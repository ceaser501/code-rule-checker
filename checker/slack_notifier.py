import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

# .check_result.txt가 없으면 종료
if not os.path.exists(".check_result.txt"):
    print("⚠️ Slack 알림 생략: 검사 결과 파일이 없습니다.")
    exit(0)

# 검사 결과 로딩
with open(".check_result.txt", "r") as f:
    result = f.read()

# ✅ 룰 위반 없음이면 Slack 생략
if "✅ 코드에는 위반된 규칙이 없습니다." in result:
    print("✅ 위반 없음: Slack 알림 생략")
    sys.exit(0)

# Slack Webhook 전송
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
if not SLACK_WEBHOOK_URL:
    print("❌ SLACK_WEBHOOK_URL 누락")
    exit(1)

payload = {
    "text": result
}

response = requests.post(SLACK_WEBHOOK_URL, json=payload)

if response.status_code == 200:
    print("📤 Slack 알림 전송 완료")
    os.remove(".check_result.txt")
else:
    print(f"❌ Slack 전송 실패: {response.status_code}")
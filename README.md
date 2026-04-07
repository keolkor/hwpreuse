# hwpreuse
hwpx문서 양식을 재사용하는 프로그램
requirements.txt 파일에 의존패키지 정보 있음

openai에 맞춰서 만들어 짐
your_api_key에 발급받은 api를 입력하고 실행하면 됨
client = OpenAI(
    api_key="your_api_key",
    base_url="https://api.openai.com/v1/chat/completions"
)

업로드된 양식을 그대로 사용 -> AI가 문단.표 위치를 기반으로 변경 -> AI에 요청사항 전달 -> JSON 응답형태 강제
-> JSON 응답내용 업로드된 양식에 적용 

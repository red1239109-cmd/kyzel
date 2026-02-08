import logging
from src.types.session import Session
from src.types.events import HumanMsg

# 만든 '신형 엔진'들 Import
from src.model import LLM                 # src/model/llm.py
from src.agent import AgentLoop           # src/agent/loop.py

# 로그 설정 (터미널에 진행 상황을 보여줌)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
log = logging.getLogger("main")

def main():
    # 1. 뇌(LLM) 준비
    # (vLLM이 없으면 자동으로 가짜 모드(Mock)로 돌아가니 걱정 NO)
    llm = LLM(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    # 2. 기억(Session) 준비
    session = Session.create()
    
    # 3. 사용자 명령 입력
    user_query = "파이썬으로 피보나치 수열 10개를 구하는 코드를 짜고 실행해줘."
    session.add(HumanMsg(content=user_query))
    
    log.info(f"사용자 명령: {user_query}")

    # 4. 에이전트(Loop) 소환 및 실행
    # (LLM과 Session을 연결해서 스스로 생각하고 행동하게 만듦)
    agent = AgentLoop(llm=llm) 
    
    log.info("에이전트 가동 시작! 🚀")
    agent.run(session) # <--- 여기서 마법이 일어납니다
    
    log.info("에이전트 임무 완료.")

if __name__ == "__main__":
    main()

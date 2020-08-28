# Money.fin AI
## 구동 방법
개발 후 적습니다

## 가이드
- 각자 구현할 기능을 issue에 생성합니다

- 각각 기능마다 issue를 생성하고 라벨을 붙이며 branch를 생성합니다

- project에서 칸반형식을 따라 tracking을 합니다

- 구현시 develop branch에 PR를 날립니다

- reviewer에 상대방을 assign 하고 상대방은 code review를 해줍니다

- review가 끝나면 merge 합니다

- 이번 프로젝트 reviewer는 코드를 작성한 사람 외에 Backend repo에 있는 모두를 뜻합니다

## Branch 규칙
- Branch 생성 규칙은 다음과 같습니다
```
issue{이슈번호}_{이슈와 관련된 주제}
```

## Commit 규칙

- commit 규칙은 다음과 같습니다 ([참조](https://www.conventionalcommits.org/ko/v1.0.0-beta.4/))
```
<타입>[문제번호]:<설명> // 최대 72자

[본문(선택사항)]

[꼬리말(선택사항)]
```

**예시**
- code review 적용
    - improvement(api): 변경내용
-  test code 작성
    - test(function): 무슨 내용에 대한 테스트
- 버그 수정
    - fix(api): 버그 수정

**설명**
- 줄에는 간략하게 서술한다

**타입의 종류**

| Type  | Description 
|:--------|:--------|
| improvement | 버그 수정이나 기능 추가 없이 현재 구현체를 개선 |
| fix |  bug를 찾아서 수정| 
| test | 테스트 코드작성 |
| feat | 기능추가|

## Issue 생성 가이드
- 기본적인 템플릿에 따라 작성
- 구현해야할 기능 구체적으로 설명
- 적절한 라벨링과 Project에 적용
- 생성할 떄 자기 자신한테 assign합니다

## PR 가이드
- 중점적으로 봐줬으면 하는 내용 서술
- 질문이나 개선사항 서술
- 본문 끝에 closes #{이슈번호}를 사용하여 PR에 이슈가 언급되게 함

# Reference
- [Korean embedding](https://github.com/Kyubyong/wordvectors)

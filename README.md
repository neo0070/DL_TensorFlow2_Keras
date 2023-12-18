# DL_TensorFlow2_Keras

  

## 소개

Deep Learning with TensorFlow 2 and Keras, 2nd Edition: Regression, ConvNets, GANs, RNNs, NLP, and more with TensorFlow 2 and the Keras API

  

### 회의 진행/모임 방식

- 매주 금요일 오후 7시에 정기적으로 회의 합니다.
  - Zoom을 사용하여 온라인 회의를 진행 합니다.
  - 요청에 의해 일정은 변경될 수 있습니다.
- 한달에 1회 offline 미팅을 진행 합니다.

  

## 프로젝트 범위와 단계

- Deep Learning with TensorFlow 2 and Keras, Second Edition
- GitHub issue에서 관리

  

## 폴더 구조

- packages/ : 최종 소스
- docs/ : 산출물과 매뉴얼
  
  - chapter_1 ~ chapter_16
- laboratory/: 개인별 연구 공간
  - pnuskgh: 김계현
  - neo0070: 임창현
  - naver: 박진
  - skc: 발걸음
  - sds: 채종호

  

## 개발 환경

- Python 3.10.6​

  

### 커뮤니케이션

- 카카오톡
- Zoom : 온라인 미팅
- [GitHub](https://github.com/neo0070/DL_TensorFlow2_Keras) : 소스 관리

  

### Code Convention

- ESLint 사용
- Prettier 사용
- [Python Code Convention](https://scshim.tistory.com/609)

  

### Git Convention

- 형상 관리를 위한 branch 전략
  - master : 제품으로 출시될 수 있는 브랜치
  - develop : 다음 출시 버전을 개발하는 브랜치
  - feature : 기능을 개발하는 브랜치
  - release : 이번 출시 버전을 준비하는 브랜치
  - hotfix : 출시 버전에서 발생한 버그를 수정 하는 브랜치
- Merge Request > 동료 Review > Merge
- Commit message 규칙
  - [Type] commit message
  - type
    - feature : 새로운 기능 추가
    - fix : 버그 수정
    - docs : 문서 업데이트
    - style : frontend의 style 수정
    - refactor : 코드의 리팩토링
    - test : 테스트코드 업데이트
    - env : 환경 구축

  

## 작업 단계와 계획

- [GitHub 사이트](https://github.com/neo0070/DL_TensorFlow2_Keras/tree/develop)

  

| 회차 |     일자      | 발표자 | 비고                                      |
| :--: | :-----------: | :----: | ----------------------------------------- |
|      | 2023.06.23 금 |        | 첫 온라인 미팅                            |
|      | 2023.07.04 화 |        | 제1차 오프라인 모임                       |
|  1   | 2023.07.14 금 | 임창현 | 1장. 텐서플로 2.0으로 신경망 구성         |
|  2   | 2023.07.19 수 | 채종호 | 2장. 텐서플로 1.x와 2.x                   |
|  3   | 2023.07.28 금 | 임창현 | 3장. 회귀                                 |
|  4   | 2023.08.03 목 | 김계현 | 4장. 컨볼루션 신경망                      |
|  5   | 2023.08.11 금 | 김계현 | 4장. 컨볼루션 신경망                      |
|  6   | 2023.08.18 금 | 김계현 | 5장. 고급 컨볼루션 신경망                 |
|  7   | 2023.08.25 금 | 김계현 | 5장. 고급 컨볼루션 신경망                 |
|  8   | 2023.08.31 목 | 김지훈 | 6장. 생성적 적대 신경망                   |
|  9   | 2023.09.08 금 | 김지훈 | 6장. 생성적 적대 신경망                   |
|      | 2023.09.15 금 |        | 제2차 오프라인 모임                       |
|  10  | 2023.09.22 금 |  박진  | 7장. 단어 임베딩                          |
|  11  | 2023.10.06 금 |  박진  | 7장. 단어 임베딩                          |
|  12  | 2023.10.20 금 | 김계현 | LLM 동향                                  |
|      |               | 임창현 | GenAI 최신 내용                           |
|  16  | 2023.10.27 금 | 김계현 | LLM 규제 현황                             |
|      |               | 임창현 | 내부 문서 review                          |
|  17  | 2023.11.10 금 | 임창현 | 내부 문서 review                          |
|      | 2023.11.24 금 |        | 제3차 오프라인 모임 (Workshop)            |
|  18  | 2023.12.18 월 |  박진  | 8장. 순환신경망                           |
|  19  | 2024.01.04 목 | 김지훈 | 9장. 오토인코더                           |
|      |               |        | 10장. 비지도학습 (채종호, 미정)           |
|      |               | 임창현 | 11장. 강화학습                            |
|      |               |        | 12장. 텐서플로와 클라우드                 |
|      |               |        | 13장. 모바일, IoT, 텐서플로.js용 텐서플로 |
|      |               |        | 14장. AutoML 소개                         |
|      |               |        | 15장. 딥리닝의 수학적 배경                |
|      |               |        | 16장. TPU                                 |

  

## 참고 문헌

- [GitHub Desktop](https://desktop.github.com/)
- [샘알트만 | 인공지능(Chat GPT)은 구글을 대체할 수 있을까?](https://youtu.be/cgfFg5s_wXs)
- [일론 머스크가 새로운 인공지능 회사를 만드는 이유](https://youtu.be/M5MT7dRo1I4)
- [AI가 만드는 기회 (Nat Friedman - Former GitHub CEO)](https://youtu.be/z47Hx-acRdU)
- [인공지능 : 우리는 이대로 괜찮을까?](https://youtu.be/FuIsdCHPoDs)
- [2030년에는 컴퓨터와 사람의 뇌가 합쳐질 겁니다 | 레이 커즈와일](https://youtu.be/uc66zrI28UY)
- [테슬라 자율주행의 설계자 안드레 카파시](https://youtu.be/ay8E_moegfk)
- [OpenAI 샘 알트만 | 인공지능 chatGPT는 균형을 잡아갈 것](https://youtu.be/vZ8J36xrK3s)
- [교재 영문 ebook | PDF File](https://download.packt.com/free-ebook/9781838823412)
- [Datasets](https://github.com/tensorflow/datasets)
- CNN
  - [합성곱 신경망](https://www.tensorflow.org/tutorials/images/cnn?hl=ko)
- [Son's Notation ㅣ LECTURE NOTE, 머신러닝/딥러닝](https://sonsnotation.blogspot.com/)
- [교재 내 Source Code | Deep-Learning-with-TensorFlow-2-and-Keras](https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-2-and-keras)
  - [교재 내 이미지 | PDF](https://static.packt-cdn.com/downloads/9781838823412_ColorImages.pdf)
- [Models](https://www.tensorflow.org/api_docs/python/tf/keras/models)
- [Activation Functions](https://www.tensorflow.org/api_docs/python/tf/keras/activations)
- [Loss Functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses)
- [Optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
- [Metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics)

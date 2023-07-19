ㅁ 목적 : 텐서플로 1.x와 2.0의 차이점을 설명 
         1.x 프로그래밍 패러다임을 살펴본 후 2.x에서 사용할 수 있는 새로운 기능과 패러다임 설명

---------------------------------------------------------------------------------------------------------
1. 텐서플로 1.x의 이해
  1-1. 텐서플로 1.x 계산 그래프 구조
    다른 프로그램 언어와 다르게 만들고자 하는 신경망의 청사진을 만들어야 하며
    이를 위해, 프로그램을 계산 그래프(Computation graph)의 정의와 실행 두 부분으로 나눠야 함

    * 계산 그래프 : 노드(Node)와 에지(Edge)를 가진 네트워크
      - 노드 : 객체(텐서, tensors) 와 연산(operations)
      - 에지 : 연산 간에 흐르는 텐서
    * 그래프 실행 : 텐서와 연산 객체가 평가되는 환경을 캡슐화하는 세션 객체를 사용해 수행
    * 그래프를 사용하는 이유
      ① (심층)신경망을 설명해 줄 수 있는 자연스러운 비유
      ② 공통 하위 표현식을 제거하고, 커널을 합치고, 중복 표현식을 제거해서 자동으로 최적화
      ③ 훈련 중에 쉽게 배포할 수 있으며, CPU/GPU/TPU 같은 다양한 환경과
        클라우드, IoT, 모바일, 기존 서버 같은 다양한 환경에 배포 가능

  1-2. 상수, 변수, 플레이스홀더와 작업
    텐서플로는 간단히 말해 다양한 수학연산을 텐서로 정의하고 수행하는 라이브러리를 제공
    텐서는 기본적으로 n차원 배열임 : 스칼라-0차원, 벡터-1차원, 행쳘-2차원 텐서
    3가지 유형의 텐서를 제공 : 상수, 변수, 플레이스홀더(그래프에 값을 넣는데 사용, 보통 신경망 훈련시 새로운 훈련 예시 제공)

  1-3. 연산의 예시
    - 상수
    - 시퀀스
    - 랜덤 텐서
    - 변수 : 초기화, 저장, 플레이스홀더 정의

  1-4. 텐서플로 2.x에서의 1.x 예제
    1.x API는 신경망과 다른 많은 유형의 머신러닝 프로그램을 나타내는 계산 그래프를 생성/조작하는 유연한 방법을 제공
    2.x는 더 낮은 수준의 세부 정보를 추상화하는 높은 수준의 API를 제공

-----------------------------------------------------------------------------------------------------------
2. 텐서플로 2.x의 이해
  2.x는 tf.keras와 같은 하이레벨 API사용을 권장하지만, 내부의 세부정보를 더 많이 제어하는 1.x의 로우레벨API를 유지함
  2-1. 즉시 실행 : 여전히 그래프가 있지만 특별한 세션 인터페이스나 플레이스홀더 없이도 노들르 즉시 정의/변경/실행 가능
    - 1.x가 정적 계산 그래프를 정의한 반면, 2.x는 기본적으로 즉시 실행을 지원함(tf.keras API도 즉시 실행과 호환됨)
    - 파이선은 일반적으로 더 동적이며, 또 다른 딥러닝 패키지인 파이토치는 더 명령적이고 동적인 방식으로 사물을 정의함

  2-2. 오토그래프 : 즉시 실행 파이썬 코드를 가져와서 자동으로 그래프 생성 코드로 변환
    - 2.x에서는 기본적으로 if-while, print(), 기타 파이썬 기본 특징과 같은 제어 흐름을 포함해 명령형 파이썬 코드를
      지원하고 기본적으로 순수 텐서플로 그래프 코드로 변환할 수 있음
    - 사용법 : 파이썬 코드에 특정decorator tf.function을 annotation처럼 추가 (일반적으로 10배 가까이 신간 단축)
    ※파이썬 코드 자체는 일반적으로 빠르고 자동으로 최적화할 수 있는 그래프 형식으로 변환하는데 어려움

  2-3. 케라스 API : 3가지 프로그래밍 모델
    tf.keras는 다음 3가지 프로그래밍 모델과 함께 더 하이레벨 API를 제공 (특정 요구사항에 따라 세가지르 섞어 사용)
    ① 순차적 API : 90%의 사례에 적합한 매우 우아하고 직관적이며 간결한 모델
    ② 함수적 API : 다중 입력, 다중 출력, 비순차 흐름과의 잔존 연력, 공유, 재사용 가능 계층을 포함해 
                   좀 더 복잡한 (비선형)위상(topology)으로 모델을 구축하려는 경우 유용
    ③ 모델 서브클레싱 : 최고의 유연성을 제공하며 일반적으로 자신의 계층을 정의해야 할 때 사용

  2-4. 콜백(Callbacks) : tf.keras로 훈련할 때 유용한 특징으로, 훈련 중에 동작을 확장하거나 수정하고자 모델로 전달하는 객체임
    - tf.keras.callbacks.ModelCheckpoint : 정기적으로 모델의 체크포인트를 저장하고 문제가 발생할 때 복구하는데 사용
    - tf.keras.callbacks.LearningRateScheduler : 최적화하는 동안 학습률을 동적으로 변경할때 사용
    - tf.keras.callbacks.EarlyStopping : 검증 성능이 한동안 개선되지 않을 경우 훈련을 중단할 때 사용
    - tf.keras.callbacks.TensorBoard : 텐서보드를 사용해 모델의 행동을 모니터링할 때 사용

  2-5. 모델과 가중치 저장 : 모델을 훈련한 후에는 가중치를 지속적으로 저장해두면 유용함

  2-6. tf.data.datasets로 훈련
    2.x 또 다른 이점은 오디오/이미지/비디오/텍스트/번역 과 같은 다양한 범주의 이기종(대형) 데이터셋을 처리하는
    기본 메커니즘인 텐서플로 데이터셋의 도입 (pip를 사용해 설치)
    - 데이터셋은 원칙적인 방식으로 입력 데이터를 처리하는 데 필요한 라이브러리
    - 연산 종류 : 생성, 번환, 반복자

  2-7. tf.keras와 추정기
    직접 그래프 계산이나 tf.keras 상위 레벨 API외에도 텐서플로에는 추정기(Estimators)라는 추가 상위레벨 API 집합이 있음
    - 추정기 : 간단히 비유하자면 블록을 만들거나 이미 만들어진 블록을 사용하는 또 다른 방법
              단일 시스템 또는 분산 다중 서버에서 훈련시킬수 있는 대규모 양산(production-ready) 환경에 적합한 고효율
              학습 모델이며, 모델을 다시 코딩하지 않아도 CPU/GPU/TPU에서 실행 가능
    - 2.x를 사용할 때 추정기를 이미 채택한 경우에는 계속 사용하되 처음 시작하는 경우에는 tf.keras사용을 권장함

  2-8. 비정현 텐서 지원
    균일하지 않은 형태를 가진 특수 유형의 고밀도 텐서
    이는 텍스트 문장 및 계층적 데이터처럼 배치마다 차원이 변경될 수 있는 시퀀스나 기타 데이터 문제를 처리하는데 특히 유용

  2-9. 맞춤형 훈련
    텐서플로는 그래디언트를 계산할 수 있으며(자동 미분) 머신러닝 모델을 개발하기 쉽다.
    tf.keras를 사용하는 경우 fit()을 사용해 모델을 훈련시키고 그래디언트가 내부적으로 어떻게 계산되는지는 자세히 몰라도 됨
    그러나, 최적화를 좀 더 세밀하게 제어하려는 경우에는 맞춤형 훈련이 유용함

  2-10. 텐서플로 2.x에서 분산 훈련
    2.x에서는 분산GPU, 다중 머신, TPU를 사용하는 모델을 거의 추가 코드 없이 간단한 방법으로 훈련이 가능
    - tf.distribute.Strategy API (tf.keras 및 tf.estimator API와 즉시 실행을 모두 지원)

    * 다중 GPU
    하나의 기계에서 여러GPU에 대한 동기식 분산훈련을 하려면
    ① GPU에 분산되는 방식으로 데이터를 로드
    ② GPU에도 일부 연산을 분산
    ※GPU를 쉽게 사용 가능하고, 단일 서버에 사용되는 tf.keras코드를 조금만 변경하면 됨
    - MutiWorkerMirroredStrategy
    - TPUStrategy
    - ParameterServerStrategy

  2-11. 네임스페이스의 변화
    1.x에서 루트 네임스페이스의 밀도가 높아 검색이 어려웠던 부분을 개선 (루트 네임스페이스 : 1.x(500개 이상), 2.x(171개))

  2-12. 1.x에서 2.x로 변환
    1.x스크립트는 2.x에서 직접 작동하지 않으므로 변환 필요

  2-13. 텐서플로 2.x의 효율적인 사용
    ① 기본적으로 tf.keras같은 하이레벨API를 사용
    ② tf.function 데모레이터를 추가해 오토그래프로 그래프 모드에서 효율적으로 수행
    ③ 
  
















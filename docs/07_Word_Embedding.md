# 7. 단어 임베딩
source : https://github.com/AcornPublishing/tensorflow2-keras-deeplearning/tree/main/Chapter%207
image : http://www.acornpub.co.kr/acorn_guest/9781838823412_ColorImages.pdf


## 단어 임베딩: 시작과 기초
어휘의 단어나 구(phrase - 단어의 연속)가 실수의 백터로 매핑되는 자연어처리(NLP - Natural Lanaguage Processing)에서의 언어 모델링과 특정 학습 기술의 집합

a) 백터 : 벡터는 크기와 방향을 모두 가진 양  

b) 백터화 : 텍스트를 숫자로 변환, 초기에는 one-hot encoding 사용
<details>
  <summary>원핫 인코딩의 한계</summary>
  각 단어를 다른 단어와 완전히 독립적으로 취급, 벡터들 사이에는 어떤 의미적인 관계도 표현되지 않는다는 것을 의미  
  예를들어 아이유와 윤하는 가수/여자/솔로 등의 공통점이 있지만 이러한 관계를 표현하지 못함
</details>  

<br>

c) 벡터화에 정보 검색(IR Information Retrieval) 기술 도입   
- 용어빈도-역문서 빈도, 잠재 문맥 분석, 주제 모델링 등  
- 단어간의 의미적 유사성에 대한 문서 중심적 아이디어를 포착하기 위해 시도    

<br>

d) 분산 가설에 기반 : 유사한 맥락에서 등장하는 단어들은 비슷한 의미를 가짐

<br><br>

## 분산 표현
한 단어의 의미를 문맥상에서 다른 단어와의 관계를 고려해 퍼착하려 시도  
주변 단어를 보면 그 단어가 무었인지 알게 될 것이다 by J. R. Firth  
분산 임베딩 공간이란 유사한 문맥에서 사용된 단어들은 서로 가깝게 위치하는 공간을 의미  

```
왕은 남자다.
여왕은 여자다.
```


![분산 임베딩 공간 예시](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Ft1.daumcdn.net%2Fcfile%2Ftistory%2F99580F495C870DB10B)  
> 이미지 출처 : https://eda-ai-lab.tistory.com/118  
> 이미지 출처 링크의 원본 링크가 404여서 인용한 사이트를 출처로 표기

<br><br>

## 정적 임베딩
가장 오래된 유형의 단어 임베딩  
단어를 키로 사용하고 해당 벡터를 값으로 사용한 사전으로 비유 가능  
단어의 사용 방법에 상관없이 동일한 벡터값을 사용하기 때문에 중의어 문제에 대해 대응 불가  

<br>

### Word2Vec
2013년 토마스 미코로프가 이끄는 구글 연구팀에 의해 처음 개발, 두가지 아키텍처(CBOW, Skip-Gram)을 가짐
![7-2](https://user-images.githubusercontent.com/47945637/269584398-2df879b4-3c33-430a-bb48-ced28f696527.PNG)

<br>

```
제시 문장 : 나는 한국의 제주도를 여행하며 맛있는 음식들을 먹었다   
```
> 예시로 형태소분석은 고려하지 않고 뛰어쓰기를 기준으로 단어로 취급

#### a) CBOW : 연속 단어 주머니(CBOW Continuous Bag Of Words)
주변 단어의 창이 주어지면 모델은 목표 단어를 예측, 순서는 영향 X, 비교적 빠름  
하단의 문장을 창크기를 5, 즉 콘텐츠 단어의 왼쪽과 오른쪽에 각각 두개의 문맥(context) 단어를 가정하면 문맥창은 아래와 같이 표시

<details>
  <summary>CBOW를 통해 목표단어 예측</summary>

- **나는** 한국의 제주도를   
- 나는 **한국의** 제주도를 여행하며   
- 나는 한국의 **제주도를** 여행하며 맛있는  
- 한국의 제주도를 **여행하며** 맛있는 음식들을   
- 제주도를 여행하며 **맛있는** 음식들을 먹었다  
- 여행하며 맛있는 **음식들을** 먹었다  
- 맛있는 음식들을 **먹었다**

단어집합이 주어지면 목표 단어를 예측(단어집합에 대한 희소 백터가 입력)    
목표 단어를 가장 잘 나타내는 밀집 백터를 예측하는 것을 목표(즉 모델은 최댓값이 중심단어에 해당하는 밀집 백터를 예측하는 것을 학습)
- [한국의] [제주도를] [맛있는] [음식들을]  >> 여행하며
- [여행하며] [맛있는] [먹었다] >> 음식들을

</details>  

<br>

#### b) 스킵그램(Skip-Gram) 
목표 단어가 주어지면 모델이 주변단어를 예측, 빈도가 낮은 단어 예측에 유리  

<details>
  <summary>네거티브 샘플링을 통한 스킵 그램</summary>

**입력**
- 제주도, 나는 > 1
- 제주도, 한국의 > 1
- 제주도, 여행하며 > 1
- 제주도, 맛있는 > 1

**네거티브 입력**
- 제주도, 미국 > 0
- 제주도, 염세적인 > 0 

**예측**
- 중심 단어 "제주도를"을 사용하여 "나는", "한국의", "여행하며", "맛있는"을 예측

</details>  

<br>

#### c) 임베딩 
훈련의 부수효과인 학습된 가중치이며 다양한 자연어 처리 작업에 활용
- CBOW (Continuous Bag of Words)와 스킵그램(Skip-gram) 모델은 단어의 의미를 벡터 형태로 표현하기 위해 훈련
- 훈련 과정에서 모델의 내부 가중치는 계속 업데이트되며, 최종적으로 이 가중치들은 각 단어의 의미를 잘 표현하는 벡터로 사용
- 훈련을 통해 얻어진 가중치 벡터가 임베딩

<br>

### GloVe (Global Vectors for word representation)
단어 표현 전역 백터는 단어의 백터 표현을 얻는 데 필요한 비지도학습 알고리즘  
전체 데이터셋의 통계 정보를 미리 계산한 후에 이를 바탕으로 단어 임베딩을 학습  
- 훈련 : 말뭉치에 있는 단어-단어 공통 발생 통계량의 전체 집계에 대해 수행
- 결과 표현 : 단어 벡터 공간의 선형 하부 구조를 보여줌

> Word2Vec 와 차이점 : Word2Vec은 예측모델이지만, GloVe는 카운트 기반 모뎅

**특징**
- 통계 구성 : 훈련 말뭉치에서 동시에 등장하는 (단어, 문맥) 쌍의 대형 행렬을 구성, 행렬의 각 요소는 문맥에서 단어가 얼마나 자주 발생하는지 표현  
- 행렬 분해 : 한쌍의 (단어, 특징) 및 (특징, 문맥) 행렬로 분해, 확률적 그래디언트 하강(SGD : Stochastic Gradient Descent)를 사용해 수행
- 의미 관계 캡처 : 단어 벡터 간의 덧셈과 뺄셈을 통해 단어 간의 의미 관계를 캡처, ex) "왕" - "남자" + "여자" = "여왕"과 같은 관계를 표현 가능
- 병렬 모드 사용 : Word2Vec 보다 많은 리소스를 사용하며 병렬모드로 사용

<details>
  <summary>확률적 그래디언트 하강</summary>

  최적화 알고리즘 중 하나로, 대규모 데이터셋에서 효율적으로 모델을 학습시키기 위해 사용됩
  기본적인 그래디언트 하강법(Gradient Descent)은 모든 데이터 포인트에 대한 그래디언트(경사)를 계산하여 모델의 파라미터를 업데이트

  확률적 그래디언트 하강의 주요 특징:
  - 랜덤 샘플링: SGD는 한 번의 업데이트를 위해 데이터셋에서 랜덤하게 하나 또는 작은 배치의 데이터를 선택
  - 빠른 업데이트: 작은 양의 데이터만 사용하기 때문에, 파라미터 업데이트가 빠르고, 대규모 데이터셋에서도 효율적으로 작동
  - 변동성: 랜덤하게 데이터를 선택하기 때문에, 그래디언트와 파라미터 업데이트가 불안정할 수 있습니다. 이는 학습률(learning rate) 조정 등으로 완화
  - 근사적 최적화: SGD는 정확한 최적화보다는 근사적인 최적화를 목표로 하며, 실제로 많은 문제에서 충분히 좋은 성능을 다짐
  - 온라인 학습 가능: 데이터가 순차적으로 들어오는 경우에도 모델을 지속적으로 업데이트할 수 있어, 온라인 학습에 적합

  확률적 그래디언트 하강법은 다양한 머신 러닝 알고리즘과 딥 러닝 모델에서 널리 사용되며, 특히 대규모 데이터셋에서 효율적인 학습을 가능하게 함

</details>  

<br><br>

## gensim을 사용해 자신만의 임베딩 생성
gensim은 텍스트 문서에서 의미적 뜻을 추출하게 설계된 오픈소스 파이썬 라이브러리

text8이라는 작은 말뭉치를 사용해 임베딩 만들기
- text8은 영어 위키백과의 첫 10^9 바이트
- text8 모음을 다운로드해서 Word2Vec 모델을 생성한 후 저장
- 창크기가 5인 CBOW 모델 훈련
- 소스코드 : Chapter 7/create_embedding_with_text8.py
<details>
  <summary>확률적 그래디언트 하강</summary>

  ```python
  import gensim.downloader as api       # 텍스트 데이터셋을 다운로드하는 데 사용
  from gensim.models import Word2Vec    # Word2Vec 모델을 훈련시키는 데 사용됩니다.

  info = api.info("text8")              # text8 데이터셋 가져오기
  assert(len(info) > 0)                 # 변수의 길이가 0보다 큰지 확인, 데이터 정보가 정상적으로 로드 되었는지 확인

  dataset = api.load("text8")           # text8 데이터셋을 로드하여 dataset 변수에 저장
  model = Word2Vec(dataset)             # 로드된 dataset을 사용하여 Word2Vec 모델을 훈련

  model.save("data/text8-word2vec.bin") # 훈련된 Word2Vec 모델 저장

```
> 코드는 5~10분동안 실행된 후 훈련된 모델을 data 폴더에 기록

</details>  

<br><br>

## gensim을 사용한 임베딩 공간 탐색
**실습해 보기**
<details>
  <summary>실습 소스</summary>

```python
  from gensim.models import KeyedVectors                  # 단어와 그에 해당하는 벡터를 쉽게 관리

  def print_most_similar(word_conf_pairs, k):             # word_conf_pairs는 단어와 유사도 점수를 가진 튜플의 리스트이며, k는 출력할 단어의 수
      for i, (word, conf) in enumerate(word_conf_pairs):  # word_conf_pairs 리스트를 반복하면서 각 단어(word)와 그에 대한 유사도 점수(conf)를 가져옴, enumerate 함수는 현재 인덱스도 함께 반환
          print("{:.3f} {:s}".format(conf, word))         # 유사도 점수(conf)와 단어(word)를 출력, 유사도 점수는 소수점 아래 3자리까지 표시
          if i >= k-1:
              break
      if k < len(word_conf_pairs):
          print("...")


  ## 단어 목록 가져오기 ##
  model = KeyedVectors.load("data/text8-word2vec.bin")    # data/text8-word2vec.bin 파일에서 저장된 Word2Vec 모델을 로드하여 model 변수에 저장
  word_vectors = model.wv                                 # 로드된 모델에서 단어 벡터(wv)를 가져와 word_vectors 변수에 저장

  # words = word_vectors.vocab.keys()
  #   > 이 부분은 이 에러는 Gensim 라이브러리의 4.0.0 버전에서 vocab 속성이 KeyedVector 클래스에서 제거되었기 때문에 에러 발생
  #   > 이전에는 vocab 속성을 사용하여 단어 목록에 접근할 수 있었지만, Gensim 4.0.0 이후로는 이 방법이 더 이상 지원되지 않음
  words = word_vectors.index_to_key                       # 단어 목록을 index_to_key 속성을 사용하여 가져와 words 변수에 저장
  print([x for i, x in enumerate(words) if i < 10])       # 단어 목록(words)에서 처음 10개의 단어를 출력
  assert("king" in words)                                 # 단어 목록(words)에 "king"이 포함되어 있는지 확인합니다. 만약 "king"이 없다면, assert 문은 에러를 발생


  ## king과 유사한 단어 검색 ##
  print()
  print("# words similar to king")

  # word_vectors 객체의 most_similar 메서드를 사용하여 "king"이라는 단어와 가장 유사한 단어들과 그 유사도를 계산
  # 이 메서드는 단어와 그에 해당하는 유사도 점수를 가진 튜플의 리스트를 반환
  print_most_similar(word_vectors.most_similar("king"), 5)


  ##  france와 paris의 관계를 통해, "berlin"은 어떤 국가에 대한 수도인지를 예측 ##
  print()
  print("# vector arithmetic with words (cosine similarity)") # cosine 유사도 사용
  print("# france + berlin - paris = ?")
  print_most_similar(word_vectors.most_similar(               # 연산 결과에 가장 가까운 단어 벡터 찾는 함수
      positive=["france", "berlin"], negative=["paris"]), 1   # Vector("france")+Vector("berlin")−Vector("paris")
  )

  print("# vector arithmetic with words (Levy and Goldberg)") # Levy and Goldberg이 제안한 방식
  print("# france + berlin - paris = ?")
  print_most_similar(word_vectors.most_similar_cosmul(        
      positive=["france", "berlin"], negative=["paris"]), 1   # cosine 유사도를 곱셈으로 계산
  )


  ## 단어의 목록 중 이상한 항목을 탐지 ##
  print()
  print("# find odd one out")
  print("# [hindus, parsis, singapore, christians]")
  print(word_vectors.doesnt_match(["hindus", "parsis",        # 종교를 나열한 목록 중 singapore는 국가명
      "singapore", "christians"]))


  ## 두 단어 사이의 유사성 계산 ##
  print()
  print("# similarity between words")
  for word in ["woman", "dog", "whale", "tree"]:              # 비교할 단어 제시 단어 제시
      print("similarity({:s}, {:s}) = {:.3f}".format(         
          "man", word,
          word_vectors.similarity("man", word)                # man과의 유사도 측정
      ))

  print("# similar by word")                                  
  print(print_most_similar(   
      word_vectors.similar_by_word("singapore"), 5)           # singapore와 가장 유사한 5개의 단어와 그들의 코사인 유사도를 튜플 형태로 반환
  )


  ## 두 단어 사이의 거리 계산 ##
  print()
  print("# distance between vectors")
  print("distance(singapore, malaysia) = {:.3f}".format(
      word_vectors.distance("singapore", "malaysia")          # 임베딩 공간에서 두 단어사이의 거리, 1 - similarity()
  ))


  # ## 어휘단어 벡터 ##
  vec_song = word_vectors["song"]
  print("\n# output vector obtained directly, shape:", vec_song.shape)

  # vec_song_2 = word_vectors.word_vec("song", use_norm=True)
  # print("# output vector obtained using word_vec, shape:", vec_song_2.shape)
  #  > 이 부분은 word_vec 메서드는 더 이상 권장되지 않으며, 대신 get_vector 메서드를 사용해야 한다고 경고
  #  > get_vector() 메서드는 use_norm이라는 예상치 못한 키워드 인자를 받았다고 에러 발생, get_vector 메서드가 use_norm 인자를 지원하지 않음
```

</details>  

<br><br>

## 워드 임베딩을 사용한 스팸 탐지
대규모 말뭉치에서 생성된 다양하고 강력한 임베딩을 광범위하게 사용 가능함에 따라 머신러닝 모델에 사용할 텍스트 입력으로 사용
텍스트는 일련의 토큰으로 취급되며, 입베딩은 각 토큰에 대해 고정차원의 밀집 벡터를 제공
각 토큰은 벡터로 대체되고 텍스트 시퀀스를 예제행렬로 변환, 각 행렬은 임베딩의 차원에 해당하는 고정된 개수의 특징을 가짐
<details>
  <summary>GPT(Generative Pre-trained Transformer) 모델과의 차이</summary>

  GPT는 Transformer 모델(MLP Multi-Layer Perceptron)를 기반으로 하며, 이는 워드 임베딩과 RNN 또는 LSTM을 사용하는 이전 세대의 모델과는 다른 접근 방식  
  GPT는 텍스트를 토큰화한 후, 각 토큰을 고차원의 벡터로 동적 변환하며 문맥에 따라 달라짐, 즉 같은 단어라도 문맥에 따라 다른 벡터로 표현  
  > gpt2 1600차원, GPT-3 12888 차원

  이러한 특성은 워드 임베딩 방식에서는 볼 수 없는 것이며, 이로 인해 GPT는 더욱 복잡한 언어 구조와 의미를 파악  
  문맥에 따라 의미가 달라지는 단어나 구, 높은 수준의 언어 추론과 같은 복잡한 자연어 처리 작업에서 뛰어난 성능을 보입니다.

</details>  

1차원 컨볼루션 신경망 버전으로 실습 - SMS 혹은 문자 메세지를 스팸으로 분류하는 스팸 탐지기

<details>
  <summary>실습 소스</summary>

  ```python
  import argparse
  import gensim.downloader as api
  import numpy as np
  import os
  import shutil
  import tensorflow as tf                                         # pip3 install tensorflow 

  from sklearn.metrics import accuracy_score, confusion_matrix    # pip3 install scikit-learn


  def download_and_read(url):
      local_file = url.split('/')[-1]                             # URL의 마지막 부분을 추출하여 로컬 파일 이름으로 사용
      p = tf.keras.utils.get_file(local_file, url,                # TensorFlow의 유틸리티 함수를 사용하여 파일을 다운로드하고 압축 풀기
          extract=True, cache_dir=".")
      labels, texts = [], []                                      # 두개의 빈 리스트 생성
      local_file = os.path.join("datasets", "SMSSpamCollection")  # datasets 디렉토리의 SMSSpamCollection 파일 지정
      with open(local_file, "r") as fin:                          # 파일 오픈
          for line in fin:                                        # 라인단위로 분리
              label, text = line.strip().split('\t')              # 각 라인을 탭으로 분리하여 레이블과 텍스트 획득
              labels.append(1 if label == "spam" else 0)          # 레이블이 "spam"이면 1을, 그렇지 않으면 0을 labels 리스트에 추가
              texts.append(text)
      return texts, labels                                        # texts는 SMS 메시지의 내용을, labels는 해당 메시지가 스팸인지 아닌지를 나타내는 레이블을 포함

  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


  def build_embedding_matrix(sequences, word2idx, embedding_dim, 
          embedding_file):
      if os.path.exists(embedding_file):
          E = np.load(embedding_file)
      else:
          vocab_size = len(word2idx)
          E = np.zeros((vocab_size, embedding_dim))
          word_vectors = api.load(EMBEDDING_MODEL)
          for word, idx in word2idx.items():
              try:
                  E[idx] = word_vectors.word_vec(word)
              except KeyError:   # word not in embedding
                  pass
              # except IndexError: # UNKs are mapped to seq over VOCAB_SIZE as well as 1
              #     pass
          np.save(embedding_file, E)
      return E



  ## 토큰화 하고 텍스트 채우기 ##
  #  주어진 텍스트 데이터를 처리하여 딥 러닝 모델에 입력할 수 있는 형태로 변환하는 과정
  print()
  # UCI 머신러닝 저장소의 SMS 스팸수집 데이터셋, 5,574개의 sms 레코드가 있고 747개가 스팸
  DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
  texts, labels = download_and_read(DATASET_URL)

  tokenizer = tf.keras.preprocessing.text.Tokenizer()                             # 텍스트 토크나이저 객체를 생성
  tokenizer.fit_on_texts(texts)                                                   # 어휘 생성, 각 단어는 고유한 정수 ID에 매핑
  text_sequences = tokenizer.texts_to_sequences(texts)                            # sms 메세지를 정수 시퀀스(리스트 형식)로 변환, fit_on_texts 결과 기반
  text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences)  # 모든 정수 시퀀스를 같은 길이로 패딩(모자란 길이는 0 추가)
  num_records = len(text_sequences)                                               # 전체 레코드(문장)의 수를 num_records에 저장
  max_seqlen = len(text_sequences[0])                                             # 패딩된 시퀀스의 최대 길이를 max_seqlen에 저장
  print("{:d} sentences, max length: {:d}".format(num_records, max_seqlen))       # 전체 레코드(문장) 수와 최대 시퀀스 길이를 출력

  # labels
  # 레이블을 원-핫 인코딩으로 변환
  NUM_CLASSES = 2
  cat_labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)     # 각 레이블을 해당 클래스의 인덱스만 1이고 나머지는 0인 배열로 변환

  # vocabulary
  # 어휘에 대한 여러 유용한 딕셔너리를 생성하는 작업을 수행
  word2idx = tokenizer.word_index                           # 토크나이저에서 어휘의 단어를 해당 인덱스에 매핑하는 딕셔너리를 가져와 word2idx에 저장
  idx2word = {v:k for k, v in word2idx.items()}             # word2idx 딕셔너리의 키와 값을 뒤집어, 인덱스를 단어에 매핑하는 새로운 딕셔너리 idx2word를 생성
  word2idx["PAD"] = 0                                       # 패딩을 위한 특별한 토큰 "PAD"를 딕셔너리 word2idx에 추가하고, 그 인덱스를 0으로 설정
  idx2word[0] = "PAD"                                       # 딕셔너리 idx2word에 인덱스 0을 "PAD" 토큰에 매핑
  vocab_size = len(word2idx)                                # 어휘 크기를 word2idx 딕셔너리의 길이로 설정
  print("vocab size: {:d}".format(vocab_size))              # 어휘 크기(고유한 단어의 집합)를 출력



  ## 임베딩 행렬 구축 ##
  # gensim 툴킷은 다양한 훈련된 임베딩 모델 사용 : Word2Vec, GloVe, fastText, ConceptNet Numberbatch등
  print()

  # embedding
  EMBEDDING_DIM = 300                                                 # 임베딩 차원을 설정
  DATA_DIR = "data"
  EMBEDDING_NUMPY_FILE = os.path.join(DATA_DIR, "E.npy")              # 임베딩 행렬이 저장될 파일의 경로를 설정, "data" 디렉토리 안에 "E.npy"라는 이름으로 저장
  EMBEDDING_MODEL = "glove-wiki-gigaword-300"                         # GloVe(Gigawords 말뭉치를 기반으로 하는 300차원 벡터 사용) 모델 사용

  # build_embedding_matrix 함수를 호출하여 임베딩 행렬을 생성
  # 이 함수는 텍스트 시퀀스, 단어-인덱스 매핑, 임베딩 차원, 그리고 저장할 파일의 경로를 인자로 받음
  E = build_embedding_matrix(text_sequences, word2idx, EMBEDDING_DIM,
      EMBEDDING_NUMPY_FILE)
  print("Embedding matrix:", E.shape)                                 # 어휘 크기, 생성된 임베딩 행렬의 차원을 출력



  ## 스팸 분류기 정의 ##
  # 1차원 컨볼루션 신경망 사용
  print()

  # argparse 모듈을 사용하여 커맨드 라인 인자를 파싱
  parser = argparse.ArgumentParser()                # ArgumentParser 객체를 생성합니다. 이 객체는 커맨드 라인 인자를 파싱하는 데 사용
  parser.add_argument("--mode", help="run mode",    # --mode라는 이름의 커맨드 라인 인자를 추가하라는 지시, help 인자는 이 옵션에 대한 설명을 제공
      choices=[
          "scratch",
          "vectorizer",
          "finetuning"
      ])
  args = parser.parse_args()                        # 실제로 커맨드 라인 인자를 파싱하고, 결과를 args 객체에 저장
  run_mode = args.mode                              # args 객체에서 mode 인자의 값을 추출하여 run_mode 변수에 저장

  # 스팸 분류를 위한 합성곱 신경망 모델을 정의하는 클래스
  class SpamClassifierModel(tf.keras.Model):                          # tf.keras.Model을 상속받아 Keras 모델로 사용
      def __init__(self, vocab_sz, embed_sz, input_length,            # 클래스의 생성자를 정의
              num_filters, kernel_sz, output_sz, 
              run_mode, embedding_weights, 
              **kwargs):
          super(SpamClassifierModel, self).__init__(**kwargs)         # 상위 클래스의 생성자를 호출
          if run_mode == "scratch":                                   # "scratch" 모드, 임베딩을 처음부터 학습, 가중치를 학습
              self.embedding = tf.keras.layers.Embedding(vocab_sz, 
                  embed_sz,
                  input_length=input_length,
                  trainable=True)                                     
          elif run_mode == "vectorizer":                              # "vectorizer" 모드, 주어진 임베딩 가중치를 사용, 가중치를 학습하지 않음
              self.embedding = tf.keras.layers.Embedding(vocab_sz, 
                  embed_sz,
                  input_length=input_length,
                  weights=[embedding_weights],
                  trainable=False)                                    
          else:                                                       # finetuning 모드를 기본으로 설정, 주어진 임베딩 가중치를 사용
              self.embedding = tf.keras.layers.Embedding(vocab_sz,    # 주어진 임베딩 가중치를 사용, 가중치를 학습
                  embed_sz,
                  input_length=input_length,
                  weights=[embedding_weights],
                  trainable=True)                                     
          self.dropout = tf.keras.layers.SpatialDropout1D(0.2)        # 과적합을 방지하기 위한 Spatial Dropout 레이어를 추가
          self.conv = tf.keras.layers.Conv1D(filters=num_filters,     # 1D 합성곱 레이어를 추가합니다. 이 레이어는 텍스트 데이터의 지역적 패턴을 학습
              kernel_size=kernel_sz,
              activation="relu")
          self.pool = tf.keras.layers.GlobalMaxPooling1D()            # 최대 풀링 레이어를 추가하여 합성곱 레이어의 출력을 다운샘플링
          self.dense = tf.keras.layers.Dense(output_sz,               # 완전 연결 레이어를 추가하여 분류를 수행
              activation="softmax"
          )

      def call(self, x):                # 모델의 순전파를 정의하는 메서드
          # 입력 x를 임베딩, 드롭아웃, 합성곱, 풀링, 완전 연결 레이어를 통해 전달
          x = self.embedding(x)
          x = self.dropout(x)
          x = self.conv(x)
          x = self.pool(x)
          x = self.dense(x)
          return x

  # model definition
  conv_num_filters = 256                                # 1D 합성곱 레이어에 사용될 필터(또는 커널)의 수를 256으로 설정, 합성곱 레이어가 학습할 특징의 수
  conv_kernel_size = 3                                  # 1D 합성곱 레이어의 커널 크기를 3으로 설정, 이는 각 커널이 한 번에 고려할 연속된 단어의 수
  model = SpamClassifierModel(                          # 클래스의 인스턴스를 생성, 이 클래스는 위에서 정의한 스팸 분류를 위한 합성곱 신경망 모델
      vocab_size,                                       # 어휘 크기, 고유한 단어의 수
      EMBEDDING_DIM,                                    # 임베딩 벡터의 차원
      max_seqlen,                                       # 입력 시퀀스의 최대 길이
      conv_num_filters,                                 # 합성곱 레이어의 필터 수, 256
      conv_kernel_size,                                 # 합성곱 레이어의 커널 크기, 3
      NUM_CLASSES,                                      # 출력 클래스의 수, 스팸 또는 스팸이 아닌 경우의 2개의 클래스
      run_mode,                                         # 모델의 실행 모드
      E)  
  model.build(input_shape=(None, max_seqlen))
  model.summary()                                       #  Keras 모델의 구조를 보기 좋게 출력해주는 메서드로, 각 레이어의 이름, 타입, 출력 형태, 그리고 해당 레이어의 파라미터 수를 포함한 표를 출력

  # model.summary() 결과 예시 : vectorizer mode
  # _________________________________________________________________
  #  Layer (type)                Output Shape              Param #   
  # =================================================================
  #  embedding (Embedding)       multiple                  2703000        # 단어를 고차원의 벡터로 변환, 2703000은 이 레이어에 있는 파라미터의 수
                                                                  
  #  spatial_dropout1d (Spatial  multiple                  0              # 공간 드롭아웃 레이어, 이 레이어는 일부 입력을 무작위로 0으로 설정하여 과적합을 방지
  #  Dropout1D)                                                           # 0은 이 레이어에 추가적인 학습 가능한 파라미터가 없음
                                                                  
  #  conv1d (Conv1D)             multiple                  230656         # 1D 합성곱 레이어, 이 레이어는 입력 시퀀스의 지역 패턴을 학습
                                                                  
  #  global_max_pooling1d (Glob  multiple                  0              # 글로벌 최대 풀링 레이어, 이 레이어는 전체 시퀀스에서 가장 중요한 특징을 선택
  #  alMaxPooling1D)                                                 
                                                                  
  #  dense (Dense)               multiple                  514            # 밀집 연결 레이어, 이 레이어는 입력을 출력으로 변환하는 데 사용
                                                                  
  # =================================================================
  # Total params: 2934170 (11.19 MB)                                      # 모델의 총 파라미터 수
  # Trainable params: 231170 (903.01 KB)                                  # 학습 가능한 파라미터 수
  # Non-trainable params: 2703000 (10.31 MB)                              # 학습 불가능한 파라미터 수

  # compile and train
  model.compile(optimizer="adam", loss="categorical_crossentropy",
      metrics=["accuracy"])



  ## 모델의 훈련과 평가 ##
  print()

  # dataset
  BATCH_SIZE = 128

  dataset = tf.data.Dataset.from_tensor_slices((text_sequences, cat_labels))      # text_sequences와 cat_labels로부터 데이터셋을 생성 
                                                                                  # text_sequences는 텍스트 메시지의 정수 시퀀스, cat_labels는 해당 메시지의 레이블(햄 또는 스팸)
  dataset = dataset.shuffle(10000)                                                # 데이터셋의 항목을 무작위로 섞음, 10000은 버퍼의 크기
  test_size = num_records // 4                                                    # 전체 데이터셋의 1/4 크기를 테스트 데이터셋의 크기로 설정
  val_size = (num_records - test_size) // 10                                      # 남은 데이터 중 1/10 크기를 검증 데이터셋의 크기로 설정
  test_dataset = dataset.take(test_size)                                          # 처음부터 test_size만큼의 데이터를 가져와 테스트 데이터셋으로 설정
  val_dataset = dataset.skip(test_size).take(val_size)                            # test_size만큼의 데이터를 건너뛴 후, val_size만큼의 데이터를 가져와 검증 데이터셋으로 설정
  train_dataset = dataset.skip(test_size + val_size)                              # test_size와 val_size만큼의 데이터를 합한 수만큼 건너뛴 나머지 데이터를 학습 데이터셋으로 설정

  test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)              # 테스트 데이터셋을 BATCH_SIZE 크기의 배치로 분할
                                                                                  # drop_remainder=True는 데이터셋의 마지막 배치가 BATCH_SIZE보다 작을 경우 해당 배치를 삭제하라는 의미, 모든 배치의 크기가 동일하게 유지
  val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)                # 검증 데이터셋을 BATCH_SIZE 크기의 배치로 분할
  train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)            # 학습 데이터셋을 BATCH_SIZE 크기의 배치로 분할

  # 모델 훈련
  NUM_EPOCHS = 3                        # 전체 학습 데이터셋을 몇 번 반복하여 학습할 것인지를 지정

  # 데이터 분포는 햄 4827개, 스팸 747개(총 5574개)
  # 대략 87%의 햄과 13%의 스팸이기 때문에 모든 메시지를 햄으로만 예측하도 87%의 정확도를 가짐
  # 스팸(1) 아이템은 ham(0) 메시지보다 8배 가중치를 부여, 모델이 스팸 메시지를 더 중요하게 여기게 되어, 스팸 분류 성능이 향상
  CLASS_WEIGHTS = { 0: 1, 1: 8 }

  model.fit(
      train_dataset,                    # 학습 데이터를 제공
      epochs=NUM_EPOCHS,                # 전체 학습 데이터셋 반복 횟수
      validation_data=val_dataset,      # 검증 데이터셋, 각 에포크가 끝날 때마다 이 검증 데이터셋을 사용하여 모델의 성능을 평가, 모델이 학습 데이터에 과적합되지 않았는지 확인
      class_weight=CLASS_WEIGHTS        # 각 클래스에 대한 가중치를 지정하는 딕셔너리
      )


  # 테스트 집합으로 평가
  labels, predictions = [], []
  for Xtest, Ytest in test_dataset:             # 테스트 데이터셋의 각 배치에 대해 반복
      Ytest_ = model.predict_on_batch(Xtest)    # 현재 배치의 입력 데이터 Xtest에 대한 모델의 예측값을 계산
      ytest = np.argmax(Ytest, axis=1)          # 실제 레이블 Ytest에서 가장 큰 값의 인덱스를 찾아 ytest에 저장, 원-핫 인코딩된 레이블을 정수 형태로 변환하는 과정
      ytest_ = np.argmax(Ytest_, axis=1)        # 모델의 예측값 Ytest_에서 가장 큰 값의 인덱스를 찾아 ytest_에 저장
      labels.extend(ytest.tolist())             # 실제 레이블의 리스트 labels에 ytest의 값을 추가
      predictions.extend(ytest.tolist())        # 예측값의 리스트 predictions에 ytest_의 값을 추가

  print("test accuracy: {:.3f}".format(accuracy_score(labels, predictions)))    # 레이블과 예측값을 사용하여 정확도를 계산하고 출력
  print("confusion matrix")
  print(confusion_matrix(labels, predictions))                                  # 레이블과 예측값을 사용하여 혼동 행렬을 계산하고 출력

  
  # 결과 해석
  # Epoch 1/3 - 모델이 데이터셋을 훈련
  # loss: 0.5506 - 훈련 데이터에 대한 손실
  # accuracy: 0.8489 - 훈련 데이터에 대한 정확도
  # val_loss: 0.1169 - 검증 데이터에 대한 손실
  # val_accuracy: 0.9583 - 검증 데이터에 대한 정확도
  # test accuracy: 1.000 - 테스트 데이터셋에 대한 모델의 정확도

  # confusion matrix
  # [[1108 0] - "ham" 메시지에 대한 예측 결과, 1108 개의 "ham" 메시지가 올바르게 "ham"으로 분류되었고, 0개의 "ham" 메시지가 잘못 "spam"으로 분류
  # [ 0 172]] - "spam" 메시지에 대한 예측 결과, 172개의 "spam" 메시지가 올바르게 "spam"으로 분류되었고, 0개의 "spam" 메시지가 잘못 "ham"으로 분류

```

</details> 

### 스팸 탐지기 실행
scratch, vectorizer, finetuning 세가지 모드로 검증
- 스크래치 : 임베딩을 처음부터 학습, 가중치를 학습 > 첫번째 정확도가 낮지만 두번째부터 가장 높은 정확도가 유지
- 백터화 : 주어진 임베딩 가중치를 사용, 가중치를 학습하지 않음 > 유리하게 시작, 가중치 학습이 안되어 3번째 에폭의 정확도가 가능 낮음
- 미세 조정 : 주어진 임베딩 가중치를 사용, 가중치를 학습 > 유리하게 시작, 첫 에폭부터 가장 높은 정확도
![7-3](https://user-images.githubusercontent.com/47945637/269808297-7027ad28-9e0c-4840-878f-138c287ddf10.PNG)

<br><br>

## 신경망 임베딩 : 단어 이외의 용도
단어 임베딩 기술은 다양한 방식으로 발전했으며, 방향 중 하나는 단어 임베딩을 단어가 아닌 설정에 적용하는 것   
비슷한 맥락에서 발생하는 개체들은 서로 밀접하게 관련되어 있는 경향이 있으며, 상황에 의존
- 개체들 : 사과, 배
- 경향 : 임베딩 알고리즘이 단어들 사이의 관계를 학습하면서, 비슷한 맥락에서 사용되는 단어들을 서로 가까이 배치
- 상황 : 임베딩은 단어가 주어진 문맥 내에서 어떻게 사용되는지에 따라 그 단어의 벡터 표현을 조정(중의어)

<br>

### Item2Vec
협업 필터링 사용례를 위해 제안 - 사용자와 유사한 구매 이력이 있는 다른 사람의 구매를 기반으로 상품 추천  
항목을 단어로 사용하고 아이템셋(시계열 순차 구매 항목)을 문장으로 사용

<br>

### node2vec
그래프에서 노드의 특징을 학습하는 기법으로 제안
그래프에서 다수의 고정 길이 랜덤 워크를 실행함으로서 그래프 구조의 임베딩을 학습
> 랜덤 워크 : 그래프의 노드에서 시작하여, 각 단계에서 이웃 노드 중 하나를 무작위로 선택하여 이동하는 과정  

노드는 단어이고 랜덤 워크는 문장 - 랜덤 워크를 통해 생성된 경로는 단어의 시퀀스와 유사하게 간주

<br>

**실습 - node2vec와 유사한 모델 생성**  
- 데이터셋은 11463 x 5812의 행렬로 단어개수 표현 - 행은 단어, 열은 컨퍼런스 논문  
- 논문 그래프를 만드는데, 두 논문사이의 에지(edge)는 두 논문 모두에 등장하는 단어  
- node2vec와, DeepWalk는 모두 무방향, 무가중치 적용  
- 에지는 두 논문 사이의 단어 동시 발생 횟수에 따라 가중치를 가질 수 있음  

<details>
  <summary>실습 소스</summary>

  ```python
  import gensim
  import logging
  import numpy as np
  import os
  import shutil
  import tensorflow as tf

  from scipy.sparse import csr_matrix
  # from scipy.stats import spearmanr
  from sklearn.metrics.pairwise import cosine_similarity

  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)     # 로깅 설정을 초기화

  DATA_DIR = "./data"
  UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00371/NIPS_1987-2015.csv"


  def download_and_read(url):
      local_file = url.split('/')[-1]
      p = tf.keras.utils.get_file(local_file, url, cache_dir=".")
      row_ids, col_ids, data = [], [], []
      rid = 0
      f = open(p, "r")
      for line in f:
          line = line.strip()
          if line.startswith("\"\","):
              # header
              continue
          if rid % 100 == 0:
              print("{:d} rows read".format(rid))
          # compute non-zero elements for current row
          counts = np.array([int(x) for x in line.split(',')[1:]])
          nz_col_ids = np.nonzero(counts)[0]
          nz_data = counts[nz_col_ids]
          nz_row_ids = np.repeat(rid, len(nz_col_ids))
          rid += 1
          # add data to big lists
          row_ids.extend(nz_row_ids.tolist())
          col_ids.extend(nz_col_ids.tolist())
          data.extend(nz_data.tolist())
      print("{:d} rows read, COMPLETE".format(rid))
      f.close()
      TD = csr_matrix((
          np.array(data), (
              np.array(row_ids), np.array(col_ids)
              )
          ),
          shape=(rid, counts.shape[0]))
      return TD


  ## 문서 간의 유사성을 나타내는 이진 에지 행렬을 생성 - 이 행렬은 두 문서가 얼마나 유사한지(즉, 공통 용어의 수)를 기반으로 문서 간의 관계 표현
  # 데이터를 읽고 용어-문서 행렬로 변환
  print("# read data and convert to Term-Document matrix")
  TD = download_and_read(UCI_DATA_URL)    # 용어-문서 행렬(Term-Document Matrix, TD) 반환 - 행은 특정 용어(단어), 각 열은 특정 문서

  # 무방향, 무가중치 에지 행렬 계산
  print("# compute undirected, unweighted edge matrix")
  E = TD.T * TD                           # 용어-문서 행렬의 전치행렬(TD.T)을 원래의 용어-문서 행렬(TD)와 곱하여 문서-문서 유사성 행렬(E)을 생성

  # 이진화
  print("# binarize")
  E[E > 0] = 1                            # 문서-문서 유사성 행렬의 각 요소는 두 문서간의 유사성을 나타내므로 0이 아닌 요소를 모두 1로 설정
  print(E.shape)


  ## 랜덤 워크 생성 프로세스
  NUM_WALKS_PER_VERTEX = 32
  MAX_PATH_LENGTH = 40
  RESTART_PROB = 0.15
  
  RANDOM_WALKS_FILE = os.path.join(DATA_DIR, "random-walks.txt")

  def construct_random_walks(E, n, alpha, l, ofile):                                # 주어진 에지 행렬(E)을 사용하여 랜덤 워크를 생성
      if os.path.exists(ofile):                                                     # 이미 생성된 랜덤 워크 확인
          print("random walks generated already, skipping")
          return
      f = open(ofile, "w")
      for i in range(E.shape[0]):                                                   # 각 꼭지점(vertex)마다 랜덤 워크 생성:
          if i % 100 == 0:                                                          # 100개의 꼭지점마다 진행 상황을 출력
              print("{:d} random walks generated from {:d} starting vertices"
                  .format(n * i, i))
          if i <= 3273:                                                             # 3273 이하의 꼭지점은 건너뜀
              continue
          for j in range(n):                                                        # n개의 랜덤 워크 생성
              curr = i
              walk = [curr]                                                         # 랜덤 워크 초기화
              target_nodes = np.nonzero(E[curr])[1]                                 # 꼭지점에서 연결된 타깃 노드를 선택
              for k in range(l):                                                    # 랜덤 워크를 최대 길이 l까지 확장
                  # should we restart?
                  if np.random.random() < alpha and len(walk) > 5:                  # 랜덤 워크가 5보다 크고 재시작 확률 alpha보다 작은 경우 랜덤 워크를 중단
                      break
                  try:                                                              # 외향 에지를 무작위로 선택하고 랜덤 워크를 확장
                      curr = np.random.choice(target_nodes)
                      walk.append(curr)
                      target_nodes = np.nonzero(E[curr])[1]
                  except ValueError:
                      continue
              f.write("{:s}\n".format(" ".join([str(x) for x in walk])))            # 랜덤 워크 저장

      print("{:d} random walks generated from {:d} starting vertices, COMPLETE"
          .format(n * i, i))
      f.close()

  # 랜덤 워크 구성(주의: 시간이 걸리는 프로세스!)
  # construct_random_walks(E, NUM_WALKS_PER_VERTEX, RESTART_PROB, 
  #     MAX_PATH_LENGTH, RANDOM_WALKS_FILE)


  ## 모델 생성
  print()
  W2V_MODEL_FILE = os.path.join(DATA_DIR, "w2v-neurips-papers.model")               # 모델 파일 경로 설정

  class Documents(object):                                                          # 주어진 입력 파일의 내용을 반복적으로 읽어오는 역할
      def __init__(self, input_file):
          self.input_file = input_file

      def __iter__(self):                                                           # 이 메서드는 입력 파일의 각 라인을 읽어와서 공백으로 분리된 토큰의 리스트로 반환
          with open(self.input_file, "r") as f:
              for i, line in enumerate(f):
                  if i % 1000 == 0:
                      if i % 1000 == 0:
                          logging.info("{:d} random walks extracted".format(i))     # 1000 라인마다 로깅 메시지를 출력
                  yield line.strip().split()


  # 모델 훈련
  def train_word2vec_model(random_walks_file, model_file):                          # 주어진 랜덤 워크 파일을 사용하여 Word2Vec 모델을 훈련
      if os.path.exists(model_file):                                                # 모델 파일 존재 확인
          print("Model file {:s} already present, skipping training"
              .format(model_file))
          return
      docs = Documents(random_walks_file)                                           # Documents 객체 생성
      model = gensim.models.Word2Vec(                                               # Word2Vec 모델을 초기화
          docs,
          # size=128,       # gensim 버전업으로 인자 변경
          vector_size=128,  # size of embedding vector
          window=10,        # window size
          sg=1,             # skip-gram model
          min_count=2,
          workers=4
      )
      model.train(                                                                  # 50 에포크 동안 모델을 훈련
          docs, 
          total_examples=model.corpus_count,
          epochs=50)
      model.save(model_file)                                                        # 훈련된 모델을 지정된 파일 경로에 저장

  print("# train model")
  train_word2vec_model(RANDOM_WALKS_FILE, W2V_MODEL_FILE)                   


  ## 모델을 활용해 문서사이의 유사점 찾기
  print()
  def evaluate_model(td_matrix, model_file, source_id):
      model = gensim.models.Word2Vec.load(model_file).wv                            # 모델 로드
      most_similar = model.most_similar(str(source_id))                             # source_id와 가장 유사한 문서들을 찾음, most_similar는 (문서 ID, 유사도 점수)의 튜플 리스트로 반환
      scores = [x[1] for x in most_similar]                                         # 유사도 점수 가져오기
      
      # target_ids = [x[0] for x in most_similar] # 문자열이 반환되는 에러가 발생
      target_ids = [int(x[0]) for x in most_similar]                                # 타겟 id문서 가져오기

      # 소스와 각 타깃 사이에 상위 10개 코사인 유사성 비교
      # X = np.repeat(td_matrix[source_id].todense(), 10, axis=0)
      # Y = td_matrix[target_ids].todense()                                         # sklearn의 cosine_similarity 함수가 np.matrix 형식을 지원 X
      X = np.repeat(np.asarray(td_matrix[source_id].todense()), 10, axis=0)         # X는 source_id 문서의 벡터를 10번 반복한 행렬 
      Y = np.asarray(td_matrix[target_ids].todense())                               # Y는 타겟 문서들의 벡터를 포함하는 행렬  
      # cosims = [cosine_similarity(X[i], Y[i])[0, 0] for i in range(10)]           # cosine_similarity 함수에 전달된 배열 X[i]와 Y[i]가 1D 배열, cosine_similarity 함수는 2D 배열 필요
      cosims = [cosine_similarity(X[i].reshape(1, -1), Y[i].reshape(1, -1))[0, 0] for i in range(10)]  # X와 Y의 각 행 사이의 코사인 유사도를 계산

      for i in range(10):                                                     # source_id와 가장 유사한 10개의 문서 ID, 그리고 그들 사이의 코사인 유사도와 Word2Vec 유사도를 출력
          # print("{:d} {:s} {:.3f} {:.3f}".format(
          #     source_id, target_ids[i], cosims[i], scores[i]))              # int 타입의 객체에 대해 '{:s}' 형식 지정자를 사용, s는 문자열
          print("{:d} {:d} {:.3f} {:.3f}".format(
              source_id, target_ids[i], cosims[i], scores[i]))    

  # 평가
  source_id = np.random.choice(E.shape[0])
  evaluate_model(TD, W2V_MODEL_FILE, source_id)

  ## 결과 : 주어진 문서 (source_id = 3782)와 다른 문서들 사이의 유사도
  # 3782 3667 0.000 0.618 >> source_id로 주어진 문서의 ID / 타겟 문서의 ID / 코사인  유사도(1에가까울수록 유사) / Word2Vec 유사도(1에가까울수록 유사)
  # 3782 1705 0.000 0.616
  # 3782 3162 0.014 0.599
  # 3782 174 0.025 0.588
  # 3782 2926 0.018 0.576
  # 3782 2124 0.000 0.568
  # 3782 3877 0.000 0.559
  # 3782 3035 0.003 0.558
  # 3782 547 0.003 0.556
  # 3782 2279 0.000 0.555

  ```

</details> 

<br><br>

## 문자와 부분단어 임베딩
단어 임베딩 전햑의 또 다른 진화는 단어 임베딩 대신 문자와 부분단어 임베딩
문자 수준의 임베딩 이점  
- 문자 어휘는 유한하고 작음 - ex 영어는 70자(글자 26, 숫자 10, 나머지 특수문자)
- 모든 단어를 표현할 수 있고 어휘밖의 단어는 존재하지 않음
- 자주 쓰이지 않는 단어나 철자가 틀린 글자, 다양한 형태의 단어 변형에 더 효율적으로 작동

<br>

### 문자 임베딩 
- 문자 임베딩은 단어를 개별 문자로 분해하여 각 문자에 대한 임베딩을 학습
- ex) apple이라는 단어는 ['a', 'p', 'p', 'l', 'e']로 분해
- 문자 수준의 임베딩은 종종 RNN, LSTM, 혹은 CNN과 같은 딥러닝 모델을 사용하여 단어나 문장의 표현을 학습하는 데 사용
- 문자 임베딩은 희귀 단어, 이름, 전문 용어, 오타 등의 문제를 처리하는 데 특히 

<br>

### 부분단어 임베딩 (Subword Embeddings):
- 부분단어 임베딩은 단어를 의미 있는 부분단어나 문자의 조합으로 분해하여 임베딩을 학습
- ex) unhappiness"는 "un-", "happy", "-ness"와 같은 부분단어로 분해
- FastText는 부분단어 임베딩을 사용하는 가장 대표적인 알고리즘 중 하나
- FastText는 단어를 n-gram 부분단어로 분해하고, 이러한 부분단어에 대한 임베딩을 학습
- 부분단어 임베딩은 다양한 언어와 다양한 형태의 단어 변형을 처리하는 데 유용하며, 특히 언어 간 번역이나 다양한 언어의 텍스트를 처리할 때 강점

<br><br>

## 동적 임베딩
- 이제까지 학습한 임베딩은 모두 정적 임베딩이며 다의어에 대해 약점을 가짐
- 동적 임베딩은 개별 단어뿐만 아니라 전체 시퀀스를 확인해 입력

<br>

###  CoVe(Contextualized Vectors) 
- 초기, 기계 번역 망의 인코더-디코더 쌍에서 인코더의 출력을 가져와서 동일한 단어의 단어 벡터와 연결

<br>

### ELMo(Emneddings from Language Models) 
- 큰 텍스트 코퍼스에 대해 양방향 LSTM (Bi-LSTM)을 사용하여 언어 모델을 학습합  
- 이를 통해 각 단어의 앞뒤 문맥을 함께 고려하여 임베딩을 생성
- 감정 분석, 개체명 인식, 질의 응답 등의 작업에서 ELMo를 사용하면 기존 방법보다 훨씬 더 정확한 결과 

## 문장과 문단 임베딩
- 문장과 문단 임베딩을 생성하는 간편하면서 쉬운 해결책은 구성 단어의 단어 벡터를 평균화 하는 것
- 문장/문단 임베딩은 그것들을 단어로 취급하고, 표준 단어 벡터를 사용해 각 단어를 표현하여 작업 최적화 방식으로 생성
- 텍스트의 연속성을 이용한 인코더-디코더 모델을 통해 주어진 문장의 주변문장을 예측하게 훈련
- 인코더-디코더 신경망으로 구성된 단어들의 벡터 표현은 일반적으로 사고(thought) 벡터라고 불리며, 문장 벡터는 스킵 사고(Skip-thought)
- 이 모델은 문장에서 임베딩을 생성하는데 사용할 수 있는 티아노(theano) 기반 모델이 되고, 구글 연구팀에 의해 텐서 플로로 구현됨

<br>

**예제 문장 두개로 호출하는 코드**
<details>
  <summary>실습 소스</summary>

  ```python
  import tensorflow as tf
  import tensorflow_hub as hub            # pip3 install tensorflow-hub

  module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"  # Universal Sentence Encoder 모델의 TensorFlow Hub URL을 설정
  tf.compat.v1.disable_eager_execution()                                # 즉시 실행을 비활성화, TensorFlow 2.x에서는 기본적으로 즉시 실행이 활성화

  model = hub.Module(module_url)                                        # Universal Sentence Encoder 모델을 로드
  embeddings = model([                                                  # 두 개의 문장에 대한 임베딩을 계산 - 문장을 고차원 벡터로 변환
      "i like green eggs and ham",
      "would you eat them in a box"
  ])
  with tf.compat.v1.Session() as sess:                                  # TensorFlow 1.x 스타일의 세션을 시작하고 모든 변수와 테이블을 초기화
      sess.run([
          tf.compat.v1.global_variables_initializer(),
          tf.compat.v1.tables_initializer()
      ])
      embeddings_value = sess.run(embeddings)                           # 초기화된 세션에서 임베딩 값을 계산

  print(embeddings_value.shape)                                         # 계산된 임베딩 값의 형태(차원)를 출력, 각 문장에 대한 임베딩 벡터의 크기

  ## 결과
  # (2, 512)
  # > 2: 두 개의 문장에 대한 임베딩이 계산, 입력으로 제공된 두 문장 각각에 대한 임베딩 벡터 존재
  # > 512: 각 문장의 임베딩 벡터의 차원이 512, Universal Sentence Encoder는 주어진 문장을 512차원의 벡터로 변환
  ```

</details> 

<br><br>

## 언어 모델 기반 임베딩
단어 임베딩 진화 과정의 다음 단계  
언어 모델은 단어 시퀀스에 대한 확률 분포 - 주어진 특정 단어 시퀀스에 대해 다음 단어를 예측하도록 훈련  
> ex)  I love to eat 이라는 문장이 주어졌을 때, 다음 단어로 apple이 올 확률을 예측  

훈련은 대규모 텍스트의 자연 문법 구조를 활용하기 때문에 비지도학습 과정  

<br>

**특징**
- 사전 훈련 : 대규모 텍스트 데이터셋에서 언어 모델을 사전 학습하여 일반적인 언어의 패턴과 구조를 학습
- 미세 조정 : 특정 애플리키이션 영역(여행, 쇼핑..)에 맞게끔 범용 언어모델을 미세 조정
- 깊은 신경망 : Transformer 아키텍처를 기반, 텍스트의 복잡한 의존성과 관계를 포착
- 동적 임베딩 : 문맥에 따라 단어의 임베딩을 동적으로 생성, 동일한 단어라도 다른 문맥에서 다른 임베딩을 가짐
- 대표적 모델 : GPT(OpenAI), BERT(Google AI), RoBERTa, T5...

<br>

### BERT를 특징 추출기로 사용
두가지 유형으로 제공 : BERT-Base, BERT-large   
사전 훈련은 계산량이 무척 많이 소요되며 현재는 TPU로만 가능
책에는 파라메타를 활용한 커멘드라인 실습 위주여서 개념 위주로 정리

**단계**
- 사전 훈련된 BERT 모델 로드: BERT는 다양한 언어와 크기의 버전으로 사전 훈련되어 있으며, 사전 훈련된 모델을 로드하여 사용 가능
- 입력 텍스트 토큰화: BERT는 특정 토크나이저를 사용하여 입력 텍스트를 토큰으로 분리, WordPiece라는 방법을 사용하여 텍스트를 서브워드 단위로 분리
- BERT에 입력: 토큰화된 텍스트를 BERT 모델에 입력하면, 각 토큰에 대한 임베딩 벡터 획득
- 특징 추출: BERT의 출력은 각 토큰에 대한 임베딩 벡터로 구성되며 문맥을 고려한 특징을 포함, 벡터를 사용하여 문장이나 문서의 의미를 표현
- 다운스트림 작업에 적용: BERT에서 추출된 특징 벡터는 다양한 다운스트림 NLP 작업에 사용
> ex)분류, 개체명 인식, 질의 응답 등의 작업에 BERT의 특징 벡터를 사용하여 성능을 향상

<br>

### BERT 미세조정
대규모 텍스트 데이터셋에서 사전 훈련되어 일반적인 언어 지식을 포함하고 있지만, 특정 작업에 최적화가 필요
특정 작업에 BERT를 적용하기 위해서는 미세조정을 통해 해당 작업의 데이터에 맞게 모델을 추가로 훈련

<br>

**단계**
- 사전 훈련된 BERT 모델 로드: BERT의 다양한 사전 훈련된 모델 중에서 원하는 모델을 선택하여 로드
- 작업에 맞는 헤드 추가: BERT 모델의 상단에 특정 작업에 맞는 헤드(예: 분류 레이어, 회귀 레이어 등)를 추가
- 데이터 준비: 미세조정을 위한 특정 작업의 데이터를 준비하고, BERT의 토크나이저를 사용하여 텍스트를 토큰화
- 미세조정: 작업의 데이터를 사용하여 BERT 모델을 추가로 훈련
- 평가 및 적용: 미세조정된 BERT 모델을 테스트 데이터셋에서 평가하고, 실제 작업에 적용

<br>

### BERT를 사용한 분류 : 커멘드 라인
미세조정할 필요 없이 사전 훈련된 모델 위에 분류기를 직접 구축 가능
실습 예에서는 단일 문장 분류에는 COLA 형식, 문장 쌍 분류에는 MRPC 형식을 사용

<br>

### BERT를 자신의 신경망 일부로 사용
BERT는 텐서플로 Hub에서 추정기로 사용할 수 있지만 현재는 tf.hub.KerasLayer로 호출할 수 없다는 점에서 텐서플로 2.x와 완전히 호환되지 않음
텐서플로 2.x에서 BERT를 사용하는 보편적인 방법은 HuggingFace Transformers 를 사용

<br>

**HuggingFace Transformers**
- BERT와 같은 널리 사용되는 다양한 트랜스포머 아키텍처에 대한 클래스 제공
- 여러 다운스트림 작업에서 미세 조정용 클레스 제공
- 파이토치(PyTorch)용으로 작성했지만 텐서플로우에서 호출할 수 있는 편의 클래스로 확장


<details>
  <summary>실습 소스</summary>

  ```python
  import os
  import tensorflow as tf
  import tensorflow_datasets    # pip3 install tensorflow-datasets
  from transformers import BertTokenizer, BertForSequenceClassification, TFBertForSequenceClassification, glue_convert_examples_to_features
  # > pip3 install transformers

  BATCH_SIZE = 32
  FINE_TUNED_MODEL_DIR = "./data/"                                                        # 미세 조정된 모델을 저장할 디렉토리 설정

  tokenizer = BertTokenizer.from_pretrained("bert-base-cased")                            # 사전 훈련된 BERT 토크나이저를 로드
  model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")              # 사전 훈련된 BERT 모델을 로드하여 분류 작업을 위해 사용

  # load data
  data, info = tensorflow_datasets.load("glue/mrpc", with_info=True)                      # GLUE MRPC 데이터세트를 로드
  num_train = info.splits["train"].num_examples                                           # 훈련 데이터의 예제 수를 가져옴
  num_valid = info.splits["validation"].num_examples                                      # 검증 데이터의 예제 수를 가

  # Prepare dataset for GLUE as a tf.data.Dataset instance
  Xtrain = glue_convert_examples_to_features(data["train"], tokenizer, 128, "mrpc")       # 훈련 데이터를 BERT 입력 형식으로 변환
  Xtrain = Xtrain.shuffle(128).batch(32).repeat(-1)                                       # 훈련 데이터를 섞고 배치 크기로 나눈 후 무한히 반복
  Xvalid = glue_convert_examples_to_features(data["validation"], tokenizer, 128, "mrpc")  # 검증 데이터를 BERT 입력 형식으로 변환
  Xvalid = Xvalid.batch(32)                                                               # 검증 데이터를 배치 크기로 나눔

  opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)                       # 최적화 알고리즘으로 Adam을 사
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)                  # 손실 함수로 SparseCategoricalCrossentropy를 사용
  metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")                         # 평가 지표로 정확도를 사용
  model.compile(optimizer=opt, loss=loss, metrics=[metric])                               # 모델 컴파일

  # Train and evaluate using tf.keras.Model.fit()
  train_steps = num_train // 32                                                           # 훈련 스탭수 계산
  valid_steps = num_valid // 32                                                           # 검증 스탭수 계싼

  history = model.fit(Xtrain, epochs=2, steps_per_epoch=train_steps,                      # 모델을 훈련 데이터로 훈련
      validation_data=Xvalid, validation_steps=valid_steps)

  model.save_pretrained(FINE_TUNED_MODEL_DIR)                                             # 미세 조정된 모델을 저장

  # load saved model
  saved_model = BertForSequenceClassification.from_pretrained(                            # 저장된 모델을 로드
      FINE_TUNED_MODEL_DIR, from_tf=True)

  # predict sentence paraphrase
  sentence_0 = "At least 12 people were killed in the battle last week."                  # 첫 번째 문장을 정의
  sentence_1 = "At least 12 people lost their lives in last weeks fighting."              
  sentence_2 = "The fires burnt down the houses on the street."

  inputs_1 = tokenizer.encode_plus(sentence_0, sentence_1, return_tensors="pt")           # 첫 번째와 두 번째 문장을 BERT 입력 형식으로 변환
  inputs_2 = tokenizer.encode_plus(sentence_0, sentence_2, return_tensors="pt")

  pred_1 = saved_model(**inputs_1)[0].argmax().item()                                     # 첫 번째 입력에 대한 예측(패러프레이즈 여부)을 수행
  pred_2 = saved_model(**inputs_2)[0].argmax().item()

  def print_result(id1, id2, pred):                                                       # 예측 결과를 출력하는 함수
      if pred == 1:
          print("sentence_1 is a paraphrase of sentence_0")
      else:
          print("sentence_1 is not a paraphrase of sentence_0")

  print_result(0, 1, pred_1)
  print_result(0, 2, pred_2)


  ## 결과
  # sentence_1 is a paraphrase of sentence_0 
  # > 첫번째와 두번째 문장은 비슷하기 때문에 패러프레이즈로 판단
  # sentence_1 is not a paraphrase of sentence_0 
  # > 첫번째와 세번째 문장은 완전히 다르기 때문에 패러프레이즈가 아닌것으로 판단
  #
  # 패러프레이즈(paraphrase) : 주어진 문장이나 문단의 의미를 변경하지 않으면서 다른 방식으로 표현한 문장이나 문단을 의미

  ```

</details> 
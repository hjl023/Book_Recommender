# -*- coding: utf-8 -*-
#  책 추천 프로그램
- 권장 실행 환경: Google Colab
- Data: `ffinal_data.pkl` (type: pandas.core.frame.DataFrame)
    - 교보문고 2023년 12월 기준 분야별 베스트셀러

## <크롤링>
"""

from selenium import webdriver
from tqdm import tqdm

# Chrome 드라이버 옵션 설정
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # 화면 표시 여부 (코딩 완료 후에는 끄면 됨)
options.add_argument("--disable-dev-shm-usage")  # Chrome 메모리 재한 해제
options.add_argument("--no-sandbox")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 5.1;64; x64) rv:47.10) Gecko/20100101 Firefox/47.0")

# Chrome 드라이버 생성
driver = webdriver.Chrome(options=options)

# Kyobo 판매상품 정보 추출
book_info = []

for n in tqdm(kyobobook.판매상품10):
    url = f"https://product.kyobobook.co.kr/detail/{n['이지']}"
    driver.get(url)
    time.sleep(3)

    try:
        genre = driver.find_element(By.CLASS_NAME, "category_list_item").text
        contents = driver.find_element(By.CLASS_NAME, "Intro_bottom").text
        index = driver.find_element(By.CLASS_NAME, "book_contents_items").text

        book_info.append((genre, contents, index))
    except:
        book_info.append((n))
        continue

# 판매상품 정보 데이터프레임으로 변환
book = pd.DataFrame(book_info)

# 데이터프레임 컬럼명 설정
book.columns = ["판매상품10", "분야", "책 소개", "책 목차"]

# 인덱스 재설정
book = book.reset_index()

# 인덱스 삭제
del book["Index"]

# Kyobo 판매상품 정보와 합병
book_final = pd.concat([kyob, book], axis=1)

# 불필요한 컬럼 삭제
del book_final["판매상품10"]

# 데이터프레임 저장
book_final.to_pickle("book_data.pkl")

"""## <환경설치>"""

# 퍄키지 설치
!pip install konlpy # konlpy 설치
!pip install summa # 텍스트 요약
# !pip install -U --no-cache-dir gdown --pre
# !gdown --no-cookies --id 1KuBG40WNpVPV1ilfGaiCI2D3l8JVSfbS

# 라이브러리 불러오기 및 객체 생성
from konlpy.tag import Hannanum, Okt, Kkma
from summa import keywords
from sklearn.metrics import jaccard_score
# import gensim
# from gensim.models.word2vec import Word2Vec
import re
import pandas as pd
from tqdm import tqdm, tqdm_pandas
import gensim
import pickle

tqdm_pandas(tqdm())
hannanum = Hannanum()
kkma = Kkma()
okt = Okt()

"""## 함수 정의"""

stopwords = ['저자', '작가', '소개', '인기', '우리', '계기', '보유', '내용', '이야기'] # 불용어

# 키워드 추출 함수
def preprocess(string) :
    string = ' '.join(re.findall('[가-힣]+|[a-z]+', string.lower())) # 특수문자들 제거 및 모든 영어 소문자 변환
    string = okt.nouns(string)
    for ind, i in enumerate(string) :
        if i in stopwords :
            string[ind] = ''
    string = ' '.join(string)
    string = ' '.join(re.findall('\w{2,}', string.lower())) # 1글자인 단어들 삭제
    words_n = 5 # 뽑을 키워드 갯수

    if len(set(string.split())) < words_n :
        words_n = len(set(string.split())) # 단어 갯수가 적을 경우 뽑을 키워드 갯수를 낮춘다

    string = keywords.keywords(string, words = words_n, scores = True) # 키워드 뽑기

    word = []
    score = []
    for i, j in string :
        i = ''.join(i).split()
        word.append(i)
    word_list = []
    for i in word :
        for j in i :
            word_list.append(j)
    for i in word_list : # 단어들 1개씩
        for ind, j in enumerate(word) :
            if i in j :
                score.append([i, round(string[ind][1], 2)])
    if len(score) > 5 :
        score = score[:5]

    return score

# 키워드들을 딕셔너리 형태로 변환
def DicTrans(keyword) :

    intro = [i[0] for i in keyword]

    dic = {}
    for ind, i in enumerate(intro) :
        dic[i] = keyword[ind][1]

    return dic

# 인풋데이터와, 책 리스트의 자카드 유사도 계산
def JarccardSimilarity (input_key, data_key) :
    score = 0

    in_keys = set(input_key.keys())
    da_keys = set(data_key.keys())
    same = in_keys & da_keys
    for i in list(same) :
        score += (input_key[i] * 1.7) + (data_key[i] * 1.3) # 가중치: 입력 1.7 , 데이터베이스 1.3
    a = set.union(in_keys ,da_keys)
    return score / len(a)

# 제목의 자카드 유사도 계산
def JarccardSimilarity_title (input_key, data_key) :
    score = 0

    in_keys = set(input_key.keys())
    da_keys = set(data_key)
    same = in_keys & da_keys
    for i in list(same) :
        score += (input_key[i] * 1.7) + 1.3
    a = set.union(in_keys, da_keys)
    return score / len(a)

def Recommendation(input_text, data_set, Book_Count):

    input_text = preprocess(input_text)
    input_text = DicTrans(input_text)

    intro_score = data_set['소개글_키워드'].apply(lambda a: JarccardSimilarity(input_text, a))
    index_score = data_set['목차_키워드'].apply(lambda a: JarccardSimilarity(input_text, a))
    title_score = data_set['상품명_Hannanum'].apply(lambda a: JarccardSimilarity_title(input_text, a))

    intro_sim_list = intro_score.sort_values(ascending=False)[:Book_Count]
    index_sim_list = index_score.sort_values(ascending=False)[:Book_Count]
    title_sim_list = title_score.sort_values(ascending=False)[:Book_Count]

    # 각 유사도 기준에 따른 데이터프레임 생성
    intro_df = []
    index_df = []
    title_df = []

    for title, t, result_df in zip(['소개글 기반 유사도', '목차 기반 유사도', '제목 기반 유사도'],
                                   [intro_sim_list, index_sim_list, title_sim_list],
                                   [intro_df, index_df, title_df]):
        for key, score in zip(t.index, t): # 인덱스 번호랑 유사도 점수
            if score == 0:
                continue
            book_info = []
            for i in ['상품명', '저자', '소개글', '목차', '분야', '정가'] :
                book_info.append(data_set.loc[key][i])
#            book_info = data_set.loc[key, '상품명', '저자', '소개글', '목차', '분야', '정가']
            book_info.append(float(score))
            book_info.append(title)
            result_df.append(book_info)
    intro_df = pd.DataFrame(intro_df, columns = ['상품명', '저자', '소개글', '목차', '분야', '정가', '유사도', '유사도 기준'])
    index_df = pd.DataFrame(index_df, columns = ['상품명', '저자', '소개글', '목차', '분야', '정가', '유사도', '유사도 기준'])
    title_df = pd.DataFrame(title_df, columns = ['상품명', '저자', '소개글', '목차', '분야', '정가', '유사도', '유사도 기준'])
    return intro_df, index_df, title_df

"""## 실행파일

#### (테스트용)
"""

# 책 리스트 불러오기
with open('ffinal_data.pkl', 'rb') as f:
    Books = pickle.load(f).reset_index(drop=True)

# 입력 받기
# Book_Field = int(input("분야를 입력 하시겠습니까? (True(입력): 1, False(미입력): 0): "))
Book_Field = 1
print()

if bool(Book_Field):
    print('<분야 입력 제한 범위> \n', Books['분야'].unique())
    print()
    # Your_Field = input("분야를 입력 해주세요: ")
    Your_Field = '인문'
    book_list = Books[Books['분야'] == Your_Field]
    print()
else:
    book_list = Books

# Book_Count = int(input("추천 받을 책 권수를 입력해주세요 (0 이상의 정수): "))
Book_Count =3
print()

# Your_Text = input('과제를 입력 해주세요: ')
text = """이 과제는 인류학적 접근을 통해 자본주의 경제 시스템이 인간 사회와 문화에 미치는 다양하고 복잡한
 영향을 체계적으로 탐구하며, 학생들은 자본주의의 형성과 확산, 소비문화, 불평등, 글로벌화, 환경파괴 등 다양한
  측면에서 인류학의 독특한 관점을 활용하여 현대 사회의 복잡한 사회 경제적 문제와 그 영향을 사회학적,
  문화인류학적, 공동체적 시각에서 심층적으로 이해하고 분석하며, 인간의 역사와 문화와 함께 성장한 자본주의와의
  관계에 대한 통찰력을 개발하고자 하는 목적을 달성하기 위한 레포트를 작성하도록 합니다."""
Your_Text = text
print()

your_keyword = preprocess(Your_Text) # 입력 문장에서 키워드 추출

# 키워드 출력 구문
print('<입력 문장의 키워드>')
for i in range(len(your_keyword)):
    print(your_keyword[i][0], your_keyword[i][1], sep=' : ')
print()

# 유사도 추출 후, 책 추천
if bool(Book_Field):
    field_data = Books[Books['분야'] == Your_Field]
    not_field_data = Books[Books['분야'] != Your_Field]

    print(f'선택하신 분야: {Your_Field}의 도서 추천 리스트 입니다', end='\n\n')
    intro_fd, index_fd, title_fd = Recommendation(Your_Text, field_data, Book_Count)

    print('<"소개글" 유사도 기반>', end='\n\n') # 한 줄
    display(intro_fd)
    print('\n\n') # 두 줄

    print('<"목차" 유사도 기반>', end='\n\n')
    display(index_fd)
    print('\n\n')

    print('<제목 유사도 기반>', end='\n\n')
    display(title_fd)
    print('\n\n')

    print('=' * 300)

    print('\n\n')
    print('선택하신 분야 외의 도서 추천 리스트 입니다', end='\n\n')
    intro_re_fd, index_re_fd, title_re_fd = Recommendation(Your_Text, not_field_data, Book_Count)

    print('<"소개글"> 유사도 기반', end='\n\n')
    display(intro_re_fd)
    print('\n\n')

    print('"<목차>" 유사도 기반', end='\n\n')
    display(index_re_fd)
    print('\n\n')

    print('"<제목>" 유사도 기반', end='\n\n')
    display(title_re_fd)

    # 다운파일
    Book_Recommdation = pd.concat([intro_fd, index_fd, title_fd, intro_re_fd, index_re_fd, title_re_fd]).reset_index(drop=True)
    Book_Recommdation.to_excel('Book_Recommdation.xlsx')

else:
    intro_all, index_all, title_all = Recommendation(Your_Text, Books, Book_Count)

    print('도서 추천 리스트 입니다', end='\n\n')

    print('<"소개글"> 유사도 기반', end='\nn\n')
    display(intro_all)
    print('\n\n')

    print('<"목차"> 유사도 기반', end='\n\n')
    display(index_all)
    print('\n\n')

    print('<"제목"> 유사도 기반', end='\n\n')
    display(title_all)

    # 다운파일
    Book_Recommdation = pd.concat([intro_all, index_all, title_all]).reset_index(drop=True)
    Book_Recommdation.to_excel('Book_Recommdation.xlsx')

# 책 리스트 불러오기
with open('ffinal_data.pkl', 'rb') as f:
    Books = pickle.load(f).reset_index(drop=True)

# 입력 받기
Book_Field = int(input("분야를 입력 하시겠습니까? (True(입력): 1, False(미입력): 0): "))
print()

if bool(Book_Field):
    print('<분야 입력 제한 범위> \n', Books['분야'].unique())
    print()
    Your_Field = input("분야를 입력 해주세요: ")
    book_list = Books[Books['분야'] == Your_Field]
    print()
else:
    book_list = Books

Book_Count = int(input("추천 받을 책 권수를 입력해주세요 (0 이상의 정수): "))
print()

Your_Text = input('과제를 입력 해주세요: ')
print()

your_keyword = preprocess(Your_Text) # 입력 문장에서 키워드 추출

# 키워드 출력 구문
print('<입력 문장의 키워드>')
for i in range(len(your_keyword)):
    print(your_keyword[i][0], your_keyword[i][1], sep=' : ')
print()

# 유사도 추출 후, 책 추천
if bool(Book_Field):
    field_data = Books[Books['분야'] == Your_Field]
    not_field_data = Books[Books['분야'] != Your_Field]

    print(f'선택하신 분야: {Your_Field}의 도서 추천 리스트 입니다', end='\n\n')
    intro_fd, index_fd, title_fd = Recommendation(Your_Text, field_data, Book_Count)

    print('<"소개글" 유사도 기반>', end='\n\n') # 한 줄
    display(intro_fd)
    print('\n\n') # 두 줄

    print('<"목차" 유사도 기반>', end='\n\n')
    display(index_fd)
    print('\n\n')

    print('<제목 유사도 기반>', end='\n\n')
    display(title_fd)
    print('\n\n')

    print('=' * 300)

    print('\n\n')
    print('선택하신 분야 외의 도서 추천 리스트 입니다', end='\n\n')
    intro_re_fd, index_re_fd, title_re_fd = Recommendation(Your_Text, not_field_data, Book_Count)

    print('<"소개글"> 유사도 기반', end='\n\n')
    display(intro_re_fd)
    print('\n\n')

    print('"<목차>" 유사도 기반', end='\n\n')
    display(index_re_fd)
    print('\n\n')

    print('"<제목>" 유사도 기반', end='\n\n')
    display(title_re_fd)

    # 다운파일
    Book_Recommdation = pd.concat([intro_fd, index_fd, title_fd, intro_re_fd, index_re_fd, title_re_fd]).reset_index(drop=True)
    Book_Recommdation.to_excel('Book_Recommdation.xlsx')

else:
    intro_all, index_all, title_all = Recommendation(Your_Text, Books, Book_Count)

    print('도서 추천 리스트 입니다', end='\n\n')

    print('<"소개글"> 유사도 기반', end='\n\n')
    display(intro_all)
    print('\n\n')

    print('<"목차"> 유사도 기반', end='\n\n')
    display(index_all)
    print('\n\n')

    print('<"제목"> 유사도 기반', end='\n\n')
    display(title_all)

    # 다운파일
    Book_Recommdation = pd.concat([intro_all, index_all, title_all]).reset_index(drop=True)
    Book_Recommdation.to_excel('Book_Recommdation.xlsx')
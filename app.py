import streamlit as st
import kss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from kiwipiepy import Kiwi
import re

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="강의 노트 요약 서비스", layout="wide")

# --------------------------------------------------------------------------
# [1] 사용자 원본 분석기 로직 (절대 수정하지 않음)
# --------------------------------------------------------------------------

@st.cache_resource
def get_kiwi():
    return Kiwi()

kiwi = get_kiwi()

def noun_tokenizer(text):
    """Kiwi를 사용하여 텍스트에서 명사(NNG, NNP)만 추출"""
    tokens = kiwi.tokenize(text)
    nouns = []
    for token in tokens:
        if token.tag in ['NNG', 'NNP']:
            if len(token.form) > 1: 
                nouns.append(token.form)
    return nouns

def extract_keywords(text, num_keywords=10):
    """TF-IDF를 사용하여 텍스트에서 핵심 키워드를 추출"""
    tfidf_vectorizer = TfidfVectorizer(tokenizer=noun_tokenizer)
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    except ValueError as e:
        if "empty vocabulary" in str(e):
            return []
        else:
            raise e
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    sorted_keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:num_keywords]]

def extract_sentences(text, keywords, num_sentences=5):
    """텍스트를 문장별로 분리하고 핵심 문장 추출"""
    try:
        sentences = kss.split_sentences(text)
    except:
        sentences = text.split('.')
        
    if not sentences:
        return []
    keyword_dict = {word: 1 for word in keywords}
    sentence_scores = []
    for sentence in sentences:
        score = 0
        nouns_in_sentence = noun_tokenizer(sentence)
        for noun in nouns_in_sentence:
            if noun in keyword_dict:
                score += 1
        sentence_scores.append(score)
    sorted_sentence_indices = np.argsort(sentence_scores)[::-1]
    top_sentence_indices = sorted(sorted_sentence_indices[:num_sentences])
    key_sentences = [sentences[i] for i in top_sentence_indices]
    
    return key_sentences

# --------------------------------------------------------------------------
# [2] 웹 UI 구성 (입출력 연결 - 개선됨)
# --------------------------------------------------------------------------

st.title("📜 강의 노트 요약 프로그램")
st.markdown("Command + Enter로 실행 가능합니다.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("입력")
    
    # [변경점] st.form을 사용하여 'Command + Enter' 지원 및 버튼 위치 고정
    with st.form(key='summary_form'):
        # height를 500 -> 250으로 줄여서 버튼이 바로 보이게 함
        input_text = st.text_area("강의 노트를 붙여넣으세요", height=250)
        # 폼 제출 버튼 (이걸 누르거나, 입력창에서 Cmd+Enter 치면 실행됨)
        submit_btn = st.form_submit_button("요약 실행")

# 버튼을 누르거나(submit_btn), 폼 안에서 엔터를 치면 실행됨
if submit_btn and input_text:
    
    # --- 사용자 원본 실행 로직 (그대로 유지) ---
    lecture_note = re.sub(r'\s+', ' ', input_text)
    
    KEYWORD_RATIO = 0.2
    MIN_KEYWORDS = 5
    SENTENCE_RATIO = 0.3
    MIN_SENTENCES = 3

    all_nouns = noun_tokenizer(lecture_note)
    
    if not all_nouns:
        unique_noun_count = 0
        NUM_KEYWORDS = 0
    else:
        unique_noun_count = len(set(all_nouns))
        num_keywords = max(MIN_KEYWORDS, int(unique_noun_count * KEYWORD_RATIO))
        NUM_KEYWORDS = min(num_keywords, unique_noun_count)

    try:
        all_sentences = kss.split_sentences(lecture_note)
        total_sentence_count = len(all_sentences)
    except:
        total_sentence_count = lecture_note.count('.') 

    if total_sentence_count == 0:
        NUM_SENTENCES = 0
    else:
        num_sentences = max(MIN_SENTENCES, int(total_sentence_count * SENTENCE_RATIO))
        NUM_SENTENCES = min(num_sentences, total_sentence_count)

    keywords = extract_keywords(lecture_note, NUM_KEYWORDS)
    key_sentences = extract_sentences(lecture_note, keywords, NUM_SENTENCES)
    # ---------------------------------------------

    with col2:
        st.subheader("결과")
        st.info(f"(전체 문장: {total_sentence_count}개, 분석 명사: {unique_noun_count}개)")
        
        st.markdown(f"#### 🔑 핵심 키워드 (상위 {NUM_KEYWORDS}개)")
        st.write(", ".join(keywords)) 
        
        st.divider()
        
        st.markdown(f"#### 🎯 핵심 요약 문장 (상위 {NUM_SENTENCES}개)")
        for i, sentence in enumerate(key_sentences):
            st.success(f"{i+1}. {sentence.strip()}")
            
elif submit_btn and not input_text:
    st.warning("입력된 내용이 없습니다.")

import streamlit as st
from project import collab_recommend, content_recommend, hybrid_recommend, book_similar, books_by_genre, books, transactions
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Bookflix", page_icon="📚", layout="wide")

# ------------------ ✨ Custom CSS for Font Size ------------------
st.markdown("""
    <style>
    h1 {font-size: 60px !important; color: #e50914; font-weight: 800;}
    h2, h3, h4 {font-size: 26px !important;}
    p, li, label, div[data-testid="stMarkdownContainer"] {
        font-size: 20px !important;
    }
    .stSelectbox label, .stTextInput label {
        font-size: 22px !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Page Title ----------------
st.title("📚Bookflix")
st.subheader("Because choosing the next great read shouldn’t be a guessing game.")

# ---------------- Student ID input ----------------
student_id = st.text_input("Enter your Student ID:").strip()
if student_id:
    st.markdown("### Personalized Recommendations")
    
    content_books = content_recommend(student_id)
    collab_books = collab_recommend(student_id)
    hybrid_books = hybrid_recommend(student_id)

    # Function to get full book info
    def get_book_info(book_ids):
        df = books[books['book_id'].isin(book_ids)][['title', 'author', 'genre']]
        # Format: Title by Author (Genre)
        return [f"**{row['title']}** by *{row['author']}* ({row['genre']})" 
                for _, row in df.iterrows()]

    st.markdown("**✨You May Like:**")
    for i, info in enumerate(get_book_info(content_books), 1):
        st.write(f"{i}. {info}")

    st.markdown("**👥People Who Read Similar Books Also Liked:**")
    for i, info in enumerate(get_book_info(collab_books), 1):
        st.write(f"{i}. {info}")

    st.markdown("**💡Top Picks For You:**")
    for i, info in enumerate(get_book_info(hybrid_books), 1):
        st.write(f"{i}. {info}")

# ------------------ 🔍 Explore by Book (Simple Dropdown Version) ------------------
books['content'] = (
    books['title'].fillna('') + ' ' +
    books['author'].fillna('') + ' ' +
    books['genre'].fillna('')
).str.lower()

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(books['content'])
cosine_sim = cosine_similarity(tfidf_matrix)

st.markdown("---")
st.subheader("📖 Explore by Book")

book_titles = sorted(books['title'].dropna().tolist())
selected_book = st.selectbox("Select a book you like:", options=book_titles)

if selected_book:
    idx = books[books['title'] == selected_book].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    book_indices = [i[0] for i in sim_scores]
    rec_df = books.iloc[book_indices][['title', 'author', 'genre']]

    st.markdown(f"### 📚 Because you liked **{selected_book}**, you may also enjoy:")
    for _, row in rec_df.iterrows():
        st.write(f"**{row['title']}** by *{row['author']}* ({row['genre']})")

# ---------------- Explore by Genre ----------------
st.markdown("---")
st.subheader("🎨 Explore by Genre")
genres = sorted(books['genre'].dropna().unique().tolist())
selected_genre = st.selectbox("Select a genre:", options=genres)

if selected_genre:
    df_genre = books_by_genre(selected_genre)
    st.markdown(f"Books in genre **{selected_genre}**:")
    for _, row in df_genre.iterrows():
        st.write(f"**{row['title']}** by {row['author']}")

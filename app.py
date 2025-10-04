import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load & Clean
books = pd.read_csv("Books.csv")
transactions = pd.read_csv("Transactions.csv")
books.columns = [c.lower() for c in books.columns]
transactions.columns = [c.lower() for c in transactions.columns]

for col in ['book_id', 'student_id']:
    transactions[col] = transactions[col].astype(str).str.strip()
    if col == 'book_id': books[col] = books[col].astype(str).str.strip()

popular_books = transactions['book_id'].value_counts().index.tolist()

# Collaborative Filtering
user_item = transactions.pivot_table(index='student_id', columns='book_id', aggfunc='size', fill_value=0)
n_components = min(50, min(user_item.shape)-1) if min(user_item.shape) > 1 else 1
svd = TruncatedSVD(n_components=n_components, random_state=42)
pred_df = pd.DataFrame(np.dot(svd.fit_transform(user_item), svd.components_), index=user_item.index, columns=user_item.columns)

def collab_recommend(student_id, top_n=10):
    if student_id not in pred_df.index: return popular_books[:top_n]
    scores = pred_df.loc[student_id]
    borrowed = transactions[transactions['student_id']==student_id]['book_id'].tolist()
    scores = scores.drop(index=borrowed, errors='ignore')
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

# Content-Based Filtering
books['content'] = (books['title'] + ' ' + books['author'] + ' ' + books['genre']).str.lower()
tfidf_matrix = TfidfVectorizer(stop_words='english', max_features=5000).fit_transform(books['content'])
cosine_sim = cosine_similarity(tfidf_matrix)
bookid_to_idx = {bid: i for i, bid in enumerate(books['book_id'])}
idx_to_bookid = {i: bid for bid, i in bookid_to_idx.items()}

def content_recommend(student_id, top_n=10):
    borrowed = transactions[transactions['student_id']==student_id]['book_id'].tolist()
    if not borrowed: return popular_books[:top_n]
    sim_scores = np.zeros(len(books))
    for b in borrowed:
        if b in bookid_to_idx: sim_scores += cosine_sim[bookid_to_idx[b]]
    for b in borrowed:
        if b in bookid_to_idx: sim_scores[bookid_to_idx[b]] = 0
    return [idx_to_bookid[i] for i in np.argsort(sim_scores)[::-1][:top_n]]

# Rank-Based Hybrid Recommendation
scaler = MinMaxScaler()
def hybrid_recommend(student_id, top_n=10):
    c_recs, s_recs = content_recommend(student_id,50), collab_recommend(student_id,50)
    # Keep unique collaborative books not in content-based
    s_unique = [b for b in s_recs if b not in c_recs]

    # Assign rank scores
    c_rank = {b: 50-i for i,b in enumerate(c_recs)}
    s_rank = {b: 50-i for i,b in enumerate(s_unique)}

    # Normalize
    if c_rank:
        vals = np.array(list(c_rank.values())).reshape(-1,1)
        scaled = scaler.fit_transform(vals).flatten()
        c_rank = {k: scaled[i] for i,k in enumerate(c_rank.keys())}
    if s_rank:
        vals = np.array(list(s_rank.values())).reshape(-1,1)
        scaled = scaler.fit_transform(vals).flatten()
        s_rank = {k: scaled[i] for i,k in enumerate(s_rank.keys())}

    all_books = list(dict.fromkeys(c_recs + s_unique + popular_books[:100]))
    final_scores = {b: 0.5*c_rank.get(b,0) + 0.5*s_rank.get(b,0) for b in all_books}
    return [b for b,_ in sorted(final_scores.items(), key=lambda x:x[1], reverse=True)[:top_n]]

# Streamlit UI
st.set_page_config(page_title="📚 Bookflix", layout="centered")
st.title("📚 Bookflix")
st.markdown("Get personalized book suggestions based on your reading history!")

# Search bar for Student ID
student_input = st.text_input("Enter your Student ID:")

def show_books(book_ids):
    df = books[books['book_id'].isin(book_ids)][['title','author','genre']]
    for i,row in df.iterrows():
        st.write(f"**{row['title']}** by *{row['author']}* ({row['genre']})")

if st.button("🔍 Show Recommendations") and student_input.strip():
    student_id = student_input.strip()
    content_books = content_recommend(student_id)
    collab_books = collab_recommend(student_id)
    hybrid_books = hybrid_recommend(student_id)

    st.subheader("✨ You may like:")
    show_books(content_books)

    st.subheader("👥 People who read similar books also liked:")
    show_books(collab_books)

    st.subheader("💡 Top Picks for You:")
    show_books(hybrid_books)
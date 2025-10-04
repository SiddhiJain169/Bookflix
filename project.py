import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

#Load & Clean
books = pd.read_csv("Books.csv")
transactions = pd.read_csv("Transactions.csv")
books.columns = [c.lower() for c in books.columns]
transactions.columns = [c.lower() for c in transactions.columns]

for col in ['book_id', 'student_id']:
    transactions[col] = transactions[col].astype(str).str.strip()
    books[col] = books[col].astype(str).str.strip() if col == 'book_id' else None

popular_books = transactions['book_id'].value_counts().index.tolist()

#Collaborative Filtering
user_item = transactions.pivot_table(index='student_id', columns='book_id', aggfunc='size', fill_value=0)
n_components = min(50, min(user_item.shape)-1) if min(user_item.shape) > 1 else 1
svd = TruncatedSVD(n_components=n_components, random_state=42)
matrix_reduced = svd.fit_transform(user_item)
pred_ratings = np.dot(matrix_reduced, svd.components_)
pred_df = pd.DataFrame(pred_ratings, index=user_item.index, columns=user_item.columns)

def collab_recommend(student_id, top_n=10):
    if student_id not in pred_df.index:
        return popular_books[:top_n]
    scores = pred_df.loc[student_id]
    borrowed = transactions[transactions['student_id']==student_id]['book_id'].tolist()
    scores = scores.drop(index=borrowed, errors='ignore')
    return scores.sort_values(ascending=False).head(top_n).index.tolist()

#Content-Based Filtering
books['content'] = (books['title'] + ' ' + books['author'] + ' ' + books['genre']).str.lower()
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(books['content'])
cosine_sim = cosine_similarity(tfidf_matrix)

bookid_to_idx = {bid: i for i, bid in enumerate(books['book_id'])}
idx_to_bookid = {i: bid for bid, i in bookid_to_idx.items()}

def content_recommend(student_id, top_n=10):
    borrowed = transactions[transactions['student_id']==student_id]['book_id'].tolist()
    if not borrowed:
        return popular_books[:top_n]
    sim_scores = np.zeros(len(books))
    for b in borrowed:
        if b in bookid_to_idx:
            sim_scores += cosine_sim[bookid_to_idx[b]]
    for b in borrowed:
        if b in bookid_to_idx:
            sim_scores[bookid_to_idx[b]] = 0
    top_indices = np.argsort(sim_scores)[::-1][:top_n]
    return [idx_to_bookid[i] for i in top_indices]

#Hybrid Recommendation
scaler = MinMaxScaler()

def hybrid_recommend(student_id, top_n=10, w_content=0.6, w_collab=0.4):
    c_recs = content_recommend(student_id, top_n=50)
    s_recs = collab_recommend(student_id, top_n=50)
    all_cands = list(dict.fromkeys(c_recs + s_recs + popular_books[:100]))

    c_scores = {b: (len(c_recs)-i) for i,b in enumerate(c_recs)}
    s_scores = {b: (len(s_recs)-i) for i,b in enumerate(s_recs)}

    def normalize(d):
        if not d: return {}
        vals = np.array(list(d.values())).reshape(-1,1)
        scaled = scaler.fit_transform(vals).flatten()
        return {k: scaled[i] for i,k in enumerate(d.keys())}

    c_scores = normalize(c_scores)
    s_scores = normalize(s_scores)

    final = {b: w_content*c_scores.get(b,0)+w_collab*s_scores.get(b,0) for b in all_cands}
    recs = sorted(final.items(), key=lambda x:x[1], reverse=True)
    return [b for b,_ in recs[:top_n]]

if __name__ == "__main__":
    student_ids = transactions['student_id'].unique()
    sample_id = student_ids[0]

    content_books = content_recommend(sample_id, top_n=10)
    collab_books = collab_recommend(sample_id, top_n=10)
    hybrid_books = hybrid_recommend(sample_id, top_n=10)

    def titles_from_ids(book_ids):
        return books[books['book_id'].isin(book_ids)]['title'].tolist()

    print(f"\n📘 Recommendations for Student ID: {sample_id}\n")
    print("✨ You may like (Content-Based):")
    for i,b in enumerate(titles_from_ids(content_books),1):
        print(f"  {i}. {b}")
    print("\n👥 People who read similar books also liked (Collaborative):")
    for i,b in enumerate(titles_from_ids(collab_books),1):
        print(f"  {i}. {b}")
    print("\n💡 Top Picks for You (Hybrid):")
    for i,b in enumerate(titles_from_ids(hybrid_books),1):
        print(f"  {i}. {b}")
    print("\n✅ Done.")
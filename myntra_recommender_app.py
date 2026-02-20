import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

st.set_page_config(page_title="Myntra Recommender", layout="centered")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; max-width: 950px; margin: auto; }

    .rec-title {
        font-size: 20px; font-weight: 700;
        color: #FF3F6C; margin-bottom: 4px;
    }

    .user-banner {
        background: linear-gradient(135deg, #fff0f4 0%, #ffe4ec 50%, #ffd6e0 100%);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        border: 1px solid #ffb3c6;
        box-shadow: 0 3px 12px rgba(255,63,108,0.10);
    }
    .user-banner .you   { font-size: 11px; font-weight: 600; letter-spacing: 1.4px;
                          text-transform: uppercase; color: #FF3F6C; margin-bottom: 4px; }
    .user-banner .name  { font-size: 26px; font-weight: 800; color: #1a1a1a; margin-bottom: 8px; }
    .user-banner .sim   { font-size: 13px; color: #555; margin-top: 2px; }
    .user-banner .peer  { font-weight: 700; color: #FF3F6C; }

    .card {
        border-radius: 12px;
        padding: 18px 16px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        height: 100%;
        min-height: 180px;
    }
    .card-0 { background: linear-gradient(145deg, #fff5f7, #ffe8ee); }
    .card-1 { background: linear-gradient(145deg, #f0f7ff, #deeeff); }
    .card-2 { background: linear-gradient(145deg, #f5fff5, #dff5e3); }
    .card-3 { background: linear-gradient(145deg, #fffbf0, #fff0cc); }
    .card-4 { background: linear-gradient(145deg, #f8f0ff, #eedeff); }
    .card .brand  { font-size: 15px; font-weight: 700; color: #222;
                    text-transform: uppercase; margin-bottom: 3px; }
    .card .type   { font-size: 13px; color: #666; margin-bottom: 10px; }
    .card .meta   { font-size: 12px; color: #999; margin-bottom: 10px; }
    .card .price  { font-size: 16px; font-weight: 700; color: #222; }
    .card .rating { font-size: 12px; color: #FF9900; margin-top: 4px; }
    .stButton > button {
        background-color: #FF3F6C !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 28px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        transition: background 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #e0325a !important;
    }

    .stSelectbox > label {
        color: #FF3F6C !important;
        font-weight: 600 !important;
    }
    .stSelectbox > div > div {
        background: linear-gradient(135deg, #fff0f4, #ffe4ec) !important;
        color: #1a1a1a !important;
        border-radius: 8px !important;
        border: 1.5px solid #ffb3c6 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .stSelectbox > div > div > div {
        color: #1a1a1a !important;
    }
    .stSelectbox svg {
        fill: #FF3F6C !important;
    }

    .card .reason { font-size: 11px; color: #aaa; margin-top: 8px;
                    border-top: 1px solid #f0f0f0; padding-top: 8px; }
    .card .badge  { display:inline-block; background:#fff0f4; color:#FF3F6C;
                    border-radius:4px; padding:2px 8px; font-size:11px;
                    font-weight:600; margin-bottom:8px; }

    .context-bar {
        background: #f9f9f9;
        border-left: 3px solid #FF3F6C;
        padding: 10px 16px;
        border-radius: 6px;
        font-size: 14px;
        color: #333;
        margin-bottom: 18px;
    }

    .viewed-banner {
        background: linear-gradient(135deg, #f0f7ff 0%, #deeeff 50%, #cce4ff 100%);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        border: 1px solid #a8d0f5;
        box-shadow: 0 3px 12px rgba(74,144,217,0.10);
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .viewed-label {
        font-size: 11px; font-weight: 600; letter-spacing: 1.4px;
        text-transform: uppercase; color: #4A90D9;
    }
    .viewed-product {
        font-size: 24px; font-weight: 800; color: #1a1a1a;
    }
    .viewed-meta {
        font-size: 13px; color: #666; margin-top: 2px;
    }

</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#FF3F6C; font-weight:800;'>Myntra Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ffffff; font-size:13px; font-weight:500; letter-spacing:0.5px;'>Content-Based &nbsp;&nbsp;|&nbsp;&nbsp; Collaborative — User to User &nbsp;&nbsp;|&nbsp;&nbsp; Collaborative — Item to Item &nbsp;&nbsp;|&nbsp;&nbsp; Hybrid</p>", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("myntra_shopping_dataset.xlsx")
    user_item_df = df[['user_female', 'product_id', 'rating']].rename(columns={'user_female': 'user'}).drop_duplicates()
    products_df  = df.drop_duplicates(subset='product_id').reset_index(drop=True)
    return products_df, user_item_df

products_df, user_item_df = load_data()


# ─────────────────────────────────────────
# PRE-COMPUTE MODELS
# ─────────────────────────────────────────
@st.cache_resource
def build_tfidf(_products_df):
    df = _products_df.copy()
    df['description'] = (
        df['brand'].fillna('') + ' ' + df['category'].fillna('') + ' ' +
        df['sub_category'].fillna('') + ' ' + df['color'].fillna('') + ' ' +
        df['occasion'].fillna('') + ' ' + df['age_group'].fillna('')
    )
    tfidf        = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['description'])
    cosine_sim   = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices      = pd.Series(df.index, index=df['product_id'])
    return tfidf, tfidf_matrix, cosine_sim, indices

@st.cache_data
def build_user_sim(user_item_df):
    utility = user_item_df.pivot_table(index='user', columns='product_id', values='rating').fillna(0)
    sim     = cosine_similarity(utility)
    sim_df  = pd.DataFrame(sim, index=utility.index, columns=utility.index)
    return utility, sim_df

@st.cache_data
def build_item_sim(user_item_df):
    utility = user_item_df.pivot_table(index='product_id', columns='user', values='rating').fillna(0)
    sim     = cosine_similarity(utility)
    sim_df  = pd.DataFrame(sim, index=utility.index, columns=utility.index)
    return sim_df

tfidf, tfidf_matrix, cosine_sim, indices = build_tfidf(products_df)
utility_matrix, user_sim_df = build_user_sim(user_item_df)
item_sim_df = build_item_sim(user_item_df)

all_users    = sorted(user_sim_df.index.tolist())
all_products = products_df['product_id'].tolist()


# ─────────────────────────────────────────
# HELPER — RENDER PRODUCT CARDS
# ─────────────────────────────────────────
def render_cards_hybrid(result_df, reason_col=None):
    cols = st.columns(5)
    for i, (_, row) in enumerate(result_df.head(5).iterrows()):
        display  = str(row.get('display_name', ''))
        brand    = str(row.get('brand', '')).upper()
        occ      = str(row.get('occasion', '')).title()
        price    = int(row['price']) if pd.notna(row.get('price')) else 'N/A'
        rating   = row.get('rating', '')
        stars    = "★" * int(rating) + "☆" * (5 - int(rating)) if rating else ""
        reason_line = ""
        if reason_col and reason_col in row.index:
            reason_line = f"<div class='reason'>{row[reason_col]}</div>"
        card_html = f"""
        <div class='card card-{i}'>
            <div class='brand'>{display}</div>
            <div class='type'>{brand}</div>
            <div class='meta'>{occ}</div>
            <div class='price'>Rs. {price}</div>
            <div class='rating'>{stars} &nbsp; {rating} / 5</div>
            {reason_line}
        </div>
        """
        with cols[i]:
            st.markdown(card_html, unsafe_allow_html=True)

def render_cards(result_df, reason_col=None, reason_label=""):
    cols = st.columns(5)
    for i, (_, row) in enumerate(result_df.head(5).iterrows()):
        brand   = str(row.get('brand', '')).upper()
        subcat  = str(row.get('sub_category', '')).title()
        color   = str(row.get('color', '')).title()
        occ     = str(row.get('occasion', '')).title()
        price   = int(row['price']) if pd.notna(row.get('price')) else 'N/A'
        rating  = row.get('rating', '')
        stars   = "★" * int(rating) + "☆" * (5 - int(rating)) if rating else ""

        reason_line = ""
        if reason_col and reason_col in row:
            val = row[reason_col]
            reason_line = f"<div class='reason'>{reason_label} {val}</div>"

        card_html = f"""
        <div class='card card-{i}'>
            <div class='brand'>{brand}</div>
            <div class='type'>{subcat}</div>
            <div class='meta'>{color} &nbsp;&middot;&nbsp; {occ}</div>
            <div class='price'>Rs. {price}</div>
            <div class='rating'>{stars} &nbsp; {rating} / 5</div>
            {reason_line}
        </div>
        """
        with cols[i]:
            st.markdown(card_html, unsafe_allow_html=True)


# ─────────────────────────────────────────
# ENGINE DROPDOWN
# ─────────────────────────────────────────
engine = st.selectbox(
    "Select Recommendation Engine",
    [
        "Content-Based Recommendation",
        "Collaborative — User to User Filtering",
        "Collaborative — Item to Item Filtering",
        "Hybrid Recommendation"
    ]
)
st.markdown("---")


# ══════════════════════════════════════════
#  1 — CONTENT-BASED
# ══════════════════════════════════════════
if engine == "Content-Based Recommendation":

    st.markdown("<div class='rec-title'>Content-Based Recommendation</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:white; font-size:13px;'>Type what you are looking for — brand, color, occasion, type. TF-IDF converts your query into a vector and finds the most similar products.</p>", unsafe_allow_html=True)

    query = st.text_input("What are you looking for?", placeholder="e.g. casual black shirt")

    if st.button("Get Recommendations") and query:
        # Transform user query into TF-IDF vector
        query_vec   = tfidf.transform([query])
        # Compute similarity between query and all products
        sim_scores  = linear_kernel(query_vec, tfidf_matrix).flatten()
        # Get top 5 indices
        top_indices = sim_scores.argsort()[-5:][::-1]

        result = products_df.iloc[top_indices].copy().reset_index(drop=True)
        result['similarity'] = [f"{int(round(float(sim_scores[i]), 2) * 100)}% match" for i in top_indices]

        st.markdown(f"<div class='context-bar'>Showing results for: <b>{query}</b></div>", unsafe_allow_html=True)
        render_cards(result, reason_col='similarity', reason_label="Match score:")


# ══════════════════════════════════════════
#  2 — COLLABORATIVE: USER-USER
# ══════════════════════════════════════════
elif engine == "Collaborative — User to User Filtering":

    st.markdown("<div class='rec-title'>Collaborative Recommendation — User to User</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:white; font-size:13px;'>Select a user and a category. The model automatically identifies the most similar shopper using cosine similarity on ratings, then recommends products from that category that the similar shopper loved but you have not seen yet.</p>", unsafe_allow_html=True)

    all_categories = sorted(products_df['sub_category'].dropna().str.title().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        target_user = st.selectbox("Select User:", all_users, index=0)
    with col2:
        selected_cat = st.selectbox("Select Category:", all_categories, index=0)

    if st.button("Get Recommendations"):
        # Automatically find the most similar user
        similar_scores = user_sim_df.loc[target_user].drop(target_user).sort_values(ascending=False)
        best_user      = similar_scores.index[0]
        best_score     = similar_scores.iloc[0]

        # User banner
        st.markdown(f"""
        <div class='user-banner'>
            <div class='you'>You are</div>
            <div class='name'>{target_user}</div>
            <div class='sim'>
                MOST SIMILAR USER: <span class='peer'>{best_user}</span>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                SIMILARITY SCORE: <span class='peer'>{int(best_score * 100)}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Products best_user rated highly that target_user has not seen, filtered by category
        rated_by_target = set(user_item_df[user_item_df['user'] == target_user]['product_id'])
        peer_ratings    = user_item_df[user_item_df['user'] == best_user].copy()
        unseen          = peer_ratings[~peer_ratings['product_id'].isin(rated_by_target)]

        # Merge to get product details and filter by selected category
        unseen = unseen.merge(
            products_df[['product_id', 'brand', 'sub_category', 'color', 'occasion', 'price', 'rating']],
            on='product_id', how='left', suffixes=('_peer', '')
        )
        unseen = unseen[unseen['sub_category'].str.title() == selected_cat]
        unseen = unseen.sort_values('rating', ascending=False).reset_index(drop=True)

        # If similar user doesn't have enough, fill remaining spots from top-rated
        # products in that category that target_user hasn't seen
        if len(unseen) < 5:
            already_shown = set(unseen['product_id'].tolist())
            fallback = products_df[
                (products_df['sub_category'].str.title() == selected_cat) &
                (~products_df['product_id'].isin(rated_by_target)) &
                (~products_df['product_id'].isin(already_shown))
            ].sort_values('rating', ascending=False)
            fallback = fallback.copy()
            fallback['peer_source'] = 'top_rated'
            needed = 5 - len(unseen)
            unseen = pd.concat([unseen, fallback.head(needed)], ignore_index=True)

        unseen = unseen.head(5).reset_index(drop=True)

        if unseen.empty:
            st.warning(f"No {selected_cat} products found. Try a different category.")
        else:
            peer_rating_map = unseen.set_index('product_id')['rating'].to_dict()
            unseen['peer_rated'] = unseen.apply(
                lambda row: f"{best_user} rated this {peer_rating_map.get(row['product_id'], '')} / 5"
                if row.get('peer_source') != 'top_rated'
                else f"Top rated {selected_cat}", axis=1
            )
            st.markdown(f"Recommended <b>{selected_cat}s</b> liked by similar users:", unsafe_allow_html=True)
            render_cards(unseen, reason_col='peer_rated', reason_label="")


# ══════════════════════════════════════════
#  3 — COLLABORATIVE: ITEM-ITEM
# ══════════════════════════════════════════
elif engine == "Collaborative — Item to Item Filtering":

    st.markdown("<div class='rec-title'>Collaborative Recommendation — Item to Item</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:white; font-size:13px;'>Select a product. The model finds items rated similarly by the same users — products that share the same fanbase are considered similar.</p>", unsafe_allow_html=True)

    # Build display names: Brand + Color + Type
    product_display = products_df.apply(
        lambda r: f"{str(r['brand']).upper()} {str(r['sub_category']).title()}", axis=1
    )
    product_display_map = dict(zip(product_display, products_df['product_id']))
    selected_display = st.selectbox("Select the product you like:", list(product_display_map.keys()), index=0)
    selected_item    = product_display_map[selected_display]

    if st.button("Show Similar Products"):
        sel_info = products_df[products_df['product_id'] == selected_item].iloc[0]
        sel_name = f"{str(sel_info['brand']).upper()} {str(sel_info['sub_category']).title()}"
        sel_color = str(sel_info['color']).title()

        # Stylish "you viewed" banner
        st.markdown(f"""
        <div class='viewed-banner'>
            <span class='viewed-label'>You viewed</span>
            <span class='viewed-product'>{sel_name}</span>
            <span class='viewed-meta'>{sel_color} &nbsp;·&nbsp; {str(sel_info['occasion']).title()}</span>
        </div>
        """, unsafe_allow_html=True)

        if selected_item in item_sim_df.index:
            similar = (
                item_sim_df.loc[selected_item]
                .drop(selected_item)
                .sort_values(ascending=False)
                .head(5)
            )
            result = products_df[products_df['product_id'].isin(similar.index)].copy()
            result['item_similarity'] = result['product_id'].map(similar.to_dict())
            result = result.sort_values('item_similarity', ascending=False).reset_index(drop=True)
            # Convert similarity to a match percentage feel
            result['sim_label'] = result['item_similarity'].apply(
                lambda x: f"{int(round(x, 2) * 100)}% match"
            )

            st.markdown("Users who liked this also liked:", unsafe_allow_html=True)
            render_cards(result, reason_col='sim_label', reason_label="")
        else:
            st.warning("Product not found in collaborative matrix.")


# ══════════════════════════════════════════
#  4 — HYBRID
# ══════════════════════════════════════════
elif engine == "Hybrid Recommendation":

    st.markdown("<div class='rec-title'>Hybrid Recommendation</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:white; font-size:13px;'>Select a user and a product they like. Content-Based finds similar products by description. Item-Item Collaborative finds products liked by the same users. Hybrid combines both scores 50/50.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        h_user = st.selectbox("Select a User:", all_users)
    with col2:
        # Build display: "Sneakers (Green)" format
        hybrid_display = products_df.apply(
            lambda r: f"{str(r['sub_category']).title()} ({str(r['color']).title()}) — {str(r['brand']).upper()}", axis=1
        )
        hybrid_display_map = dict(zip(hybrid_display, products_df['product_id']))
        h_selected_display = st.selectbox("Select a product they like:", list(hybrid_display_map.keys()))
        h_product = hybrid_display_map[h_selected_display]

    if st.button("Generate Recommendations"):
        i_info    = products_df[products_df['product_id'] == h_product].iloc[0]
        prod_name = h_selected_display

        # Content-based scores (TF-IDF)
        idx       = int(indices[h_product])
        cb_scores = dict(enumerate(cosine_sim[idx]))

        # Item-item collaborative scores
        if h_product in item_sim_df.index:
            cf_series = item_sim_df.loc[h_product].drop(h_product)
        else:
            cf_series = pd.Series(dtype=float)

        pid_to_idx    = pd.Series(products_df.index, index=products_df['product_id'])
        rows = []
        for pid in products_df['product_id']:
            if pid == h_product:
                continue
            i      = pid_to_idx[pid]
            cb     = cb_scores.get(i, 0)
            cf     = cf_series.get(pid, 0) if pid in cf_series.index else 0
            hybrid = round(0.5 * cb + 0.5 * cf, 4)
            rows.append({'product_id': pid, 'hybrid_score': hybrid,
                         'content_score': round(cb, 3), 'collab_score': round(cf, 3)})

        hybrid_df = pd.DataFrame(rows).sort_values('hybrid_score', ascending=False).head(5)
        result    = hybrid_df.merge(
            products_df[['product_id', 'brand', 'sub_category', 'color', 'occasion', 'price', 'rating']],
            on='product_id', how='left'
        ).reset_index(drop=True)
        result['score_label']    = result['hybrid_score'].apply(lambda x: f"{int(round(x, 2) * 100)}% match")
        # Display name: Category (Color) instead of product_id
        result['display_name']   = result.apply(
            lambda r: f"{str(r['sub_category']).title()} ({str(r['color']).title()}) — {str(r['brand']).upper()}", axis=1
        )

        st.markdown(f"<div class='context-bar'>PERSONALIZED FOR  <b>{h_user}</b> — BASED ON THE INTEREST IN <b>{prod_name}</b></div>", unsafe_allow_html=True)
        render_cards_hybrid(result, reason_col='score_label')


# ─────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center; color:#bbb; font-size:12px;'>Myntra Shopping Dataset — Recommendation Engine Project</p>", unsafe_allow_html=True)
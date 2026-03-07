🛍️ Myntra Recommendation Engine

A Streamlit-based recommendation system that suggests fashion products using multiple recommendation techniques.

🚀 Features

1. Content-Based Filtering (TF-IDF + Cosine Similarity)

2. User–User Collaborative Filtering

3. Item–Item Collaborative Filtering

4 . Hybrid Recommendation Model (50% Content + 50% Collaborative)

MAIN PAGE
![imge alt](https://github.com/sakshiparadkar/Myntra-Recommendation-Engine-/blob/3a782ab577540d3aed8e0773e3c43a23c985760b/1rec.png)

CONTENT-BASED FILTERING
![imge alt](https://github.com/sakshiparadkar/Myntra-Recommendation-Engine-/blob/3a782ab577540d3aed8e0773e3c43a23c985760b/2rec.png)

COLLABORATIVE USER-TO USER FILTERING
![imge alt](https://github.com/sakshiparadkar/Myntra-Recommendation-Engine-/blob/3a782ab577540d3aed8e0773e3c43a23c985760b/3rec.png)

COLLABORATIVE ITEM TO ITEM FILTERING
![image_alt](https://github.com/sakshiparadkar/Myntra-Recommendation-Engine-/blob/215deb0c767385fe5982700a58cb6e413c8c4ca6/4recc.png)

HYBRID FILTERING
![imge alt](https://github.com/sakshiparadkar/Myntra-Recommendation-Engine-/blob/3a782ab577540d3aed8e0773e3c43a23c985760b/5rec.png)

🧠 Tech Stack

Python | Pandas & NumPy | Scikit-learn | Streamlit


📊 How It Works

1.Converts product features into TF-IDF vectors

2.Uses cosine similarity to measure similarity

3.Builds utility matrix for collaborative filtering

4.Combines scores in hybrid model for better accuracy

5.Displays TOP 5 personalized recommendations

▶️ Run Locally
pip install -r requirements.txt
streamlit run app.py

📌 Project Type

Machine Learning | Recommendation System | E-commerce Personalization

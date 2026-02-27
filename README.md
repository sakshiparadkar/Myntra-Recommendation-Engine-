🛍️ Myntra Recommendation Engine

A Streamlit-based recommendation system that suggests fashion products using multiple recommendation techniques.

🚀 Features

1. Content-Based Filtering (TF-IDF + Cosine Similarity)

2. User–User Collaborative Filtering

3. Item–Item Collaborative Filtering

4 . Hybrid Recommendation Model (50% Content + 50% Collaborative)
![imge alt](https://github.com/sakshiparadkar/Myntra-Recommendation-Engine-/blob/3a782ab577540d3aed8e0773e3c43a23c985760b/1rec.png)
![imge alt](https://github.com/sakshiparadkar/Myntra-Recommendation-Engine-/blob/3a782ab577540d3aed8e0773e3c43a23c985760b/2rec.png)
![imge alt](https://github.com/sakshiparadkar/Myntra-Recommendation-Engine-/blob/3a782ab577540d3aed8e0773e3c43a23c985760b/3rec.png)
![imge alt](https://github.com/sakshiparadkar/Myntra-Recommendation-Engine-/blob/3a782ab577540d3aed8e0773e3c43a23c985760b/4rec.png)
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

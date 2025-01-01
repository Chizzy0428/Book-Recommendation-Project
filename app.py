import pickle
import streamlit as st
import numpy as np

st.header('Book Recommender System Using Machine learning')
model = pickle.load(open('artifacts/model.pkl', 'rb'))
books_name = pickle.load(open('artifacts/book_name.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
books_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))
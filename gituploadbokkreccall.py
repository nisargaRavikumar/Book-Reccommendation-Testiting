import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
lmodel = keras.models.load_model("my_model")
def recommend(user_id):
    books = pd.read_csv('books_cleaned.csv')
    ratings = pd.read_csv('ratings.csv')
  
    book_id = list(ratings.book_id.unique()) #grabbing all the unique books
  
    book_arr = np.array(book_id) #geting all book IDs and storing them in the form of an array
    user_arr = np.array([user_id for i in range(len(book_id))])
    prediction = lmodel.predict([book_arr, user_arr])
  
    prediction = prediction.reshape(-1) #reshape to single dimension
    prediction_ids = np.argsort(-prediction)[0:5]

    recommended_books = pd.DataFrame(books.iloc[prediction_ids], columns = ['book_id', 'isbn', 'authors', 'title', 'average_rating' ])
    print('Top 5 recommended books for you: \n')
    return recommended_books
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Books Recommendations for User</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    user = st.number_input('Ennter a user ID')
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        st.dataframe(recommend(user))
     
if __name__=='__main__': 
    main()






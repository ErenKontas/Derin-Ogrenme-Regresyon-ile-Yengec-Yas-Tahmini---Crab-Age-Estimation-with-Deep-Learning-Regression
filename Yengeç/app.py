import pandas as pd
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Load data and update column names
df = pd.read_csv('train.csv')
df.columns = df.columns.str.replace(r'[\s\.]', '_', regex=True)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Length', 'Diameter', 'Height', 'Weight', 'Shucked_Weight', 'Viscera_Weight', 'Shell_Weight']),
        ('cat', OneHotEncoder(), ['Sex']) 
    ]
)

# Streamlit application
def yas_pred(Length, Diameter, Height, Weight, Shucked_Weight, Viscera_Weight, Shell_Weight, Sex):
    input_data = pd.DataFrame({
        'Sex': [Sex],
        'Length': [Length],
        'Diameter': [Diameter],
        'Height': [Height],
        'Weight': [Weight],
        'Shucked_Weight': [Shucked_Weight],
        'Viscera_Weight': [Viscera_Weight],
        'Shell_Weight': [Shell_Weight]
    })
    
    input_data_transformed = preprocessor.fit_transform(input_data)

    model = joblib.load('Yengeç_Yaş.pkl')

    prediction = model.predict(input_data_transformed)
    return float(prediction[0])

# Streamlit interface
def main():
    st.title("Age Prediction Model")
    st.write("Enter Input Data")
    
    Sex = st.selectbox('Sex', options=['M', 'F'])
    Length = st.slider('Length', float(df['Length'].min()), float(df['Length'].max()))
    Diameter = st.slider('Diameter', float(df['Diameter'].min()), float(df['Diameter'].max()))
    Height = st.slider('Height', float(df['Height'].min()), float(df['Height'].max()))
    Weight = st.slider('Weight', float(df['Weight'].min()), float(df['Weight'].max()))
    Shucked_Weight = st.slider('Shucked Weight', float(df['Shucked_Weight'].min()), float(df['Shucked_Weight'].max()))
    Viscera_Weight = st.slider('Viscera Weight', float(df['Viscera_Weight'].min()), float(df['Viscera_Weight'].max()))
    Shell_Weight = st.slider('Shell Weight', float(df['Shell_Weight'].min()), float(df['Shell_Weight'].max()))
    
    if st.button('Predict'):
        yas = yas_pred(Length, Diameter, Height, Weight, Shucked_Weight, Viscera_Weight, Shell_Weight, Sex)
        st.write(f'The predicted age is: {yas:.2f}')

if __name__ == '__main__':
    main()

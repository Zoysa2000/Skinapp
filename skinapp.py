import torch
from PIL import Image
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
from copy import deepcopy
from torchvision import transforms
import base64
from io import BytesIO
import cv2
import asyncio

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Load your pre-trained model
best_model = torch.load("./Model/model.pth")
best_model.eval()
image = './Logo/logo.png'
# Define a function to preprocess the image for inference
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)
    return image

# Define index_label globally
index_label = {0: "Dry Skin", 1: "Normal Skin", 2: "Oily Skin", 3: "High Oily Skin"}

def predict_skin_type_and_oiliness_level(image_path):
    img = Image.open(image_path).convert("RGB")
    original_img = deepcopy(img)
    img = preprocess_image(img)  # Corrected line
    img = img.view(1, 3, 224, 224)
    best_model.eval()
    with torch.no_grad():
        if torch.cuda.is_available():
            img = img.cuda()
        
        out = best_model(img)
        skin_type = index_label[out.argmax(1).item()]
        oiliness_level = get_oiliness_level(out.cpu().numpy())
        
        return original_img, skin_type, oiliness_level

def get_oiliness_level(predictions):
    oiliness_index = np.argmax(predictions)
    # Assuming the predictions are in the order of dry, normal, oily
    if oiliness_index == 0:
        return 1  # Dry or very low
    elif oiliness_index == 1:
        return 2  # Low
    elif oiliness_index == 2:
        return 3  # Medium
    else:
        return 4  # Very high


# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Skin detection", page_icon=":tdata", layout="wide")

    # Apply custom CSS to set background color to white
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color:#000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.container():
        left_column,middle_column, right_column = st.columns([1,2,4])
    with left_column:
     st.markdown(
        "<h1 style='color:#EFE8A2;'></h1>",
        unsafe_allow_html=True
    )  
    with middle_column:
     image = './Logo/logo.png'
     st.image(image,  use_column_width=True)
         
    with right_column:
     st.markdown(
    "<h1 class='shinespy-heading'>ShineSpy</h1>",
    unsafe_allow_html=True
     )
     st.markdown(
    "<p style='color:White;font-size:20px'>Discover the perfect skincare for you!</p>",
    unsafe_allow_html=True
     )
     st.markdown(
    "<p style='color:White;font-size:15px'>SkinSpy is an innovative application designed to analyze and address the oilyness level of the human face, categorizing it into oily, normal, or dry skin types. Leveraging machine learning (ML) and image processing techniques, SkinSpy provides personalized treatment recommendations tailored to individual skin conditions.</p>",
    unsafe_allow_html=True
     )
     st.button("Discover More", type="primary")

     st.markdown(
    """
    <style>
    @media only screen and (max-width: 768px) {
        .shinespy-heading {
           color:#EFE8A2;
           text-align: center;
        }
    }
        .shinespy-heading {
           color:#EFE8A2;
           margin-top:60px;
        }
        .stButton>button {
        background-color: #EFE8A2; 
        color: black; 
        border-color: #EFE8A2; 
    }
       .stButton>button:hover {
        background-color: #EFE8A2; 
        color: black;
        border-color:#EFE8A2; 
    }

    </style>
    """,
    unsafe_allow_html=True
     )
    st.markdown(
        "<h2 style='color:#EFE8A2;'>✌️ Check Your Oiliness</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align: justify;color:#ffff'>🌟Click the button below to use the webcam or upload your face image and get efficient skin type results. Our application utilizes advanced image processing to analyze facial features accurately. It swiftly detects skin types—dry, normal, oily, or highly oily—providing users with valuable insights into their skincare needs.</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h5 style='color:#EFE8A2'> 📷 Upload your Image </h5>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        # Predict skin type and oiliness level
        original_img, skin_type, oiliness_level = predict_skin_type_and_oiliness_level(uploaded_file)
        # Display the original image with prediction results
        st.markdown("<h5 style='color:#EFE8A2;'><span>🔥 Predicted Skin Type</span> </h5>", unsafe_allow_html=True)

        with st.container():
            left_column,middle_column, right_column = st.columns([2.5,0.2,5.5])
        with left_column:
         st.markdown(
        f'<div style="display: flex; justify-content:left;">'
        f'<img src="data:image/png;base64,{image_to_base64(original_img)}" style="width:100%;height:350px;">'
        f'</div>',
        unsafe_allow_html=True
       )
        with middle_column:
         st.markdown(
           "<p></p>",
           unsafe_allow_html=True
           )
        with right_column:
        # Check if the skin type is oily and provide treatment advice
         
         if skin_type == "Oily Skin":
           st.markdown("<h2 style='color:#EFE8A2;'><span></span> {} <span><br> Oiliness Level = Level</span> {}</h2>".format('<span style="color:#EFE8A2;">{}</span>'.format(skin_type), '<span style="color:#EFE8A2;">{}</span>'.format(oiliness_level)), unsafe_allow_html=True)
           st.markdown("<h5 style='color:#EFE8A2'><span>💧 Treatment for your {}</span></h5>".format('<span style="color:#EFE8A2;">{}</span>'.format(skin_type)), unsafe_allow_html=True)
           st.markdown(
           "<p style='text-align: justify;color:#ffff'>🌟For individuals grappling with oily skin, establishing an effective skincare regimen is paramount to manage excess oil production and foster a balanced complexion. It all starts with a diligent cleansing routine employing a gentle yet potent foaming cleanser to effectively eliminate impurities and excess oil without compromising the skin's natural moisture barrier. Follow this up with a non-alcoholic toner infused with ingredients such as witch hazel or tea tree oil, which helps rebalance the skin's pH levels and tighten pores, thus reducing the appearance of oiliness. Integration of lightweight, oil-free moisturizers formulated with hydrating agents like hyaluronic acid or glycerin is crucial to provide adequate hydration without exacerbating shine. Regular exfoliation using products containing salicylic acid or glycolic acid aids in unclogging pores, removing dead skin cells, and minimizing the appearance of enlarged pores.</p>",
           unsafe_allow_html=True
           )
         if skin_type=="Dry Skin":
           st.markdown("<h2 style='color:#EFE8A2;'><span></span> {} <span><br> Oiliness Level = Level</span> {}</h2>".format('<span style="color:#EFE8A2;">{}</span>'.format(skin_type), '<span style="color:#EFE8A2;">{}</span>'.format(oiliness_level)), unsafe_allow_html=True)
           st.markdown("<h5 style='color:#EFE8A2'><span>💧 Treatment for your {}</span></h5>".format('<span style="color:#EFE8A2;">{}</span>'.format(skin_type)), unsafe_allow_html=True)
           st.markdown(
           "<p style='text-align: justify;color:#ffff'>🌟Dry skin requires a meticulous and nurturing skincare regimen to combat its challenges and restore a healthy, radiant complexion. Initiate your routine with a gentle, creamy cleanser, adept at purifying the skin without depleting its natural oils, thus laying a foundation of hydration and vitality. Follow this with a hydrating toner enriched with ingredients like hyaluronic acid or rose water, infusing your skin with much-needed moisture and soothing relief. Next, embrace the indulgence of a rich, emollient moisturizer, fortified with shea butter, ceramides, or squalane, to deeply hydrate and fortify the skin barrier, shielding against environmental aggressors. Regular exfoliation is key, employing either a gentle scrub or chemical exfoliant to slough off dead skin cells and stimulate cell turnover, unveiling a smoother, more luminous complexion. Consider integrating a hydrating serum or facial oil for an added moisture boost, replenishing and nourishing the skin from within.</p>",
           unsafe_allow_html=True
           )
         if skin_type=="Normal Skin":
           st.markdown("<h2 style='color:#EFE8A2;'><span></span> {} <span><br> Oiliness Level = Level</span> {}</h2>".format('<span style="color:#EFE8A2;">{}</span>'.format(skin_type), '<span style="color:#EFE8A2;">{}</span>'.format(oiliness_level)), unsafe_allow_html=True)
           st.markdown("<h5 style='color:#EFE8A2'><span>💧 Treatment for your {}</span></h5>".format('<span style="color:#EFE8A2;">{}</span>'.format(skin_type)), unsafe_allow_html=True)
           st.markdown(
           "<p style='text-align: justify;color:#ffff'>🌟Congratulations 🎉on possessing normal skin, a blessing that requires minimal specialized care, affording you the luxury of a naturally radiant and healthy complexion. Normal skin is characterized by its exquisite balance, boasting few imperfections, a smooth texture, and neither an excess of oiliness nor dryness. Maintaining its vitality is straightforward with a streamlined skincare regimen. Commence your routine with a gentle yet thorough cleansing to rid the skin of impurities, followed by nourishing moisturization to preserve its suppleness. Daily safeguarding against UV radiation with sunscreen is paramount to shield your skin from damage and preserve its innate allure.Furthermore, integrate supplementary skincare rituals like weekly exfoliation to invigorate cellular turnover and occasional mask treatments to elevate hydration levels and provide additional nourishment. Embrace the harmonious equilibrium of your skin and revel in its innate splendor.</p>",
           unsafe_allow_html=True
           )

    st.markdown(
    "<h5 style='color:#EFE8A2; margin-top:30px'> 📸 Open your Webcamera</h5>",
    unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: justify;color:#ffff'>🌟Can click the web camera to initiate live skin analysis. Our algorithm assesses oily skin levels in real-time, providing instant feedback. With just a click, users gain valuable insights into their skin condition, enabling them to make informed skincare decisions. This feature offers convenience and accessibility, empowering users to monitor their skin health effortlessly.</p>",
        unsafe_allow_html=True
    )
      
if __name__ == "__main__":
    main()



import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="AI Mental Health System", layout="wide")

# ---------------- LOGIN PAGE ----------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.title("🔐 Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.login = True
            st.success("Login Successful")
        else:
            st.error("Invalid Credentials")

    st.stop()

st.markdown("""
<style>
body {
    background-color: #0E1117;
    color: white;
}
.big-title {
    text-align: center;
    font-size: 42px;
    color: #00C9A7;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="big-title">🧠 AI Mental Health Analyzer</p>', unsafe_allow_html=True)
st.write("Analyze your emotions using AI")

st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Dashboard")
page = st.sidebar.radio("Navigate", ["Predict", "Analytics", "Chatbot"])

if page == "Predict":
    text = st.text_area("💬 Enter your thoughts:")

    if st.button("Analyze"):
        if text.strip() == "":
            st.warning("Enter text first")
        else:
            vec = vectorizer.transform([text])
            result = model.predict(vec)

            if result[0] == 1:
                st.error("⚠️ Negative Mental State")
                st.progress(80)
            else:
                st.success("✅ Positive Mental State")
                st.progress(30)

elif page == "Analytics":
    st.subheader("📊 Model Performance")

    # Dummy accuracy (you can update later)
    accuracy = 85

    st.metric("Accuracy", f"{accuracy}%")

    # Chart
    labels = ['Positive', 'Negative']
    values = [60, 40]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Prediction Distribution")

    st.pyplot(fig)

# ---------------- CHATBOT PAGE ----------------
elif page == "Chatbot":
    st.subheader("🤖 Mental Health Chatbot")

    user_msg = st.text_input("You:")

    if st.button("Send"):
        if "sad" in user_msg.lower():
            st.write("Bot: I'm here for you. You are not alone ❤️")
        elif "happy" in user_msg.lower():
            st.write("Bot: That's great to hear! 😊")
        else:
            st.write("Bot: Tell me more about how you're feeling.")

# Footer
st.markdown("---")
st.caption("🚀 NLP + ML + Streamlit Advanced Project")
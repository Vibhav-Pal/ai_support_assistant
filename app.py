import streamlit as st
from classifier import classify_intent
from summarizer import summarize_text
from recommender import recommend

st.set_page_config(page_title="AI Support Assistant", layout="centered")
st.title("🤖 AI Support Assistant")
st.write("Paste a customer query or issue below.")

user_input = st.text_area("Your query:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            intent, confidence = classify_intent(user_input)
            st.subheader("🎯 Detected Intent")
            st.write(f"**{intent}** (confidence: {confidence:.0%})")

            if len(user_input.split()) > 30:
                st.subheader("📝 Summary")
                st.write(summarize_text(user_input))

            st.subheader("💡 Related Help Articles")
            recs = recommend(user_input)
            for _, row in recs.iterrows():
                with st.expander(row["question"]):
                    st.write(row["answer"])
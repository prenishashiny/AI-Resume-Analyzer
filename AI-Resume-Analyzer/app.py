import streamlit as st
from pypdf import PdfReader
from skills import skills
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader(
    "Upload Your Resume",
    type="pdf"
)

job_description = st.text_area(
    "Paste Job Description Here"
)

if uploaded_file:

    st.success("PDF Uploaded Successfully!")

    pdf = PdfReader(uploaded_file)

    resume_text = ""

    for page in pdf.pages:
        text = page.extract_text()

        if text:
            resume_text += text

    st.write("Characters extracted:", len(resume_text))

    found_skills = []

    for skill in skills:
        if skill.lower() in resume_text.lower():
            found_skills.append(skill)

    st.subheader("🎯 Detected Skills")

    if found_skills:
        for skill in found_skills:
            st.success(skill)
    else:
        st.error("No skills detected.")

    ats_score = (len(found_skills) / len(skills)) * 100

    st.subheader("📊 ATS Score")

    st.metric(
        "Resume ATS Score",
        f"{ats_score:.2f}%"
    )

    st.progress(int(ats_score))

    if job_description:

        cv = CountVectorizer()

        matrix = cv.fit_transform(
            [resume_text, job_description]
        )

        similarity = cosine_similarity(matrix)[0][1]

        st.subheader("💼 Job Match Score")

        st.success(
            f"{similarity * 100:.2f}% Match"
        )

        missing_skills = []

        for skill in skills:

            if skill.lower() in job_description.lower():

                if skill.lower() not in resume_text.lower():

                    missing_skills.append(skill)

        st.subheader("❌ Missing Skills")

        if missing_skills:
            st.warning(missing_skills)
        else:
            st.success("No Missing Skills Found")

        st.subheader("💡 Resume Improvement Suggestions")

        if ats_score < 40:
            st.warning(
                "Add more technical skills, projects, and certifications to improve ATS score."
            )

        if missing_skills:
            st.info(
                f"Consider adding these skills: {', '.join(missing_skills)}"
            )

        if ats_score >= 40:
            st.success(
                "Your resume has a good ATS score."
            )

    st.subheader("📄 Extracted Resume Text")

    st.write(resume_text)
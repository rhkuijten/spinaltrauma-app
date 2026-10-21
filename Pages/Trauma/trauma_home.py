import streamlit as st
from styles import apply_custom_css

apply_custom_css()


def trauma_home_page():
    # Section Title
    st.markdown(
        "<h1 class='centered-header'>Survival Prediction for Patients with Spinal Trauma</h1>",
        unsafe_allow_html=True,
    )

    # Introduction Section
    st.write(
        """
        This webpage provides an interactive interface to explore our predictive model for survival
        following spinal trauma. The model combines patient demographics, comorbidity and frailty,
        injury severity, fracture morphology, and neurological status to estimate an individual
        probability of **30-day mortality**.

        The key components of this tool include:

        - **Decision Tool:** Predict the probability of 30-day mortality after spinal trauma.
        - **Evaluation:** Presentation of performance metrics, predictor importances, and case explanations to help with interpretation.
        - **Links:** Access to the manuscript and supplements, as well as the source code on GitHub.
        """
    )
    st.write("")

    st.markdown("<h2 class='section-header'>Predictors</h2>", unsafe_allow_html=True)
    st.write(
        """
        - **Patient characteristics:** age, sex, BMI, ASA classification, mFI-5 frailty index.
        - **Injury:** injury mechanism, Injury Severity Score (ISS-08), number of fractures, traumatic brain injury on CT.
        - **Spinal injury:** AO Spine fracture types (A, B, C, F) and spinal region involvement (cervical, thoracic, lumbar, sacral). More than one type or region can be present.
        - **Neurology:** ASIA impairment scale, including the option that the examination could not be performed.
        """
    )
    st.write("")

    # Footer or additional notes
    st.write(
        """
        For more information about the model, contact the research team at **[r.h.kuijten-2@umcutrecht.nl](mailto:r.h.kuijten-2@umcutrecht.nl)**.

        We welcome feedback to improve the model and its interface. Additionally, we welcome collaboration opportunities to help validate and refine the model for future research and clinical applications.
        """
    )
    st.write("")

    # Disclaimer Section
    st.markdown("<h2 class='section-header'>Disclaimer</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='disclaimer'>
            This model is currently under review and has not been published or validated as a medical device.
            It is intended for **research purposes only** and should not be used for clinical decision-making or as a substitute for professional medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

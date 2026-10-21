"""Evaluation page for the spinal trauma survival model.

SCAFFOLD: the explanatory text below is model-agnostic and ready to go, but the
figures still need to be exported from the analysis and dropped into
Images/Trauma/ under the file names referenced here. Any figure that is still
missing renders as a labelled placeholder instead of breaking the page.

Expected files:
    Images/Trauma/ROC Curve.png
    Images/Trauma/Calibration plot.png
    Images/Trauma/DCA.png
    Images/Trauma/SHAP Feature importances.png
    Images/Trauma/SHAP Feature Beeplot.png
    Images/Trauma/Case A.png
    Images/Trauma/Case B.png
"""

import streamlit as st
from styles import apply_custom_css, show_image_or_placeholder
from feedback import feedback_tool

apply_custom_css()

IMAGE_DIR = "Images/Trauma"


def eval_page():

    ### Model Performance ###
    st.markdown(
        "<h1 class='centered-header'>Model Performance Evaluation</h1>", unsafe_allow_html=True
    )

    with st.container():
        # Discrimination & Calibration
        col1, col2 = st.columns([1, 1])
        with col1:
            show_image_or_placeholder(f"{IMAGE_DIR}/ROC Curve.png", "ROC curve")
            with st.expander("More information on discrimination"):
                st.write("""
                            Discrimination refers to the ability of a prediction model to distinguish between those who will experience an event (e.g., mortality, disease) and those who will not. A well-discriminating model assigns higher predicted probabilities to individuals who experience the event compared to those who do not.

                            **How to read the discrimination figure**
                            The most common discrimination figure is the Receiver Operating Characteristic (ROC) curve. This plot shows the trade-off between sensitivity (true positive rate) and 1-specificity (false positive rate) across different prediction thresholds. The area under the ROC curve (AUC) quantifies model discrimination:

                            - **AUC 0.90-1.00:** Excellent.
                            - **AUC 0.80-0.89:** Good.
                            - **AUC 0.70-0.79:** Fair.
                            - **AUC 0.51-0.69:** Poor.
                            - **AUC 0.50:** No better than flipping a coin.
                """)

        with col2:
            show_image_or_placeholder(f"{IMAGE_DIR}/Calibration plot.png", "Calibration plot")
            with st.expander("More information on calibration"):
                st.write("""
                            Calibration refers to how well the predicted probabilities from a model match the actual outcomes. A well-calibrated model assigns predictions that accurately reflect the observed event rates. For example, if a model predicts a 20% probability of an event, that event should occur approximately 20% of the time in similar cases.

                            **How to read the calibration plot**
                            A calibration plot compares predicted probabilities to observed event rates. Key features of the plot include:

                            - The red **diagonal line** represents perfect calibration, where predicted and observed probabilities are equal.
                            - The black **calibration curve** shows the relationship between predicted and observed values. Deviations from the diagonal indicate miscalibration.
                            - A model is **overestimating risk** if the curve is below the diagonal, meaning the predicted probabilities are higher than the actual event rates.
                            - A model is **underestimating risk** if the curve is above the diagonal, meaning predicted probabilities are too low.
                """)

        # Decision Curve Analysis
        col3, col4, col5 = st.columns([1, 2, 1])
        with col4:
            show_image_or_placeholder(f"{IMAGE_DIR}/DCA.png", "Decision curve analysis")
            with st.expander("More information on decision curve analysis"):
                st.write('''
                            Decision curve analysis (DCA) evaluates the clinical usefulness of a prediction model by quantifying the net benefit of using the model to guide decisions. Net benefit considers both the number of true positive and false positive predictions, weighted by the threshold probability—the risk level where a clinician or patient would choose to take action.

                            **How to read the decision curve figure**
                            The figure includes curves for:

                            - **The model** you are evaluating.
                            - **Treat-none** strategy (assuming no one is treated).
                            - **Treat-all** strategy (assuming everyone is treated).

                            The x-axis represents the **threshold probability**, and the y-axis shows the **net benefit**. Key points to consider:

                            - A model is clinically useful if its curve is above both the treat-none and treat-all lines across a range of threshold probabilities.
                            - The threshold probabilities where your model outperforms both strategies indicate where the model provides the most benefit.
                            - If the model curve lies below both strategies for a range of thresholds, it is not beneficial to use the model within that range.

                            **Important Note:**
                            Decision curve analysis should not be used to determine treatment decision thresholds. It serves as a statistical tool—similar to a p-value—providing evidence on whether the model has clinical value. It does not prescribe specific actions but rather supports the model's potential to inform decisions.
                        ''')
    st.write("")
    st.write("")
    st.write("")

    ### Model Explanation ###
    st.markdown("<h1 class='centered-header'>Model Explanation</h1>", unsafe_allow_html=True)

    with st.container():
        # Feature Importance
        col6, col7 = st.columns([1, 1])
        with col6:
            show_image_or_placeholder(
                f"{IMAGE_DIR}/SHAP Feature importances.png", "SHAP feature importances"
            )
            with st.expander("How to interpret this figure"):
                st.write(
                    """
                    Feature importance shows which variables contribute the most to the predictions of your model. The importance values reflect the average impact of each feature on the model's predictions, helping to explain which factors drive the model's output.

                    **How to read the figure**
                    - The **y-axis** lists the features (e.g., Injury Severity Score, age, ASIA impairment scale).
                    - The **x-axis** shows the mean importance value, which quantifies how much each feature influences the predictions on average.
                    - Higher values indicate greater importance.

                    Understanding feature importance can help validate whether the model aligns with clinical or domain knowledge and whether it is using meaningful factors to make predictions.
                    """
                )
        with col7:
            show_image_or_placeholder(
                f"{IMAGE_DIR}/SHAP Feature Beeplot.png", "SHAP beeswarm plot"
            )
            with st.expander("How to interpret this figure"):
                st.write("""
                        This figure shows how different features influence the model's predictions for individual instances.

                        **How to read the figure**
                        - The **y-axis** lists the predictors (e.g., Injury Severity Score, age, ASIA impairment scale).
                        - The **x-axis** shows the **impact on model output**, indicating how much each predictor's value affects the prediction. Positive values increase the prediction, while negative values decrease it.
                        - Each point represents a data sample, with its position reflecting how much that predictor contributed to the prediction for that sample.
                        - The **color** of the points indicates the predictor's input value:
                        - **Red** points correspond to higher predictor values (e.g. high ISS).
                        - **Blue** points correspond to lower predictor values (e.g. low ISS).

                        **Key insights:**
                        - A predictor with many points spread widely along the x-axis has a stronger and more varied influence on the model.
                        - Patterns in the color distribution (e.g., high values consistently increasing or decreasing predictions) can help identify the relationship between predictor values and predictions.

                        This figure provides a detailed view of how individual predictor values affect the model's outputs.
            """)

        st.write("")
        st.write("")
        st.write("")

        # Case explanation
        st.markdown(
            "<h1 class='centered-header'>Clinical Interpretation</h1>", unsafe_allow_html=True
        )
        st.write(
            """
            The predicted probabilities generated by our model are designed to support shared decision-making
            discussions in clinical settings. Clinicians can use these probabilities to guide conversations with
            patients and their families, integrating clinical context and individual preferences to align decisions
            with patient goals. Below, we present two cases that explain how the outputs of the model can be
            interpreted in practice.
            """
        )
        col8, col9 = st.columns([1, 1])
        with col8:
            show_image_or_placeholder(f"{IMAGE_DIR}/Case A.png", "Case A - explanation plot")
            with st.expander("How to interpret this figure"):
                st.write(
                    """
                    SCAFFOLD: replace this text with a walkthrough of Case A once the figure is exported.

                    Describe the patient, then read the plot from bottom to top: start at the average model
                    prediction, then state which predictors push the prediction down and which push it up,
                    ending at the final predicted probability.
                    """
                )
        with col9:
            show_image_or_placeholder(f"{IMAGE_DIR}/Case B.png", "Case B - explanation plot")
            with st.expander("How to interpret this figure"):
                st.write(
                    """
                    SCAFFOLD: replace this text with a walkthrough of Case B once the figure is exported.

                    Use a contrasting patient to Case A, so the two cases together show both a low and a high
                    predicted probability.
                    """
                )
        col10, col11 = st.columns([1, 1])
        with col10:
            st.write(
                """
                SCAFFOLD: describe how a low predicted probability of survival should be weighed in practice —
                for example when discussing the expected benefit of extensive surgical treatment against a
                conservative or palliative trajectory, and how patient preferences shift the threshold at which
                a given probability supports one option over another.
                """
            )
        with col11:
            st.write(
                """
                SCAFFOLD: describe how a high predicted probability of survival supports decision-making —
                for example providing reassurance for an invasive but potentially beneficial treatment in an
                elderly or frail patient, and where the model's estimate adds most to clinical gestalt.
                """
            )
    st.write("")
    st.write("")
    feedback_tool("Trauma - Evaluation")

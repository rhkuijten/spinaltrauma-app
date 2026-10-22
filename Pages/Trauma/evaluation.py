"""Evaluation page for the spinal trauma survival model.

The figures are exported from the analysis repository (Spine-Trauma/outputs) and
copied into Images/Trauma/ under the file names referenced here; see the README in
that folder for the source of each file. Any figure that is missing renders as a
labelled placeholder instead of breaking the page.

The numbers quoted in the explanatory text (AUC, calibration, odds ratios and the
waterfall walkthroughs) come from the same analysis run as the figures. Update the
text whenever the figures are regenerated.
"""

import streamlit as st
from styles import apply_custom_css, show_image_or_placeholder
from feedback import feedback_tool

apply_custom_css()

IMAGE_DIR = "Images/Trauma"


def waterfall_reading_guide():
    with st.expander("How to read these figures"):
        st.write(
            """
            Each figure explains one individual prediction. Read it from the bottom to the top:

            - The figure starts at the **reference** risk of 5.2% (dotted line at the bottom).
            - Each bar shows how one predictor moves the predicted risk: **red** bars increase it and
              **green** bars decrease it. Predictors with the smallest contributions are combined in
              **Other predictors**.
            - The numbers next to the bars are changes in absolute risk, in **percentage points**. For
              example, -1.7% means the risk drops by 1.7 percentage points (from 5.2% to 3.5%), not by
              1.7% of its value.
            - The value before the "=" is the patient's value for that predictor. For yes/no predictors,
              1 means present and 0 means absent.
            - The bold number at the top is the patient's **predicted 30-day mortality**.

            **What the reference value means**
            The reference of 5.2% is the predicted risk of a hypothetical patient with average values for
            all predictors. It is *not* the average predicted risk or the observed mortality in the cohort
            (both 13.3%): most patients have a low predicted risk and a small group a very high one, and
            averaging on the model's internal log-odds scale gives a lower value than averaging risks.

            **What the contributions mean**
            The contributions are SHAP values, computed on the model's log-odds scale and rescaled to
            percentage points. They show how the model arrived at this particular prediction. They are not
            the effect of changing a predictor or of a treatment, and the size of each contribution depends
            on the patient's other values.
            """
        )


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

                            **This model**
                            - The **red curve** is the model in the cohort it was developed in. The **grey curves** are models redeveloped on 1,000 bootstrap resamples of the data and applied to the original cohort. Closely overlapping curves indicate a stable model.
                            - The **dashed diagonal** is a model without any discrimination (AUC 0.50).
                            - The AUC in the development cohort is 0.923. Because a model always performs somewhat better in the data it was developed on, this was corrected for optimism using bootstrapping, giving an AUC of **0.908**: excellent discrimination.
                """)

        with col2:
            show_image_or_placeholder(f"{IMAGE_DIR}/Calibration plot.png", "Calibration plot")
            with st.expander("More information on calibration"):
                st.write("""
                            Calibration refers to how well the predicted probabilities from a model match the actual outcomes. A well-calibrated model assigns predictions that accurately reflect the observed event rates. For example, if a model predicts a 20% probability of an event, that event should occur approximately 20% of the time in similar cases.

                            **How to read the calibration plot**
                            A calibration plot compares predicted probabilities to observed event rates. Key features of the plot include:

                            - The black **dashed diagonal** represents perfect calibration, where predicted and observed probabilities are equal.
                            - The **red curve** shows the relationship between predicted and observed mortality for the model. Deviations from the diagonal indicate miscalibration.
                            - The **grey curves** are models redeveloped on 1,000 bootstrap resamples of the data, showing how much calibration varies between them.
                            - A model is **overestimating risk** if the curve is below the diagonal, meaning the predicted probabilities are higher than the actual event rates.
                            - A model is **underestimating risk** if the curve is above the diagonal, meaning predicted probabilities are too low.
                            - The **bars along the bottom** show how predicted risks are distributed among patients who died (red, above the axis) and patients who survived (green, below the axis). Most patients have a predicted risk below 10%.

                            After correction for optimism, the calibration slope was 0.98 and the calibration-in-the-large 0.01, close to their ideal values of 1 and 0.
                """)

        # Decision Curve Analysis
        col3, col4, col5 = st.columns([1, 2, 1])
        with col4:
            show_image_or_placeholder(f"{IMAGE_DIR}/DCA.png", "Decision curve analysis")
            with st.expander("More information on decision curve analysis"):
                st.write('''
                            Decision curve analysis (DCA) evaluates whether using a prediction model to guide a decision would do more good than harm, compared with simple default strategies. Because surgery is the default option for most patients with spinal trauma, this figure uses the **opt-out** framing: the model is used to identify patients whose predicted 30-day mortality is high enough that withholding surgery would be considered.

                            **How to read the decision curve figure**
                            The figure includes curves for:

                            - **The model** you are evaluating (red). The grey curves are models redeveloped on bootstrap resamples of the data and show how stable the curve is; they are not a confidence interval.
                            - **Operate on all** strategy (dashed grey line).
                            - **Withhold from all** strategy (black line at zero).

                            The x-axis represents the **threshold probability**: the predicted 30-day mortality at or above which surgery would be withheld. A low threshold reflects a strong wish to avoid surgery in patients who will not survive; a high threshold reflects a strong wish not to withhold surgery from patients who would survive. The y-axis shows the **standardised net benefit**: the net benefit divided by the proportion of patients who survived, so that 1 is the maximum (surgery for every patient who survives and for no patient who dies). Key points to consider:

                            - A model is clinically useful if its curve is above both reference strategies across a range of threshold probabilities.
                            - In this cohort, the model lies above "operate on all" across almost the entire range of thresholds, with the largest difference at the lower thresholds.
                            - These curves were estimated in the cohort the model was developed in and were not corrected for optimism.

                            **Important Note:**
                            Decision curve analysis should not be used to determine treatment decision thresholds. The model predicts mortality; it does not estimate the effect of surgery, so this figure cannot show that withholding surgery improves outcomes. It serves as a statistical tool—similar to a p-value—providing evidence on whether the model has clinical value. It does not prescribe specific actions but rather supports the model's potential to inform decisions.
                        ''')
    st.write("")
    st.write("")
    st.write("")

    ### Model Explanation ###
    st.markdown("<h1 class='centered-header'>Model Explanation</h1>", unsafe_allow_html=True)

    with st.container():
        col6, col7 = st.columns([1, 1])
        with col6:
            show_image_or_placeholder(
                f"{IMAGE_DIR}/Coefficient forest plot.png", "Coefficient forest plot"
            )
            with st.expander("How to interpret this figure"):
                st.write(
                    """
                    The model is a logistic regression with ridge penalisation. This figure shows, for each predictor, the odds ratio (OR) for 30-day mortality when all other predictors are held constant.

                    **How to read the figure**
                    - Each **dot** is the OR and the **horizontal line** its 95% bootstrap interval, from 1,000 repetitions of the full model development.
                    - The **dashed vertical line** marks an OR of 1 (no association). Dots to the right indicate a higher predicted mortality, dots to the left a lower predicted mortality.
                    - The x-axis is logarithmic: an OR of 10 lies as far to the right of the line as an OR of 0.1 lies to the left.
                    - Predictors are ordered by the size of their standardised coefficient, i.e. how strongly they influence predictions across the cohort.

                    **Units**
                    - Yes/no predictors (e.g. lumbar injury, traumatic brain injury): present vs absent.
                    - Injury mechanism ("Intent") categories: compared with a low-energy fall.
                    - ASA grade, ASIA grade (A to E) and mFI-5 frailty index: per grade. Number of fractures: per additional fracture (6 or more counted as 6).
                    - Injury Severity Score and BMI were transformed (Yeo-Johnson) before modelling; their OR is per standard deviation of the transformed value.
                    - Age is shown in the figure on the right.

                    The strongest associations are a neurological examination that was not possible (OR 31.6), a higher Injury Severity Score (OR 2.8 per SD) and a higher ASA grade (OR 1.7 per grade). Odds ratios describe associations within the model, not causal effects. Because of the ridge penalty, they are shrunk towards 1 to reduce overfitting.
                    """
                )
        with col7:
            show_image_or_placeholder(f"{IMAGE_DIR}/Age spline.png", "Age spline")
            with st.expander("How to interpret this figure"):
                st.write("""
                        Age was modelled with a restricted cubic spline, which allows its association with mortality to curve instead of forcing a straight line. Age therefore has no single odds ratio. Instead, this figure shows the OR at each age compared with a 50-year-old patient with otherwise identical characteristics.

                        **How to read the figure**
                        - The **red curve** is the OR compared with age 50 (dotted vertical line, where the OR is 1). The y-axis is logarithmic.
                        - The **shaded band** is the 95% bootstrap interval from 1,000 repetitions of the model development. It is wide at the youngest and oldest ages, where there are fewer patients.

                        **Key insights:**
                        - Below 50 years the curve is nearly flat: age has little influence on predicted mortality.
                        - Above 50 years the predicted odds of death rise steeply. Compared with age 50, the OR is about 1.4 at 60, 2.6 at 70, 5.5 at 80 and 11.7 at 90 years.
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
            with patient goals. Below, we present three patients from the development cohort, at the 5th, 50th and
            95th percentile of predicted risk, and show how the model arrived at each prediction.
            """
        )
        waterfall_reading_guide()

        tab_low, tab_median, tab_high = st.tabs(
            ["Low risk (0.4%)", "Median risk (4.3%)", "High risk (80.5%)"]
        )
        with tab_low:
            show_image_or_placeholder(
                f"{IMAGE_DIR}/Waterfall low risk.png", "Waterfall plot - low risk"
            )
            st.write(
                """
                **Patient at the 5th percentile of predicted risk: 0.4%.** A 21-year-old patient injured in a
                motor vehicle accident, with an Injury Severity Score of 10, an AO type F fracture, ASA grade 1
                and a neurological examination that could be performed. Almost every predictor lowers the risk:
                the low Injury Severity Score has the largest effect (-1.7 percentage points), followed by the
                AO type F fracture (-1.1), young age (-0.9) and ASA grade 1 (-0.7). Only the absence of a lumbar
                injury raises the risk slightly (+0.4).
                """
            )
        with tab_median:
            show_image_or_placeholder(
                f"{IMAGE_DIR}/Waterfall median risk.png", "Waterfall plot - median risk"
            )
            st.write(
                """
                **Patient at the 50th percentile of predicted risk: 4.3%.** A 56-year-old patient with an Injury
                Severity Score of 14, a thoracic injury, an injury mechanism classified as "other" and a
                neurological examination that could be performed. The Injury Severity Score (-1.9 percentage
                points), age (-1.6) and the possible neurological examination (-1.2) lower the risk, while the
                "other" injury mechanism (+3.4) and the absence of a lumbar injury (+1.0) raise it. Together,
                the prediction ends slightly below the reference. Because most patients have a low predicted
                risk, this median patient's risk is well below the 13.3% mortality in the cohort.
                """
            )
        with tab_high:
            show_image_or_placeholder(
                f"{IMAGE_DIR}/Waterfall high risk.png", "Waterfall plot - high risk"
            )
            st.write(
                """
                **Patient at the 95th percentile of predicted risk: 80.5%.** A 56-year-old patient whose
                neurological examination was not possible, with an Injury Severity Score of 36 and a traumatic
                brain injury on CT. The impossible neurological examination raises the risk by far the most
                (+55.5 percentage points), followed by the high Injury Severity Score (+21.4) and the brain
                injury (+5.5). Age (-5.6) and the remaining predictors (-1.5) lower it only slightly.
                """
            )
    st.write("")
    st.write("")
    feedback_tool("Trauma - Evaluation")

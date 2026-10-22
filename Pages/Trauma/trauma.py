import streamlit as st

from Pages.Trauma import trauma_utils as tu
from styles import apply_custom_css
from feedback import feedback_tool

apply_custom_css()


def reset_state():
    st.session_state["trauma_model_output"] = None
    st.session_state["trauma_explanation_plot"] = None


def trauma_page():

    st.header("Decision Tool")

    # Every input starts empty: the model has not been tested with missing values,
    # so nothing is pre-filled and Predict stays disabled until all are entered.
    st.markdown("<h2 class='section-header'>Patient characteristics</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(
            "Age (years)", value=None, min_value=18, max_value=120, step=1, on_change=reset_state
        )
        bmi = st.number_input(
            "BMI (kg/m²)",
            value=None,
            min_value=10.0,
            max_value=80.0,
            step=0.1,
            format="%0.1f",
            on_change=reset_state,
        )
    with col2:
        sex = st.radio("Sex", tu.labels(tu.SEX_OPTIONS), index=None, on_change=reset_state)
        asa = st.selectbox(
            "ASA classification",
            tu.labels(tu.ASA_OPTIONS),
            index=None,
            on_change=reset_state,
            help="American Society of Anesthesiologists physical status classification.",
        )

    mfi5 = st.selectbox(
        "mFI-5 frailty index",
        tu.labels(tu.MFI5_OPTIONS),
        index=None,
        on_change=reset_state,
        help="5-item modified frailty index: number of items present. "
        "Three or more items are treated as a single group by the model.",
    )

    st.markdown("<h2 class='section-header'>Injury</h2>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        intentie = st.selectbox(
            "Injury mechanism",
            tu.labels(tu.INTENTIE_OPTIONS),
            index=None,
            on_change=reset_state,
        )
        iss08 = st.number_input(
            "Injury Severity Score (ISS-08)",
            value=None,
            min_value=0,
            max_value=75,
            step=1,
            on_change=reset_state,
            help="Injury Severity Score based on AIS 2008.",
        )
    with col4:
        n_fractures = st.number_input(
            "Number of fractures",
            value=None,
            min_value=1,
            step=1,
            on_change=reset_state,
            help="Total number of spinal fractures.",
        )
        tbi_ct = st.radio(
            "Traumatic brain injury on CT",
            tu.labels(tu.TBI_CT_OPTIONS),
            index=None,
            on_change=reset_state,
        )

    st.markdown("<h2 class='section-header'>Spinal injury</h2>", unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        st.write("**AO Spine fracture type(s)**")
        st.caption("Types are not mutually exclusive - tick every type present (at least one).")
        ao_types = {
            column: st.checkbox(
                tu.AO_TYPE_LABELS[column], key=f"ao_{column}", on_change=reset_state
            )
            for column in tu.AO_TYPE_COLUMNS
        }
    with col6:
        st.write("**Spinal region(s) involved**")
        st.caption("Tick every region involved (at least one).")
        regions = {
            column: st.checkbox(
                tu.REGION_LABELS[column], key=f"region_{column}", on_change=reset_state
            )
            for column in tu.REGION_COLUMNS
        }

    st.markdown("<h2 class='section-header'>Neurology</h2>", unsafe_allow_html=True)
    asia = st.selectbox(
        "ASIA impairment scale",
        tu.labels(tu.ASIA_OPTIONS),
        index=None,
        on_change=reset_state,
        help="Select 'Exam not performable' when the neurological examination could not be "
        "carried out, for example due to intubation, sedation, or severe traumatic brain "
        "injury. The model accounts for this automatically.",
    )

    if "trauma_model_output" not in st.session_state:
        st.session_state["trauma_model_output"] = None
    if "trauma_explanation_plot" not in st.session_state:
        st.session_state["trauma_explanation_plot"] = None

    required = {
        "Age": age,
        "BMI": bmi,
        "Sex": sex,
        "ASA classification": asa,
        "mFI-5 frailty index": mfi5,
        "Injury mechanism": intentie,
        "Injury Severity Score": iss08,
        "Number of fractures": n_fractures,
        "Traumatic brain injury on CT": tbi_ct,
        "ASIA impairment scale": asia,
    }
    missing = [label for label, value in required.items() if value is None]
    # A checkbox has no empty state, so "none ticked" is what an unanswered group
    # looks like. Every spinal fracture has an AO type and a region.
    if not any(ao_types.values()):
        missing.append("AO Spine fracture type")
    if not any(regions.values()):
        missing.append("Spinal region")

    st.write("")
    if missing:
        st.info(
            "The model has not been tested with missing values, so every field must be "
            f"filled in before a prediction can be made. Still missing: {', '.join(missing)}."
        )
    if st.button("Predict", disabled=bool(missing)):
        with st.spinner("Predicting..."):
            pipeline = tu.load_model()

            if pipeline is None:
                st.error(
                    f"The trauma model has not been added yet. Place the fitted "
                    f"pipeline at `{tu.MODEL_PATH}` to enable predictions."
                )
            else:
                # Raw values only - the pipeline does all cleaning, imputation,
                # spline expansion, one-hot encoding and scaling internally.
                row = tu.build_input_row({
                    "LEEFTIJD_EXACT": age,
                    "Geslacht01": tu.value_for(tu.SEX_OPTIONS, sex),
                    "BMI": bmi,
                    "ISS08": iss08,
                    "NUMBER_OF_FRACTURES": n_fractures,
                    "MFI5": tu.value_for(tu.MFI5_OPTIONS, mfi5),
                    "COMORB": tu.value_for(tu.ASA_OPTIONS, asa),
                    "ASIA": tu.value_for(tu.ASIA_OPTIONS, asia),
                    "TBI_CT": tu.value_for(tu.TBI_CT_OPTIONS, tbi_ct),
                    **{column: int(checked) for column, checked in ao_types.items()},
                    **{
                        column: ("JA" if checked else "NEE")
                        for column, checked in regions.items()
                    },
                    "INTENTIE": tu.value_for(tu.INTENTIE_OPTIONS, intentie),
                })

                st.session_state["trauma_model_output"] = tu.predict(pipeline, row)

                explanation = tu.compute_shap_values(pipeline, row)
                st.session_state["trauma_explanation_plot"] = (
                    tu.plot_explanation(explanation) if explanation is not None else None
                )

    # Display the model output and explanation plot if available
    if st.session_state["trauma_model_output"] is not None:
        mortality = st.session_state["trauma_model_output"]
        st.metric(
            label=f"**Probability of {tu.OUTCOME_LABEL}**",
            value=f"{round(mortality * 100)}%",
        )

    if st.session_state["trauma_explanation_plot"] is not None:
        st.markdown("<h1 class='centered-header'>Explanation Plot</h1>", unsafe_allow_html=True)
        st.pyplot(st.session_state["trauma_explanation_plot"])
        with st.expander("How do I interpret this figure?"):
            st.write(
                """
                The figure above shows the contribution of each factor to the model's prediction.
                Please consult the *Evaluation* page at *Clinical Interpretation* for three worked
                examples and for explanation on how to interpret a prediction in clinical setting.

                Read the figure from the bottom to the top. It starts at the **reference** risk of
                5.2%: the predicted risk of a hypothetical patient with average values for all
                predictors. This is not the average predicted risk or the observed mortality in the
                development cohort, which are both 13.3%. Each bar then moves the risk up (**red**)
                or down (**green**) until it reaches this patient's prediction, shown in bold at the
                top. Contributions are shown in **percentage points of absolute risk**, so they can be
                read directly against the predicted probability.

                One caveat on that scale: the model works in log-odds, where the contributions are
                exactly additive. Putting them in percentage points rescales them by a single
                positive factor, so every contribution keeps its direction and the ranking is
                unchanged, but each number is that predictor's *share of the total shift* from the
                reference rather than its effect in isolation.

                Age is modelled as a spline and the injury mechanism as a set of categories; each is
                added back together here so it appears as a single row. Predictors whose
                contribution is 5% or less of all contributions added together (regardless of
                direction) are combined into *Other predictors*.
                *Neurological exam not possible* is derived from the ASIA field rather than asked
                separately, so it appears as its own row.

                ### Explanation of Prediction Factors

                **ISS (Injury Severity Score):**
                An anatomical score of overall trauma burden, ranging from 0 to 75. Higher scores
                reflect more severe polytrauma.

                **ASA classification:**
                Preinjury physical status, from I (healthy) to V (moribund). Higher classes reflect
                more severe systemic disease.

                **mFI-5 (5-item modified frailty index):**
                Counts the number of frailty items present. Higher values reflect greater frailty;
                three or more items are treated as a single group.

                **ASIA impairment scale:**
                Neurological status of the spinal cord injury:

                - **A:** Complete injury, no motor or sensory function preserved.
                - **B-D:** Incomplete injury, with increasing preserved function.
                - **E:** Normal motor and sensory function.

                When the examination could not be performed, the model accounts for this separately
                rather than assuming a grade.

                **AO Spine fracture types:**
                Morphological classification of the spinal fracture: A (compression), B
                (distraction / tension band), C (translation), and F (facet injury). More than one
                type can be present.
                """
            )
    st.write("")
    st.write("")
    feedback_tool("Trauma - Decision tool")

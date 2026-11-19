"""
Streamlit UI for Chest X-Ray Pneumonia Classifier MLOps Dashboard.

Five pages:
  1. 🏠 Home          – system status & health
  2. 🔍 Predict       – single-image inference
  3. 📊 Visualizations – dataset insights & feature analysis
  4. 📤 Upload Data   – submit images for retraining
  5. 🔄 Retrain       – trigger & monitor model retraining
"""

import os
import time
import io

import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# ─────────────────────────── config ────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ChestXRay MLOps",
    page_icon="🫁",
    layout="wide",
)

# ─────────────────────────── global CSS ────────────────────────────────────
st.markdown(
    """
    <style>
    /* Prediction result boxes */
    .pred-box {
        padding: 20px 28px;
        border-radius: 12px;
        font-size: 1.5rem;
        font-weight: 700;
        text-align: center;
        margin: 12px 0;
        letter-spacing: 0.03em;
    }
    .pred-normal {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .pred-pneumonia {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }

    /* Metric card tweaks */
    div[data-testid="metric-container"] {
        background: #f7f9fc;
        border: 1px solid #e1e4ea;
        border-radius: 10px;
        padding: 16px 20px;
    }

    /* Sidebar nav label */
    section[data-testid="stSidebar"] .stSelectbox label {
        font-weight: 600;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────── helpers ───────────────────────────────────────

def api_get(path: str, timeout: int = 8):
    """GET from the API; returns (data_dict_or_None, error_str_or_None)."""
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API server. Is it running?"
    except requests.exceptions.HTTPError as exc:
        return None, f"HTTP {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, **kwargs):
    """POST to the API; returns (data_dict_or_None, error_str_or_None)."""
    try:
        r = requests.post(f"{API_URL}{path}", timeout=30, **kwargs)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach the API server. Is it running?"
    except requests.exceptions.HTTPError as exc:
        return None, f"HTTP {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:
        return None, str(exc)


def fmt_uptime(seconds):
    """Convert raw seconds into a human-readable uptime string."""
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ─────────────────────────── sidebar nav ───────────────────────────────────
PAGES = ["🏠 Home", "🔍 Predict", "📊 Visualizations", "📤 Upload Data", "🔄 Retrain"]

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/x-ray.png", width=72)
    st.markdown("## 🫁 ChestXRay MLOps")
    st.markdown("---")
    page = st.selectbox("Navigate to", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"API: `{API_URL}`")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🫁 Chest X-Ray Pneumonia Classifier")
    st.subheader("MLOps Pipeline Dashboard")
    st.markdown(
        """
        This dashboard provides an end-to-end view of a **deep-learning MLOps pipeline**
        for detecting pneumonia in chest X-rays.  
        The model uses **MobileNetV2** transfer learning trained on the
        [Kaggle Chest X-Ray Images dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

        Use the sidebar to navigate between pages:
        | Page | Purpose |
        |------|---------|
        | 🔍 Predict | Upload an X-ray and get an instant diagnosis |
        | 📊 Visualizations | Explore class distribution and model confidence |
        | 📤 Upload Data | Contribute new labelled images for retraining |
        | 🔄 Retrain | Trigger and monitor a fine-tuning run |
        """
    )
    st.markdown("---")

    health, h_err = api_get("/health")
    status, s_err = api_get("/status")

    if h_err or s_err:
        st.warning("⚠️ API is offline")
        st.error(h_err or s_err)
    else:
        model_loaded = health.get("model_loaded", False)
        uptime_secs = health.get("uptime_seconds")
        last_trained = status.get("last_trained") or "Never"
        accuracy = status.get("accuracy")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            label="🔵 Model Status",
            value="Online ✅" if model_loaded else "Offline ❌",
        )
        c2.metric(
            label="⏱️ Uptime",
            value=fmt_uptime(uptime_secs),
        )
        c3.metric(
            label="📅 Last Trained",
            value=last_trained if last_trained != "Never" else "—",
            help="ISO timestamp of the most recent training run",
        )

        if accuracy is not None:
            st.metric("🎯 Accuracy (last run)", f"{accuracy:.2%}")

        st.success("API is reachable ✅")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 – PREDICT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.title("🔍 Single Image Prediction")
    st.markdown(
        "Upload a chest X-ray (JPEG or PNG) and the model will classify it as "
        "**NORMAL** or **PNEUMONIA** with a confidence score."
    )
    st.markdown("---")

    uploaded = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Accepts JPEG and PNG formats",
    )

    if uploaded is not None:
        col_img, col_res = st.columns([1, 1])

        with col_img:
            st.subheader("Uploaded Image")
            st.image(uploaded, use_column_width=True, caption=uploaded.name)

        with col_res:
            st.subheader("Prediction Result")
            run = st.button("🚀 Run Prediction", use_container_width=True)

            if run:
                with st.spinner("Running inference…"):
                    image_bytes = uploaded.getvalue()
                    data, err = api_post(
                        "/predict",
                        files={"file": (uploaded.name, io.BytesIO(image_bytes), "image/jpeg")},
                    )

                if err:
                    st.error(f"Prediction failed: {err}")
                else:
                    pred_class = data.get("class", "UNKNOWN")
                    confidence = data.get("confidence", 0.0)
                    probs = data.get("probabilities", {})

                    # Colored result box
                    box_cls = "pred-normal" if pred_class == "NORMAL" else "pred-pneumonia"
                    icon = "✅" if pred_class == "NORMAL" else "🔴"
                    st.markdown(
                        f'<div class="pred-box {box_cls}">{icon} {pred_class}</div>',
                        unsafe_allow_html=True,
                    )

                    # Confidence bar
                    st.markdown(f"**Confidence: {confidence:.1%}**")
                    st.progress(float(confidence))

                    # Probability table
                    st.markdown("#### Class Probabilities")
                    prob_df = pd.DataFrame(
                        {
                            "Class": ["NORMAL", "PNEUMONIA"],
                            "Probability": [
                                f"{probs.get('NORMAL', 0.0):.2%}",
                                f"{probs.get('PNEUMONIA', 0.0):.2%}",
                            ],
                        }
                    )
                    st.table(prob_df)
    else:
        st.info("👆 Upload an X-ray image above to get started.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 – VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualizations":
    st.title("📊 Dataset Insights & Feature Analysis")
    st.markdown(
        "Explore the training data distribution, understand class imbalance, "
        "and examine the confidence distribution of recent model predictions."
    )
    st.markdown("---")

    viz_data, viz_err = api_get("/visualizations")

    if viz_err:
        st.warning("⚠️ Could not load visualization data from the API.")
        st.error(viz_err)
        # Provide sensible fall-back mock data so the page still renders
        viz_data = {
            "class_distribution": {"NORMAL": 1341, "PNEUMONIA": 3875},
            "confidence_distribution": [],
        }

    # ── CHART 1: Class distribution ────────────────────────────────────────
    class_dist = viz_data.get("class_distribution", {})
    if class_dist:
        with st.expander("📌 Chart 1: Training Class Distribution", expanded=True):
            dist_df = pd.DataFrame(
                {
                    "Class": list(class_dist.keys()),
                    "Count": list(class_dist.values()),
                }
            )
            fig1 = px.bar(
                dist_df,
                x="Class",
                y="Count",
                color="Class",
                color_discrete_map={"NORMAL": "#28a745", "PNEUMONIA": "#dc3545"},
                text="Count",
                title="Training Set Class Distribution",
                height=400,
            )
            fig1.update_traces(textposition="outside")
            fig1.update_layout(showlegend=False, xaxis_title="Diagnosis Class", yaxis_title="Number of Images")
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown(
                """
                **Interpretation — Class Distribution:**

                The bar chart reveals a clear **class imbalance** in the training dataset: the PNEUMONIA class
                contains roughly **3× as many images** as the NORMAL class (≈ 3,875 vs 1,341 images in the
                standard Kaggle split). This imbalance is clinically meaningful — pneumonia is unfortunately
                common, so real-world data naturally skews toward positive cases.

                From a machine-learning perspective, an imbalanced dataset can bias the model toward predicting
                the majority class (PNEUMONIA), inflating raw accuracy while masking poor recall on the minority
                class. A model trained naively on this split might achieve >70 % accuracy by simply predicting
                PNEUMONIA every time — which would be clinically useless.

                To mitigate this, the pipeline uses **data augmentation** (flips, zoom, brightness jitter) on
                the minority class and reports **F1-score, Precision, and Recall** in addition to accuracy, so
                that model quality is judged holistically rather than by a single potentially misleading metric.
                """
            )

    # ── CHART 2: Class imbalance analysis (text + gauge) ───────────────────
    with st.expander("📌 Chart 2: Class Imbalance Analysis & Model Bias", expanded=True):
        normal_count = class_dist.get("NORMAL", 1341)
        pneumonia_count = class_dist.get("PNEUMONIA", 3875)
        ratio = pneumonia_count / max(normal_count, 1)

        gauge_fig = px.pie(
            values=[normal_count, pneumonia_count],
            names=["NORMAL", "PNEUMONIA"],
            color_discrete_sequence=["#28a745", "#dc3545"],
            title=f"Class Ratio  (PNEUMONIA : NORMAL ≈ {ratio:.1f} : 1)",
            hole=0.45,
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown(
            f"""
            **Interpretation — Class Imbalance & Model Bias:**

            With a PNEUMONIA-to-NORMAL ratio of **{ratio:.1f} : 1**, the dataset exhibits significant
            class imbalance. When a neural network is exposed to far more examples of one class, its
            decision boundary shifts toward that class: gradients from PNEUMONIA samples dominate the
            weight updates, causing the model to become *biased* — it learns to favour "PNEUMONIA" as
            a safe default because doing so minimises training loss on the majority of examples.

            In a **clinical setting** this bias has asymmetric consequences. A false negative
            (missed pneumonia) is far more dangerous than a false positive (unnecessary antibiotic
            course). Therefore, it is preferable to tune the classification threshold toward higher
            **recall** (sensitivity) for the PNEUMONIA class, even at the cost of slightly lower
            precision, so the model errs on the side of flagging borderline cases for radiologist review.

            Practical mitigation strategies include **class-weighted loss** (penalising
            NORMAL misclassifications less than PNEUMONIA ones), **oversampling** the NORMAL class
            with augmentation during retraining, and monitoring **ROC-AUC** as the primary
            deployment metric since it is insensitive to class prevalence.
            """
        )

    # ── CHART 3: Confidence distribution of recent predictions ─────────────
    conf_scores = viz_data.get("confidence_distribution", [])
    with st.expander("📌 Chart 3: Recent Prediction Confidence Distribution", expanded=True):
        if conf_scores:
            conf_df = pd.DataFrame({"Confidence": conf_scores})
            fig3 = px.histogram(
                conf_df,
                x="Confidence",
                nbins=20,
                range_x=[0, 1],
                color_discrete_sequence=["#5b8dee"],
                title="Histogram of Model Confidence Scores (Recent Predictions)",
                labels={"Confidence": "Confidence Score"},
                height=380,
            )
            fig3.update_layout(yaxis_title="Number of Predictions")
            st.plotly_chart(fig3, use_container_width=True)

            mean_conf = sum(conf_scores) / len(conf_scores)
            high_conf = sum(1 for c in conf_scores if c >= 0.9) / len(conf_scores)

            st.markdown(
                f"""
                **Interpretation — Confidence Score Distribution:**

                The histogram shows the **distribution of model certainty** across the most recent
                {len(conf_scores)} predictions. The average confidence is **{mean_conf:.1%}**,
                and **{high_conf:.0%}** of predictions exceeded the 90 % threshold — indicating the
                model is generally decisive rather than uncertain on these inputs.

                A well-calibrated classifier should not always be maximally confident; cases near the
                decision boundary (confidence ≈ 50 %) are genuinely ambiguous and should be escalated
                to human review. If the histogram is heavily skewed toward 100 %, the model may be
                **overconfident**, which is a known side-effect of softmax classifiers on
                out-of-distribution inputs.

                Monitoring this distribution over time is a lightweight form of **data drift
                detection**: a sudden increase in low-confidence predictions signals that the incoming
                image distribution may be shifting away from the training distribution, triggering a
                retraining cycle via the 🔄 Retrain page.
                """
            )
        else:
            st.info(
                "No prediction history found yet. Run some predictions on the **🔍 Predict** page "
                "and revisit this chart."
            )
            st.markdown(
                """
                **Interpretation — Confidence Score Distribution:**

                Once predictions are logged, this histogram will display the **spread of model
                certainty** across recent inference requests. A healthy model should produce a
                bimodal distribution — high confidence on clear-cut cases and lower confidence
                on borderline cases — rather than uniformly extreme scores.

                Tracking this metric over time acts as a lightweight **model health monitor**.
                Gradual drift toward the centre of the distribution (many uncertain predictions)
                can indicate concept drift, degraded image quality in the incoming feed, or a
                need to include new clinical conditions in future training data.

                This page will auto-populate as users submit X-rays on the Predict page.
                All predictions are logged server-side and surfaced here via the `/visualizations`
                API endpoint.
                """
            )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 – UPLOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📤 Upload Data":
    st.title("📤 Upload Data for Retraining")
    st.markdown(
        "Contribute labelled chest X-ray images to the retraining pool. "
        "Select the correct diagnosis label before uploading."
    )
    st.markdown("---")

    label = st.radio(
        "🏷️ Image Label",
        options=["NORMAL", "PNEUMONIA"],
        horizontal=True,
        help="Select the ground-truth diagnosis for all images you are about to upload.",
    )

    files = st.file_uploader(
        f"Upload {label} X-ray images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="You can select multiple files at once.",
    )

    if files:
        st.markdown(f"**{len(files)} file(s) selected** — label: `{label}`")
        # Preview first 5 thumbnails
        preview_cols = st.columns(min(len(files), 5))
        for i, f in enumerate(files[:5]):
            preview_cols[i].image(f, caption=f.name, use_column_width=True)
        if len(files) > 5:
            st.caption(f"… and {len(files) - 5} more file(s) not previewed.")

    upload_btn = st.button("⬆️ Upload Files", disabled=(not files), use_container_width=True)

    if upload_btn and files:
        with st.spinner(f"Uploading {len(files)} file(s)…"):
            upload_files = [
                ("files", (f.name, f.getvalue(), "image/jpeg")) for f in files
            ]
            data, err = api_post(
                "/upload",
                files=upload_files,
                data={"label": label},
            )

        if err:
            st.error(f"Upload failed: {err}")
        else:
            saved = data.get("saved", len(files))
            st.success(f"✅ {saved} file(s) uploaded successfully as **{label}**.")

            # Show recent uploads table if returned
            recent = data.get("recent_uploads") or []
            if recent:
                st.markdown("#### Recent Uploads")
                st.dataframe(pd.DataFrame(recent), use_container_width=True)

    # Also fetch status to show any existing upload counts
    st.markdown("---")
    st.subheader("📋 Current Upload Pool")
    status, s_err = api_get("/status")
    if s_err:
        st.warning("Could not retrieve upload pool info from the API.")
    else:
        upload_counts = status.get("upload_counts", {})
        if upload_counts:
            pool_df = pd.DataFrame(
                [{"Label": k, "Images Available": v} for k, v in upload_counts.items()]
            )
            st.dataframe(pool_df, use_container_width=True)
        else:
            st.info("No images uploaded yet, or the API did not return pool counts.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 – RETRAIN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔄 Retrain":
    st.title("🔄 Model Retraining")
    st.markdown(
        "Fine-tune the model on newly uploaded images. Retraining runs in the background; "
        "this page polls for completion every 3 seconds."
    )
    st.markdown("---")

    # ── Current metrics ─────────────────────────────────────────────────────
    st.subheader("📈 Current Model Metrics")
    metrics, m_err = api_get("/metrics")

    if m_err:
        st.warning(f"Could not load metrics: {m_err}")
    elif metrics:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
        m2.metric("Loss", f"{metrics.get('loss', 0):.4f}")
        m3.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
        m4.metric("Precision", f"{metrics.get('precision', 0):.2%}")
        m5.metric("Recall", f"{metrics.get('recall', 0):.2%}")
        m6.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
    else:
        st.info("No metrics available yet. Train the model first.")

    st.markdown("---")

    # ── Upload pool count ───────────────────────────────────────────────────
    st.subheader("🗂️ Uploaded Images Available for Retraining")
    status, s_err = api_get("/status")
    if s_err:
        st.warning("Could not retrieve upload counts.")
    else:
        upload_counts = status.get("upload_counts", {})
        total_uploads = sum(upload_counts.values()) if upload_counts else 0
        st.metric("Total Uploaded Images", total_uploads)
        if upload_counts:
            for lbl, cnt in upload_counts.items():
                st.write(f"- **{lbl}**: {cnt} image(s)")

    st.markdown("---")

    # ── Trigger retraining ──────────────────────────────────────────────────
    retrain_btn = st.button(
        "🔄 Trigger Retraining",
        use_container_width=True,
        type="primary",
    )

    if retrain_btn:
        trigger_data, trigger_err = api_post("/retrain")
        if trigger_err:
            st.error(f"Failed to start retraining: {trigger_err}")
        else:
            st.info("⏳ Retraining started in background…")

            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            MAX_POLLS = 10

            for poll_i in range(1, MAX_POLLS + 1):
                time.sleep(3)
                rt_status, rt_err = api_get("/retrain/status")

                if rt_err:
                    status_placeholder.warning(f"Could not poll status: {rt_err}")
                    break

                current_status = rt_status.get("status", "unknown")
                message = rt_status.get("message", "")
                progress_bar.progress(min(poll_i / MAX_POLLS, 1.0))

                if current_status == "completed":
                    status_placeholder.success(f"✅ Retraining completed! {message}")
                    st.balloons()
                    break
                elif current_status == "failed":
                    status_placeholder.error(f"❌ Retraining failed: {message}")
                    break
                else:
                    status_placeholder.info(
                        f"⏳ Status: **{current_status}** — {message} "
                        f"(poll {poll_i}/{MAX_POLLS})"
                    )
            else:
                status_placeholder.warning(
                    "Retraining is still running in the background. "
                    "Refresh this page to check the final result."
                )

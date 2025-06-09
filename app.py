import os
import pandas as pd
import streamlit as st

from utils.plot_utils import plot_prediction_distribution

# Config & paths
DATA_DIR = "data/videos"
GRID          = 6     # num columns in each grid
VID_W    = 180        # video width in pixels
MODEL_OPTIONS = {     # Available models paths
    "Balanced Random Forest": "data/predictions/BalancedRF.csv",
    "Logistic Regression": "data/predictions/LogReg.csv",
    "XGBoost": "data/predictions/XGBoost.csv",
}

# Load supplementary metadata
meta_df = pd.read_csv("data/videos_metadata.csv", dtype=str).fillna("")

# Create a mapping from video_id to username and description
video_meta = {
    row["video_id"]: {
        "username": row["username"],
        "description": row["description"]
    }
    for _, row in meta_df.iterrows()
}

st.set_page_config(page_title="Video model demo", layout="wide")

# global CSS for video width
st.html(
    f"<style> video {{width:{VID_W}px !important; height:auto;}} </style>",
)

# Model selector
st.markdown("### Select model:")
model_choice = st.radio("Select model", list(MODEL_OPTIONS.keys()), label_visibility="collapsed")

# Load metadata and keep rows whose video file exists
@st.cache_data
def load_model_results(csv_path, video_dir):
    model_df = pd.read_csv(csv_path, dtype=str)
    model_df["file_path"] = model_df["video_id"].apply(lambda v: os.path.join(video_dir, f"{v}.mp4"))
    model_df = model_df[model_df["file_path"].apply(os.path.isfile)].reset_index(drop=True)
    return model_df

csv = MODEL_OPTIONS[model_choice]
df = load_model_results(csv, DATA_DIR)
videos_ids = df["video_id"].tolist()

# Initialise session selection state for checkboxes
if "chosen" not in st.session_state:
    st.session_state["chosen"] = {vid: False for vid in videos_ids}

st.markdown("### Select videos:")

# Add buttons to select/deselect all videos
col1, col2, col_spacer = st.columns([1, 1, 8])
with col1:
    if st.button("Select All", icon=":material/check_box:"):
        for vid in videos_ids:
            st.session_state["chosen"][vid] = True
with col2:
    if st.button("Deselect All", icon=":material/check_box_outline_blank:"):
        for vid in videos_ids:
            st.session_state["chosen"][vid] = False

# Grid of video previews with checkboxes
rows = [videos_ids[i:i+GRID] for i in range(0, len(videos_ids), GRID)]
for r in rows:
    cols = st.columns(len(r))
    for col, vid in zip(cols, r):
        path = df.loc[df.video_id == vid, "file_path"].values[0]
        with col:
            st.video(path)
            # checkbox state is kept in session state
            st.session_state["chosen"][vid] = st.checkbox(
                label   = vid,
                key     = f"chk_{vid}",
                value   = st.session_state["chosen"][vid],
            )

# Grid of chosen videos with their model output
if st.button("Show model's output", type="primary", use_container_width=True):
    selected = [vid for vid, flag in st.session_state["chosen"].items() if flag]
    if not selected:
        st.warning("No videos selected.")
    else:
        st.subheader("Model's output")
        rows = [selected[i:i+GRID] for i in range(0, len(selected), GRID)]
        for row_ids in rows:
            cols = st.columns(GRID)
            for slot, col in enumerate(cols):
                with col:
                    # If the slot is within the range of row_ids, display the video and its labels
                    if slot < len(row_ids):
                        vid = row_ids[slot]
                        predicted = df.loc[df.video_id == vid, "predicted_label"].values[0]
                        true = df.loc[df.video_id == vid, "true_label"].values[0]
                        path = df.loc[df.video_id == vid, "file_path"].values[0]
                        prob_col = f"prob_{predicted}"
                        prob = df.loc[df.video_id == vid, prob_col].values[0]

                        meta = video_meta.get(vid, {"username": "", "description": ""})
                        username = meta["username"]
                        description = meta["description"]

                        # Display the video's description and uploader username
                        st.markdown(f"**Username:** {username}", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="max-height: 4em; min-height:4em; margin-bottom: 20px; overflow-y: auto; 
                        font-size: 0.85rem; padding: 4px; border: 1px solid #ccc; border-radius: 5px;">
                        {description}
                        </div>
                        """, unsafe_allow_html=True)
                        st.video(path)

                        # Display the model's prediction and true label
                        color = "green" if predicted == true else "red"
                        st.markdown(
                            f"<b style='color:{color};'>Model Prediction: {predicted} (confidence: {float(prob):.2f})</b>"
                            f"<br>**True Label: {true}**",
                            unsafe_allow_html=True)

                    # If the slot is empty, just show an empty space (to keep even spacing)
                    else:
                        st.empty()
        if len(selected) > 1:
            # Plot the distribution of predictions for the selected videos
            st.subheader("Prediction Distribution")
            fig = plot_prediction_distribution(df, selected)
            st.pyplot(fig, use_container_width=False)

import os
import pandas as pd
import streamlit as st

# Config & paths
DATA_DIR = "data/videos"
MODEL = "data/predictions/BalancedRF.csv"
GRID          = 6     # N columns in each grid
VID_W    = 180        # video width in pixels

st.set_page_config(page_title="Video model demo", layout="wide")

# global CSS for video width
st.markdown(
    f"<style> video {{width:{VID_W}px !important; height:auto;}} </style>",
    unsafe_allow_html=True,
)

# Load metadata and keep rows whose video file exists
@st.cache_data
def load_model_results(csv_path, video_dir):
    model_df = pd.read_csv(csv_path, dtype=str)
    model_df["file_path"] = model_df["video_id"].apply(lambda v: os.path.join(video_dir, f"{v}.mp4"))
    model_df = model_df[model_df["file_path"].apply(os.path.isfile)].reset_index(drop=True)
    return model_df

df = load_model_results(MODEL, DATA_DIR)
videos_ids = df["video_id"].tolist()

# Initialise session selection state for checkboxes
if "chosen" not in st.session_state:
    st.session_state["chosen"] = {vid: False for vid in videos_ids}

st.title("Select videos to show model's output")

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
if st.button("Show model output", type="primary", use_container_width=True):
    selected = [vid for vid, flag in st.session_state["chosen"].items() if flag]
    if not selected:
        st.warning("No videos selected.")
    else:
        st.subheader("Model output")
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
                        st.video(path)
                        st.text(
                            f"Model Prediction: {predicted}\n"
                            f"True Label: {true}"
                        )
                    # If the slot is empty, just show an empty space (to keep even spacing)
                    else:
                        st.empty()

# Video Affiliation Classifier – Web Demo
This repository contains a web-based interface built with Streamlit for visualizing the predictions of a multimodal video classification pipeline. 
It presents the predicted class of each video (e.g., affiliation label), as well as the model’s confidence score.

## Overview
This application loads preprocessed classification results and displays them in a user-friendly interface. It’s designed for easy sharing and demonstration of model performance.

## Project Structure
* app.py – Main application file.

* Procfile – For deployment on Heroku or similar platforms.

* requirements.txt – Python dependencies for the application.

### Directories:
* data/ –
  * apify_metadata/ - Original metadata of the videos used in the app, as downloaded from Apify TikTok scraper.
  * aux_features/ - Auxilliary models feature vectors of the videos as well as aux_features_metadata.json that can be used to visualize the features results.
  * predictions/ - .csv files containing the preprocessed classification results of each classifier.
  * videos/ - .mp4 files of the videos presented in the app.
  * video_frames/ - Frames extracted from each video in the videos directory.
* utils/ - Contains plot utils used in the app to plot the models' predictions results.

## Getting Started
1. Clone the repository.
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the app:
```
streamlit run app.py
```

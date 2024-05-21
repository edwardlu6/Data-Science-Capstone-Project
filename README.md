# Data Science Capstone Project

## Overview

This repository contains the code and documentation for my Data Science Capstone Project conducted from April 2024 to May 2024. The project investigates factors contributing to music popularity using a Spotify dataset of 52,000 songs. The analysis encompasses a range of statistical methods and machine learning models to uncover patterns and predict various aspects of music tracks.

## Project Description

The purpose of this capstone project is to tie together topics learned in the Data Science course. The project simulates working as a Data Scientist for Spotify, aiming to better understand what makes music popular and the audio features that define specific genres.

## Dataset

The dataset consists of data on 52,000 songs randomly picked from a variety of genres. It includes the following features:
- **songNumber**: Track ID of the song.
- **artist(s)**: The artist(s) who are credited with creating the song.
- **album_name**: The name of the album.
- **track_name**: The title of the specific track.
- **popularity**: An integer from 0 to 100 indicating the song's popularity.
- **duration**: The duration of the song in milliseconds.
- **explicit**: A binary variable indicating if the track contains explicit content.
- **danceability**: Quantifies how easy it is to dance to the song.
- **energy**: Quantifies the intensity of a song.
- **key**: The key of the song.
- **loudness**: Average loudness of a track in decibels.
- **mode**: Binary variable indicating if the song is in a major or minor key.
- **speechiness**: Quantifies the amount of spoken words in the song.
- **acousticness**: Indicates if the song features acoustic instruments.
- **instrumentalness**: Indicates the presence of vocals.
- **liveness**: Quantifies the likelihood of the recording being live.
- **valence**: Quantifies the positivity of the song's mood.
- **tempo**: The speed of the song in beats per minute.
- **time_signature**: Number of beats in a measure.
- **track_genre**: Genre assigned by Spotify.

## Analysis and Methodology

### Key Analyses

1. **Correlation Analysis**:
   - Calculated correlation between duration and popularity and visualized it using a scatterplot.
   
2. **Mann-Whitney U Test**:
   - Examined the association between popularity and explicit content, and popularity and major/minor key features.

3. **Linear Regression Models**:
   - Built and evaluated simple linear regression models for each feature to predict popularity, utilizing R-squared and RMSE as metrics.
   
4. **Multiple Regression Models**:
   - Trained and evaluated multiple regression models (including multiple linear regression, lasso regression, and elastic net) using all 10 features, assessing model performance with R-squared and RMSE.

5. **Principal Component Analysis (PCA)**:
   - Performed PCA to reduce the dataset's dimensionality and identify key components influencing song popularity.
   
6. **Logistic Regression Models**:
   - Developed logistic regression models using duration and principal components to predict whether a song is classical music, evaluated using AUC-ROC, Confusion Matrix, and Classification Report.

### Project Deliverables

1. **Project Report**:
   - A PDF document containing answers to the 10 key questions posed in the project, including descriptive text, figures, and statistical metrics.

2. **Code**:
   - A Python and JupyterNotebook file with the code that performed the data analysis and created the figures.

### Instructions for Reproducing the Analysis

1. **Setup**:
   - Clone the repository and navigate to the project directory.
   - Install the required packages using:
     ```bash
     pip install -r requirements.txt
     ```

2. **Data Preprocessing**:
   - Load the data using Pandas.
   - Handle missing data and perform necessary data transformations.

3. **Analysis**:
   - Run the provided Python scripts to perform the analysis and generate the results.

4. **Results**:
   - The results will be saved as figures and included in the project report.

## Conclusion

The project successfully applied various data science techniques to explore and model the factors influencing music popularity. The findings provide insights into the relationship between song features and their popularity, as well as the ability to classify songs based on these features.

## Repository Structure

- **spotify52kData.csv/**: Contains the dataset files.
- **notebooks/**: Jupyter notebooks used for the analysis.
- **scripts/**: Python scripts for data processing and model training.
- **Capstone Project Report/**: Generated figures and final report.
- **README.md**: Project documentation.
- **requirements.txt**: Required Python packages.

## Acknowledgements

This project was completed as part of the Data Science course (DS UA 112). The dataset and project guidelines were provided by the course instructors.

## License

This project is licensed under the MIT License.

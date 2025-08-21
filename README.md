# Hotstar Review Sentiment Analysis

This repository contains an end-to-end natural language processing pipeline for analyzing customer feedback from the Hotstar application. The dataset consists of approximately `5,000` manually labeled reviews categorized as `Positive`, `Neutral`, or `Negative`, making it a three-class supervised learning problem. The project demonstrates how unstructured text data can be transformed into structured insights that support product improvement, customer experience monitoring, and business decision-making. The workflow begins with raw text preprocessing, where reviews are converted to lowercase, cleaned using `regex` patterns to remove user mentions, URLs, numerics, and special characters, and then tokenized. The text is further refined using `NLTK stopwords` removal and lemmatization via `WordNetLemmatizer`, ensuring that the vocabulary is normalized and consistent. Punctuation and hashtags are stripped, and the resulting cleaned text is stored for feature extraction. 

Feature engineering is performed using `TF-IDF` with parameters such as `ngram_range=(1,2)`, `min_df=2`, `max_df=0.9`, and `sublinear_tf=True`, capturing both unigrams and bigrams while reducing the effect of overly rare or frequent tokens. To handle the high dimensionality of TF-IDF vectors, dimensionality reduction is applied using `TruncatedSVD` with `n_components=300` and `random_state=42`, resulting in dense semantic representations suitable for deep learning models. The reduced feature matrix is split into training and testing sets with an `80/20` ratio, producing around `4,042` training samples and `1,011` test samples. 

The classification model is a feedforward deep neural network built with `Keras Sequential API`. The architecture includes an input layer followed by dense layers of size `128` and `64`, each followed by `BatchNormalization`, `LeakyReLU` activation, and `Dropout` layers (`0.5` and `0.3` respectively) to prevent overfitting. The output layer is a softmax layer producing three probabilities corresponding to the sentiment classes. The optimizer used is `Adam` with learning rate `3e-4`, and training is enhanced with callbacks such as `EarlyStopping` and `ReduceLROnPlateau` to avoid unnecessary epochs and dynamically adjust learning rate. The model achieved a test accuracy of approximately `73%`, with balanced performance across positive, neutral, and negative classes as evaluated using a classification report and confusion matrix. 

Artifacts generated during training include `tfidf_vectorizer.pkl` for text vectorization, `svd_reducer.pkl` for dimensionality reduction, and `deep_learning_model.h5` containing the trained model weights and architecture. These artifacts are essential for deployment, as they allow the exact same transformations to be applied to new input data. The deployment layer is implemented using `Gradio`, which provides a simple and interactive web interface. The application allows users to input a Hotstar review, after which the text is preprocessed, transformed, and classified into its corresponding sentiment with a confidence score. This makes the system suitable for real-time inference and easy to demonstrate. The live application is deployed on Hugging Face Spaces at [Hotstar Review Sentiment](https://huggingface.co/spaces/Sourav-003/Hotstar-Review-Sentiment), providing public access to the model’s predictions. 

The broader motivation of the project lies in demonstrating the value of text analytics for customer experience management. By automatically classifying customer reviews, businesses like Hotstar can identify service pain points, measure satisfaction levels, and prioritize enhancements. Positive feedback clusters may guide marketing campaigns, while negative clusters highlight critical bugs or performance issues. Neutral reviews offer opportunities to convert indifferent customers into loyal ones. Thus, the project connects machine learning workflows to business impact, bridging the gap between data science and strategy. 

## Tech Stack
`Python` `pandas` `numpy` `scikit-learn` `NLTK` `TensorFlow/Keras` `Gradio`  

## Repository Structure
- `app.py` : The Gradio application script that loads artifacts and exposes a web interface.  
- `requirements.txt` : Project dependencies required to reproduce the environment.  
- `tfidf_vectorizer.pkl` : Serialized TF-IDF vectorizer for text feature extraction.  
- `svd_reducer.pkl` : Serialized TruncatedSVD reducer for dimensionality reduction.  
- `deep_learning_model.h5` : Trained deep neural network model file.  
- `Hotstar-Review-Sentiment.ipynb` : Jupyter notebook documenting EDA, preprocessing, model training, and evaluation.  

## Usage
To run the project locally, clone the repository and install dependencies using `pip install -r requirements.txt`. Ensure that the files `tfidf_vectorizer.pkl`, `svd_reducer.pkl`, and `deep_learning_model.h5` are present in the working directory. You can start the interface by executing `python app.py`, which launches the Gradio application at `http://localhost:7860`. Input any Hotstar review into the provided text box, and the application will output the predicted sentiment along with its confidence score.  

## Live Demo
Access the deployed version here: [Hotstar Review Sentiment on Hugging Face](https://huggingface.co/spaces/Sourav-003/Hotstar-Review-Sentiment)  

---

Special thanks to `Kumar Sundram` and `Learnbay` for guidance and mentorship in shaping this project and helping translate technical work into impactful outcomes.  

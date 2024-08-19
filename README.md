# Joke Classification Into Humor Styles

This project is a web application that classifies jokes into different humor styles using machine learning models. It allows users to input a joke and get a classification along with explanations and encouragement based on the humor style detected.

## Features

- **Humor Style Classification**: The app classifies jokes into one of the following humor styles:
  - Self-Enhancing
  - Self-Deprecating
  - Affiliative
  - Aggressive
  - Neutral

- **Detailed Feedback**: After classification, the app provides an explanation of the detected humor style and offers encouragement or advice based on the result.

- **Probability Display**: The app also shows the model's confidence (probability) in the prediction.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8 or higher
- `pip` (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/MaryKenneth/humor_style_classifier_app.git
   cd joke-classification
   ```

2. **Install the Required Packages**

   Install the dependencies using the provided `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Model**

   Ensure you have the trained XGBoost model file (`ali_xgboost_humour_model.pkl`) in the root directory of the project.

### Running the Application

Start the Flask web application:

```bash
python app.py
```

The app will be available at `http://127.0.0.1:5000/` in your web browser.

### Usage

1. Go to the homepage of the web app.
2. Enter your joke in the text box.
3. Submit the joke to see its humor style classification along with an explanation and encouragement.
4. The prediction probability will also be displayed to indicate how confident the model is in its prediction.

## Project Structure

humor_style_classifier_app/
├── app.py                     # The main Flask application file
├── templates/
│   └── index.html             # HTML template for the homepage
├── ali_xgboost_humour_model.pkl # Pre-trained XGBoost model
├── requirements.txt           # Lists all the dependencies needed to run the project
└── README.md                  # This README file


## Technologies Used

- **Flask**: For building the web application.
- **XGBoost**: For training the humor classification model.
- **Sentence Transformers**: For embedding the text input.

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

- [XGBoost](https://xgboost.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)
- Flask documentation

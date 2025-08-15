# ArogyaAI
Project Summary: Arogya AI - Personal Health Assistant
Description:
Arogya AI is a Streamlit-based web application designed to assist users in monitoring their health by providing diabetes risk predictions, early disease detection, personalized diet suggestions, and health insights. The application leverages machine learning and data visualization to deliver actionable insights based on user-provided health data.

Key Features:
Diabetes Risk Prediction:

Predicts the likelihood of diabetes using a trained Random Forest Classifier.
Displays results with clear visual indicators (e.g., success or warning messages).
Early Disease Detection:

Identifies potential health risks based on user input (e.g., high glucose, BMI, blood pressure).
Provides warnings and recommendations for early intervention.
Smart Diet Suggestions:

Offers personalized diet tips based on glucose levels and BMI.
Encourages healthy eating habits and physical activity.
Health Insights Dashboard:

Visualizes health data trends using interactive charts and graphs.
Includes histograms, scatter plots, correlation heatmaps, and age-group-based glucose trends.
Health History Tracking:

Saves user input to a CSV file for tracking historical health data.
Displays the last 10 records for user reference.
Tech Stack:
Frontend:

Streamlit: For building the interactive web application.
Plotly: For creating interactive visualizations.
Matplotlib & Seaborn: For static data visualizations.
Backend:

Python: Core programming language for logic and data processing.
Pandas: For data manipulation and analysis.
NumPy: For numerical computations.
Machine Learning:

Scikit-learn: Used to train a Random Forest Classifier for diabetes prediction.
Joblib: For saving and loading the trained ML model.
Data Storage:

CSV Files: Used for storing health history and health data.
Visualization Libraries:

Plotly Express: For interactive charts.
Seaborn: For heatmaps and advanced visualizations.
Workflow:
User Input:

Users provide health details (e.g., glucose, BMI, age) via the sidebar.
Inputs are processed and stored in a DataFrame.
Prediction:

The trained Random Forest model predicts diabetes risk based on user input.
Insights:

Early disease risks are detected based on thresholds.
Diet suggestions are generated based on glucose and BMI levels.
Visualization:

Health data trends are visualized using interactive and static charts.
History:

User inputs are saved to a CSV file for future reference.
The last 10 records are displayed in the "History" tab.
Challenges Solved:
Data Persistence: Implemented CSV-based storage for user history.
Interactive Visualizations: Used Plotly and Seaborn for dynamic and static charts.
User-Friendly Interface: Designed an intuitive UI with Streamlit for non-technical users.
Health Insights: Provided actionable insights based on user data.
Tools and Libraries:
Streamlit: For building the web app.
Scikit-learn: For machine learning model training.
Pandas & NumPy: For data handling and preprocessing.
Matplotlib, Seaborn, Plotly: For data visualization.
Joblib: For model serialization.
OS & Datetime: For file handling and timestamping.
Deployment:
The application can be deployed on platforms like Streamlit Cloud, Heroku, or AWS for public access.
Key Achievements:
Built a fully functional health assistant application.
Integrated machine learning for real-time predictions.
Designed a user-friendly interface with interactive visualizations.
Enabled health history tracking for better user engagement.
Potential Use Cases:
Personal health monitoring.
Early detection of diabetes and related health risks.
Educational tool for promoting healthy habits.
Data-driven insights for healthcare professionals.

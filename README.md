# Edunet Capstone Project - SII Prediction System

This project is a complete machine learning application that predicts **Problematic Internet Use (SII)** based on various physical and demographic indicators. It consists of a Machine Learning model, a FastAPI backend, and a Next.js frontend.

## Project Structure

- **model/**: Contains the Jupyter notebook for training the model, the dataset, and the generated model artifacts (`.pkl` files).
- **backend/**: A Python FastAPI application that serves the trained model via a REST API.
- **edunet-capstone-project/**: A Next.js (React) frontend application for user interaction.

## Prerequisites

- **Python 3.8+** (for Model and Backend)
- **Node.js 18+** (for Frontend)
- **Git**

---

## 1. Model Setup

Before running the application, you need to train the model and generate the necessary artifacts (`model.pkl`, `mappings.pkl`, `thresholds.pkl`, `features.pkl`).

1.  Navigate to the `model/` directory.
2.  Ensure needed libraries are installed. You may need packages like `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `joblib`.
3.  Open the **`edunet-project-tuning-model.ipynb`** notebook.
4.  **Run all cells**.
    *   This will train the ensemble model.
    *   It will save the artifacts in the `model/` folder.
    *   **Crucial:** Ensure the last cell runs successfully to save `features.pkl`.

---

## 2. Backend Setup (FastAPI)

The backend handles the prediction logic.

1.  Open a terminal and navigate to the `backend/` directory:
    ```bash
    cd backend
    ```

2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4.  Start the FastAPI server:
    ```bash
    python main.py
    ```
    *   The server will start at `http://localhost:8000`.
    *   It should print "Model and artifacts loaded successfully".

---

## 3. Frontend Setup (Next.js)

The frontend provides the user interface.

1.  Open a new terminal and navigate to the `edunet-capstone-project/` directory:
    ```bash
    cd edunet-capstone-project
    ```

2.  Install dependencies:
    ```bash
    npm install
    ```

3.  Start the development server:
    ```bash
    npm run dev
    ```

4.  Open your browser and visit:
    ```
    http://localhost:3000
    ```

## Usage

1.  Ensure both Backend (`port 8000`) and Frontend (`port 3000`) are running.
2.  On the web page, enter the **Age**, **Sex**, and any **Additional Features** in JSON format.
3.  Click **Predict**.
4.  The system will display the Raw Prediction score and the Classification Class.

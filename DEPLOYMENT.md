# Deployment Guide: AI-Based Student Dropout Prediction System

Your project consists of two separate components that need to be deployed independently:
1. **The Backend (Flask API)** located in the `api/` folder.
2. **The Frontend (Streamlit Dashboard)** located in the `dashboard/` folder.

Below is the step-by-step guidance to deploy both to popular, free-tier friendly platforms (Render and Streamlit Community Cloud).

---

## Part 1: Deploy the Backend API (on Render)

You already have a `render.yaml` file properly configured for this!

1. **Push your code to GitHub**: 
   Make sure your entire `final_project` is pushed to a repository on GitHub.
   
2. **Create a Render Account**: 
   Go to [Render.com](https://render.com/) and sign up using your GitHub account.

3. **Deploy using the Blueprint**:
   - In the Render Dashboard, click the **"New"** button and select **"Blueprint"**.
   - Connect your GitHub account and select your `final_project` repository.
   - Render will automatically detect the `render.yaml` file in your repository.
   - Click **"Apply Blueprint"**. 
   - Render will start building your Flask API. It will install packages from `api/requirements.txt` and launch the app using `gunicorn`.
   
4. **Get your API URL**:
   Once the deployment is marked as "Live", Render will give you a public URL (e.g., `https://student-dropout-api.onrender.com`). **Copy this URL**, you will need it for the frontend.

---

## Part 2: Deploy the Frontend Dashboard (on Streamlit Community Cloud)

Streamlit provides a free cloud hosting service explicitly for Streamlit apps.

1. **Sign up for Streamlit Cloud**:
   Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.

2. **Create a New App**:
   - Click **"New app"** -> **"Deploy a modern app (GitHub)"** (or similar option to deploy from an existing repo).
   - **Repository**: Select your `final_project` repository.
   - **Branch**: `main` (or whatever your default branch is).
   - **Main file path**: Enter `dashboard/app.py`.

3. **Configure Environment Secrets (Crucial Step!)**:
   Your dashboard needs to know where the backend API lives. We handle this via `st.secrets`.
   - Before clicking "Deploy", look for **"Advanced settings"** or **"Secrets"**.
   - In the secrets text box (TOML format), paste the following:
     ```toml
     BACKEND_URL = "https://your-render-url.onrender.com"
     ```
     *(Substitute the generic URL with the actual URL you got from Render in Part 1.)*
   
4. **Deploy**:
   - Click **"Deploy!"**.
   - Streamlit will read the `dashboard/requirements.txt` implicitly (if it can't find it, it might read the root `requirements.txt` instead, which is also fine since we ensured it has the needed dashboard packages).
   
5. **Verify**:
   Once Streamlit finishes booting up, it will provide you a link to your live dashboard. Open it and ensure that the "Sign In" and features successfully communicate with your backend API.

---

## Troubleshooting

- **Large Memory Issues (Backend):** You are using relatively heavy libraries (`transformers`, `torch`, `catboost`). If Render's free tier runs out of memory (RAM limit is 512MB), you may need to either:
  1. Upgrade to a paid instance on Render.
  2. Optimize your models to use lighter alternatives or remove heavy natural language frameworks if they exceed limits.
  
- **API Connection Errors on Dashboard:** If the dashboard says "Connection Error", double-check that your `BACKEND_URL` in the Streamlit secrets does **not** have a trailing slash (`/`), and that your Render API is awake. (Render free tiers spin down after 15 minutes of inactivity and take ~50 seconds to wake up).

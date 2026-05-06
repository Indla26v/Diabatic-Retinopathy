# RetinaAI: Diabetic Retinopathy Diagnostic System

An advanced AI-powered web application for detecting and classifying Diabetic Retinopathy (DR) from high-resolution fundus images. The system uses a Hybrid CNN-Transformer model (EfficientNet-B3 + Transformer) for high accuracy, providing CLAHE enhanced imaging and Grad-CAM lesion localization heatmaps.

## 🚀 How to Start the Application

### Option 1: Quick Start (Windows)
The easiest way to start both the frontend and backend simultaneously is using the provided PowerShell script:
1. Open PowerShell in the root directory (`SDP/`).
2. Run the following command:
   ```powershell
   .\start.ps1
   ```
   *Note: If you encounter an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first.*

### Option 2: Manual Start

**1. Start the Backend (FastAPI)**
1. Open a terminal in the root directory.
2. Activate the virtual environment:
   ```powershell
   # Windows
   .\.venv\Scripts\activate
   ```
3. Start the Uvicorn server:
   ```bash
   python -m uvicorn api:app --reload --port 8000
   ```
   *The backend API will run at `http://127.0.0.1:8000`*

**2. Start the Frontend (Next.js)**
1. Open a new terminal instance.
2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
3. Install dependencies (if you haven't already):
   ```bash
   npm install
   ```
4. Start the development server:
   ```bash
   npm run dev
   ```
   *The frontend UI will run at `http://localhost:3000`*

---

## 🖥️ How to Use the Application

1. **Open the Dashboard**: Once both servers are running, open your web browser and go to `http://localhost:3000`.
2. **Upload a Scan**: In the left "Workspace" panel, click the upload area to browse for a fundus image (or drag and drop a structural eye scan like a `.jpg` or `.png`).
3. **Run Analysis**: Click the **"Analyze Scan"** button below the image.
4. **View the Timeline**: 
   - Wait briefly as the AI pipeline triggers and simulates the step-by-step diagnostic workflow.
   - **Step 1**: It will first display the **CLAHE Enhanced Image** (contrast-limited adaptive histogram equalization) on the right side.
   - **Step 2**: It will generate and reveal the **Grad-CAM Heatmap**, highlighting potential lesion areas in red/orange ("hot spots").
   - **Step 3**: Finally, the system will unveil the full **Diagnostic Report**, showing:
     - The **Primary Indication** (e.g., No DR, Mild, Moderate, Severe, Proliferative DR).
     - The **Confidence Score** of the primary diagnosis.
     - A collapsible **Probability Distribution** showing the exact likelihood percentages across all 5 clinical stages.
5. **Inspect Closely**: Hover over either the CLAHE or Grad-CAM images and click to **Expand** them. This opens an interactive modal where you can zoom and pan around the high-resolution layers.
6. **Reset / New Scan**: Click the "Clear" button in the Workspace to clear the current analysis and begin processing a new patient scan.

---

### Important Clinical Advisory
This software is intended for screening and educational purposes. The AI analysis does not replace a comprehensive ophthalmic examination and all predictions must be verified by a certified healthcare professional.
# Project1: LangChain AI Project

This project contains a starter template for Python and LangChain development.

## Project Structure

```text
Project1/
├── .venv/                 # Python Virtual Environment (ignored by Git)
├── src/
│   ├── __init__.py        # Makes src folder a package
│   └── main.py            # Main entry point / example code
├── tests/
│   ├── __init__.py        # Makes tests folder a package
│   └── test_main.py       # Basic pytest unit tests
├── .env.example           # Example environment file template
├── .gitignore             # Files and directories to ignore in Git
├── README.md              # Project documentation (this file)
└── requirements.txt       # Python package dependencies
```

## Setup Instructions

### 1. Create a Python Virtual Environment
Run the following command in the project root:
```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
Make sure your virtual environment is active, then run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Copy `.env.example` to `.env` and fill in your API key:
```bash
copy .env.example .env
```
*(On macOS/Linux use: `cp .env.example .env`)*

Open `.env` and set your `OPENAI_API_KEY`.

### 5. Run Tests
You can verify that the setup works by running:
```bash
pytest
```

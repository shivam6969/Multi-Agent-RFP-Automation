# RFP Information Extractor & Automated Agent

This application provides both manual and automated RFP (Request for Proposal) processing with intelligent information extraction and workflow management.

## Features

- **🤖 Automated RFP Agent**: Automatically searches GEM.gov.in for RFPs and extracts information
- **📄 PDF Text Extraction**: Automatically extracts text from RFP PDF documents
- **🎯 Intelligent Analysis**: Identifies and categorizes key information including:
  - Deadlines and submission dates
  - Contact information
  - Requirements and specifications
  - Financial information (budget, estimated costs)
  - Bid details and tender numbers
- **📋 Session Cards**: Interactive cards showing RFP information with workflow actions
- **🔄 Workflow Management**: "Move Forward" and "Hold" actions for RFP processing
- **💾 Data Persistence**: Saves all analyses and session data for later review

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: 🤖 Automated RFP Agent (Recommended)

The intelligent agent automatically discovers RFPs and presents them as interactive session cards.

```bash
streamlit run rfp_agent.py
# or
python -m streamlit run rfp_agent.py
```

**Features:**
- 🤖 **Automated Discovery**: Searches GEM.gov.in using electrical keywords
- 📋 **Session Cards**: Each RFP displayed as a card with key information
- 🚀 **Move Forward**: Mark RFPs for next workflow step
- ⏸️ **Hold**: Keep RFPs for later review
- 📊 **Session Tracking**: Complete workflow history and statistics

**Workflow:**
1. Click "🚀 Start New Session"
2. Agent searches for RFPs automatically
3. Review each RFP card
4. Click "Move Forward" or "Hold" for each RFP
5. "Move Forward" RFPs proceed to next processing steps

### Option 2: Streamlit Manual Interface

Manual PDF URL input with interactive analysis.

```bash
streamlit run streamlit_app.py
# or
python -m streamlit run streamlit_app.py
```

### Option 3: Flask Web Interface

Traditional web application with custom HTML templates.

```bash
python app.py
```

### Option 4: Automated Extraction Script

Command-line automated extraction:

```bash
python rfp_extractor.py
```

## File Structure

```
PRJ3/
├── rfp_agent.py          # 🤖 Automated RFP Agent (Main)
├── streamlit_app.py      # Manual Streamlit interface
├── app.py                # Flask web application
├── rfp_extractor.py      # Automated extraction script
├── z.py                  # Original download script (modified)
├── y.py                  # Another scraper script
├── config.py            # Configuration settings
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates (Flask)
│   ├── index.html        # Main extraction interface
│   └── saved.html        # Saved analyses viewer
├── sessions/            # 📋 Session data storage
└── extracted_data/      # 💾 Extracted information storage
```

## Key Components

### 🤖 Automated RFP Agent
- **Auto-Discovery**: Searches using electrical equipment keywords
- **Session Management**: Tracks multiple discovery sessions
- **Card-Based UI**: Interactive cards for each RFP
- **Workflow Actions**: Move Forward / Hold functionality
- **Progress Tracking**: Real-time scraping progress

### PDF Text Extraction
Uses `pdfplumber` to extract text from PDF documents with high accuracy.

### Information Analysis
The system uses regex patterns to identify:
- **Deadlines**: Due dates, submission dates, closing dates
- **Contacts**: Email addresses, phone numbers, contact persons
- **Requirements**: Technical specifications, eligibility criteria
- **Financial Info**: Budget amounts, estimated costs
- **Bid Details**: Tender numbers, procurement types

## Session Cards

Each RFP is displayed as an interactive card containing:

- **📄 RFP Title**: Bid name and identifier
- **🏷️ Status**: Pending / Move Forward / Hold
- **📅 Deadlines**: Key submission dates
- **📞 Contacts**: Contact information
- **📋 Requirements**: Technical specifications
- **💰 Financial**: Budget and cost information
- **🚀 Actions**: Move Forward and Hold buttons

## Workflow Management

### Move Forward RFPs
- Marked for immediate processing
- Can proceed to next workflow steps
- Integrated with downstream systems

### Hold RFPs
- Kept for later review
- Can be revisited in future sessions
- Maintains all extracted information

## Data Storage

- **Sessions**: Complete session data in `sessions/` directory
- **Analyses**: Individual RFP analyses in `extracted_data/` directory
- **JSON Format**: All data stored as structured JSON for easy processing

## Example Usage

### Automated Agent Workflow:
1. Launch: `streamlit run rfp_agent.py`
2. Click "🚀 Start New Session"
3. Wait for agent to discover RFPs
4. Review each RFP card
5. Click "Move Forward" for urgent RFPs
6. Click "Hold" for RFPs needing review
7. Access session history in sidebar

### Manual Processing:
1. Get RFP PDF URL from GEM.gov.in
2. Paste URL into interface
3. View categorized information
4. Save for future reference

## Notes

- The automated agent processes electrical equipment keywords
- PDFs are downloaded temporarily for extraction then deleted
- Session data persists across application restarts
- All analyses include timestamps for tracking
- Streamlit interfaces run on localhost:8501-8503
```

Then open `http://127.0.0.1:5000` in your browser.

### Option 3: Automated Extraction

Run the automated scraper that searches GEM.gov.in for electrical equipment RFPs:

```bash
python rfp_extractor.py
```

This will:
- Search for electrical keywords on GEM bidding site
- Extract information from found RFP PDFs
- Save analyses to the `extracted_data/` directory
- Display results in the web interface

## File Structure

```
PRJ3/
├── app.py                 # Flask web application
├── streamlit_app.py       # Streamlit web application (recommended)
├── rfp_extractor.py       # Automated RFP extraction script
├── z.py                   # Original download script (modified)
├── y.py                   # Another scraper script
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates (for Flask)
│   ├── index.html        # Main extraction interface
│   └── saved.html        # Saved analyses viewer
└── extracted_data/       # Extracted information storage
```

## Key Components

### PDF Text Extraction
Uses `pdfplumber` to extract text from PDF documents with high accuracy.

### Information Analysis
The system uses regex patterns to identify:
- **Deadlines**: Due dates, submission dates, closing dates
- **Contacts**: Email addresses, phone numbers, contact persons
- **Requirements**: Technical specifications, eligibility criteria
- **Financial Info**: Budget amounts, estimated costs
- **Bid Details**: Tender numbers, procurement types

### Web Interfaces
- **Streamlit**: Modern, interactive, no HTML required
- **Flask**: Traditional web app with custom templates

## API Endpoints (Flask only)

- `GET /`: Main extraction interface
- `POST /extract`: Extract information from PDF URL
- `GET /saved`: View saved analyses

## Example Usage

1. Find an RFP PDF URL from GEM.gov.in
2. Paste the URL into the web interface
3. Click "Extract Information" (Streamlit) or "🚀 Extract Information" (Flask)
4. Review the categorized information:
   - Deadlines for submission
   - Contact details for clarification
   - Technical requirements
   - Budget information

## Data Storage

All extracted information is saved as JSON files in the `extracted_data/` directory for future reference and analysis.

## Notes

- The automated scraper processes electrical equipment keywords
- PDFs are downloaded temporarily for extraction then deleted
- Streamlit runs on localhost:8501, Flask on localhost:5000
- All analyses are saved with timestamps for tracking
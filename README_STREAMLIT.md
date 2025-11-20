# Invoice Import Tool - Streamlit Version

A web-based application for importing invoice data from Excel or PDF files and exporting them to a standardized Excel format.

## Features

- **Web-based interface** - No installation needed, runs in your browser
- **Multi-format support** - Reads both Excel (.xlsx, .xls) and PDF files
- **Automatic column detection** - Intelligently identifies vendor style, quantity, cost, and description
- **Live preview** - See extracted data before downloading
- **One-click download** - Export to Excel with timestamped filename
- **Cloud deployable** - Can be deployed to Streamlit Cloud, Heroku, or other platforms

## Local Installation

1. Install Python 3.8 or higher

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Deployment to Streamlit Cloud

### Step 1: Prepare Your Repository

1. Create a GitHub repository
2. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `README_STREAMLIT.md` (optional)

### Step 2: Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set main file path to `app.py`
6. Click "Deploy"

Your app will be live at: `https://[your-app-name].streamlit.app`

## Deployment to Other Platforms

### Heroku

Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT
```

Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t invoice-import .
docker run -p 8501:8501 invoice-import
```

## Usage

1. **Upload File**: Click "Browse files" and select your Excel or PDF invoice
2. **Process**: Click the "Process File" button
3. **Review**: Check the extracted data in the preview table
4. **Download**: Click "Download Excel File" to save the results

## Output Format

The exported Excel file contains:
| Vendor Style # | Quantity | Cost | Description |
|---------------|----------|------|-------------|
| ABC-123 | 10 | 25.00 | MEN ECO WR100 SSG STRA |
| XYZ-456 | 5 | 50.00 | LAD ECO WR100 STRG BRAC |

## Supported Input Formats

### Excel Files
- Must have table-like structure with headers
- Supported extensions: .xlsx, .xls
- Automatically detects header row

### PDF Files
- Structured tables are extracted automatically
- Text-based invoices are parsed using pattern matching
- Multi-page PDFs are supported

## Column Detection

The application looks for these keywords:
- **Style/SKU**: style, sku, item, product, part, number, code, article, model, ref
- **Quantity**: qty, quantity, amount, count, units, ordered, pieces, pcs
- **Cost**: cost, price, rate, total, value, unit, charge
- **Description**: desc, description, name, detail, title, specification

## Troubleshooting

**No data extracted:**
- Check that your file has a table structure
- Ensure column headers contain recognizable keywords
- For PDFs, verify the file is not image-based (scanned)

**Wrong columns detected:**
- The app shows which columns it matched
- Check the "Processing Details" section for diagnostics

**App won't start:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Clear Streamlit cache
streamlit cache clear
```

## Environment Variables (Optional)

For Streamlit Cloud deployment, you can set these in the app settings:

- `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` - Max file size in MB (default: 200)
- `STREAMLIT_SERVER_ENABLE_CORS` - Enable CORS (default: true)

## Security Notes

- Files are processed in memory and not stored permanently
- No data is sent to external services
- All processing happens on the server where the app is deployed

## License

This is a proprietary tool for internal use.

## Support

For issues or questions, please contact your system administrator.

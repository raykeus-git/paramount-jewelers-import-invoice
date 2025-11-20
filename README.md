# Invoice Import Tool

A simple desktop application for importing invoice data from Excel or PDF files and exporting them to a standardized Excel format.

## Features

- **Multi-format Support**: Reads both Excel (.xlsx, .xls) and PDF files
- **Automatic Column Detection**: Intelligently identifies vendor style, quantity, and cost columns
- **User-friendly GUI**: Simple interface with file selection and progress tracking
- **Standardized Output**: Exports to Excel with columns:
  - Vendor Style #
  - Quantity
  - Cost
  - Description (auto-filled with "New SKU, please add description.")

## Installation

1. Install Python 3.8 or higher

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Note: `tkinter` comes pre-installed with most Python distributions. If you encounter issues, install it separately based on your OS.

## Usage

1. Run the application:
```bash
python import.py
```

2. Click "Select File" and choose your invoice file (Excel or PDF)

3. Click "Process & Export" to extract the data

4. Choose where to save the output Excel file

5. The application will create a formatted Excel file with the extracted invoice data

## How It Works

### Excel Files
The application scans Excel columns for keywords like:
- **Style/SKU**: style, sku, item, product
- **Quantity**: qty, quantity, amount
- **Cost**: cost, price, amount, total

### PDF Files
The application extracts tables from PDF pages and identifies columns using similar keyword matching.

## Output Format

The exported Excel file contains:
| Vendor Style # | Quantity | Cost | Description |
|---------------|----------|------|-------------|
| ABC-123 | 10 | 25.00 | New SKU, please add description. |
| XYZ-456 | 5 | 50.00 | New SKU, please add description. |

## Troubleshooting

**Import errors**: Make sure all dependencies are installed:
```bash
pip install pandas openpyxl PyPDF2 pdfplumber
```

**PDF reading issues**: Some PDFs with complex formatting may not extract correctly. Try converting to Excel first for best results.

**No data extracted**: Ensure your source file has table-like structure with headers that include keywords for style, quantity, and cost.

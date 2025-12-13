import streamlit as st
import pandas as pd
import pdfplumber
import os
from datetime import datetime
import re
from io import BytesIO

# OCR imports (optional, for scanned PDFs)
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class InvoiceImporter:
    def __init__(self, vendor=None):
        self.vendor = vendor

        # Vendor-specific filter configurations
        self.vendor_filters = {
            'YGI': {
                'skip_patterns': [
                    # Amazon-style ASINs: Single letter, then digit at position 2
                    # Matches: B0742KDBYR (B+0...), X002NYIYP5 (X+0...)
                    # Does NOT match: FME4176Y-10 (F+M...), FSE11947Y26D (F+S...)
                    r'^[A-Z]\d',
                    r'^U\d{6,}',         # U-codes (U353929...)
                    r'^\d[A-Z0-9]{2,}$', # Digit followed by 2+ chars (3G4U, 1A2B)
                ],
                'skip_exact': [
                    'X002NYCJ47',  # Specific codes to filter
                    'X002NYIYP5',  # Additional specific code
                    '3G4U',        # Short alphanumeric code
                ],
                'description': 'YGI vendor - filters B0/X00/U-codes and short alt codes'
            },
            'DNG': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'DNG vendor - no filters configured'
            },
            'Rolex': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'Rolex vendor - no filters configured'
            },
            'Seiko': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'Seiko vendor - no filters configured'
            },
            'Citizen': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'Citizen vendor - no filters configured'
            },
            'Bulova': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'Bulova vendor - no filters configured'
            },
            'Casio': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'Casio vendor - no filters configured'
            },
            'IDD': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'IDD USA vendor - no filters configured'
            },
            'Invicta': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'Invicta vendor - no filters configured'
            },
            'Zenith': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'Zenith vendor - no filters configured'
            },
            'Jewelry Depo': {
                'skip_patterns': [],     # No filters yet - add as needed
                'skip_exact': [],
                'description': 'Jewelry Depo vendor - no filters configured'
            },
        }

    def should_skip_vendor_style(self, vendor_style):
        """Check if vendor style should be skipped based on vendor-specific filters"""
        if not self.vendor or self.vendor not in self.vendor_filters:
            # No vendor selected or no filters for this vendor
            return False

        # Ensure vendor_style is a string
        if not vendor_style or not isinstance(vendor_style, str):
            return False

        filters = self.vendor_filters[self.vendor]
        vendor_style_upper = vendor_style.upper()

        # Check exact matches first
        if 'skip_exact' in filters:
            if vendor_style_upper in [s.upper() for s in filters['skip_exact']]:
                # if self.vendor == 'YGI':
                #     st.info(f"🔍 YGI Debug - Exact match filter triggered for: '{vendor_style}'")
                return True

        # Check pattern matches
        if 'skip_patterns' in filters:
            for pattern in filters['skip_patterns']:
                if re.match(pattern, vendor_style, re.IGNORECASE):
                    # if self.vendor == 'YGI':
                    #     st.info(f"🔍 YGI Debug - Pattern '{pattern}' matched: '{vendor_style}' - SKIPPING")
                    return True

        # If we get here, no patterns matched
        # if self.vendor == 'YGI':
        #     st.info(f"🔍 YGI Debug - No filters matched: '{vendor_style}' - KEEPING")
        return False

    def find_best_column_match(self, columns, keywords, column_type):
        """Find the best matching column using fuzzy matching and scoring"""
        best_match = None
        best_score = 0

        for col in columns:
            col_lower = str(col).lower()
            col_clean = re.sub(r'[^a-z0-9]', '', col_lower)
            score = 0

            for keyword in keywords:
                if keyword in col_lower:
                    score += 10
                if keyword == col_clean:
                    score += 20
                if col_lower.startswith(keyword) or col_lower.endswith(keyword):
                    score += 5

            if score > best_score:
                best_score = score
                best_match = col

        return best_match

    def smart_column_detection(self, df):
        """Intelligently detect vendor style, quantity, cost, and description columns"""

        style_keywords = ['style', 'sku', 'item', 'product', 'prod', 'part', 'number',
                         'code', 'article', 'model', 'ref', 'reference', 'material']
        qty_keywords = ['qty', 'quantity', 'amount', 'count', 'units', 'ord', 'ordered',
                       'pieces', 'pcs', 'pack', 'carton']
        cost_keywords = ['cost', 'price', 'rate', 'total', 'value', 'amount', 'unit',
                        'unitprice', 'each', 'perpcs', 'charge']
        desc_keywords = ['desc', 'description', 'name', 'detail', 'title', 'specification']

        style_col = self.find_best_column_match(df.columns, style_keywords, "Style")
        qty_col = self.find_best_column_match(df.columns, qty_keywords, "Quantity")
        cost_col = self.find_best_column_match(df.columns, cost_keywords, "Cost")
        desc_col = self.find_best_column_match(df.columns, desc_keywords, "Description")

        if not style_col and not qty_col and not cost_col:
            for col in df.columns:
                sample_values = df[col].dropna().head(10)

                has_alphanumeric = any(bool(re.search(r'[a-zA-Z]', str(v))) and
                                      bool(re.search(r'\d', str(v))) for v in sample_values)
                has_numeric = all(str(v).replace('.', '').replace(',', '').replace('-', '').isdigit()
                                 for v in sample_values if str(v).strip())

                if not style_col and has_alphanumeric and len(sample_values) > 0:
                    style_col = col
                elif not qty_col and has_numeric and not style_col == col:
                    if all(float(str(v).replace(',', '')) < 10000 for v in sample_values if str(v).strip()):
                        qty_col = col
                elif not cost_col and has_numeric and col != qty_col:
                    cost_col = col

        return style_col, qty_col, cost_col, desc_col

    def read_excel(self, file):
        try:
            df_raw = pd.read_excel(file, header=None)

            header_row = None
            for idx in range(min(15, len(df_raw))):
                row_values = [str(v).lower() for v in df_raw.iloc[idx] if pd.notna(v)]
                if len(row_values) < 2:
                    continue

                non_numeric_count = sum(1 for v in row_values if not str(v).replace('.', '').replace(',', '').isdigit())

                if non_numeric_count >= 2:
                    header_row = idx
                    break

            if header_row is not None:
                df = pd.read_excel(file, header=header_row)
            else:
                df = pd.read_excel(file)

            style_col, qty_col, cost_col, desc_col = self.smart_column_detection(df)

            data = []
            for idx, row in df.iterrows():
                vendor_style = str(row[style_col]).strip() if style_col and pd.notna(row[style_col]) else ""
                quantity = str(row[qty_col]).strip() if qty_col and pd.notna(row[qty_col]) else ""
                cost = str(row[cost_col]).strip() if cost_col and pd.notna(row[cost_col]) else ""
                description = str(row[desc_col]).strip() if desc_col and pd.notna(row[desc_col]) else ""

                # Round cost to 2 decimal places
                if cost and cost.lower() not in ['nan', 'none', '']:
                    try:
                        cost = f"{float(cost):.2f}"
                    except:
                        pass

                # Remove tariff-related text from description
                if description and description.lower() not in ['nan', 'none']:
                    tariff_patterns = [
                        r'\btariff[s]?\b',
                        r'\bduty\b',
                        r'\bcustom[s]?\s+duty\b',
                        r'\bimport\s+tax\b'
                    ]
                    for pattern in tariff_patterns:
                        description = re.sub(pattern, '', description, flags=re.IGNORECASE)
                    # Clean up extra spaces after removal
                    description = ' '.join(description.split())

                if not description or description.lower() in ['nan', 'none']:
                    description = 'New SKU, please add description.'

                # Skip rows with tariff, tax, and other non-product keywords
                if vendor_style:
                    vendor_style_lower = vendor_style.lower()
                    skip_keywords = ['tariff', 'tax', 'duty', 'insurance', 'charge', 'shipping',
                                   'freight', 'subtotal', 'total', 'sales', 'discount', 'payment']
                    if any(keyword in vendor_style_lower for keyword in skip_keywords):
                        continue

                # Zenith-specific: Filter out customer numbers from Excel
                if self.vendor == 'Zenith' and vendor_style:
                    # Check if it looks like a customer number: P followed by digits
                    if re.match(r'^P\d+$', vendor_style, re.IGNORECASE):
                        continue

                # Skip row if quantity or cost is missing
                quantity_valid = quantity and quantity.lower() not in ['nan', 'none', '']
                cost_valid = cost and cost.lower() not in ['nan', 'none', '']

                if vendor_style and vendor_style.lower() not in ['nan', 'none', 'style', 'sku', 'item', 'product']:
                    # Only add row if both quantity AND cost are present
                    if quantity_valid and cost_valid:
                        data.append({
                            'Vendor Style #': vendor_style,
                            'Quantity': quantity,
                            'Cost': cost,
                            'Description': description
                        })

            return data
        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")

    def smart_column_detection_pdf(self, headers):
        """Intelligently detect columns for PDF tables"""
        style_keywords = ['style', 'sku', 'item', 'product', 'prod', 'part', 'number', 'no',
                         'code', 'article', 'model', 'ref', 'reference', 'material']
        qty_keywords = ['qty', 'quantity', 'amount', 'count', 'units', 'ordered',
                       'pieces', 'pcs', 'pack', 'carton', 'shipped']
        cost_keywords = ['cost', 'price', 'rate', 'unit', 'unitprice', 'each', 'perpcs', 'charge']
        desc_keywords = ['desc', 'description', 'name', 'detail', 'title', 'specification', 'watch']

        style_idx = -1
        qty_idx = -1
        cost_idx = -1
        desc_idx = -1
        best_style_score = 0
        best_qty_score = 0
        best_cost_score = 0
        best_desc_score = 0

        # Track if we find "price" and "amount" columns for vendor-specific logic
        price_idx = -1
        amount_idx = -1
        unit_price_idx = -1
        total_price_idx = -1

        for idx, header in enumerate(headers):
            header_lower = str(header).lower()
            header_clean = header_lower.strip()

            # For Zenith: Skip customer columns when detecting style column
            if self.vendor == 'Zenith' and ('customer' in header_lower or 'cust' in header_lower):
                continue  # Don't consider this column for style detection

            style_score = sum(10 if kw in header_lower else 0 for kw in style_keywords)
            if style_score > best_style_score:
                best_style_score = style_score
                style_idx = idx

            qty_score = sum(10 if kw in header_lower else 0 for kw in qty_keywords)
            if qty_score > best_qty_score:
                best_qty_score = qty_score
                qty_idx = idx

            # Track price-related columns for vendor-specific logic
            if 'price' in header_clean and price_idx < 0:
                price_idx = idx
            if 'amount' in header_clean and amount_idx < 0:
                amount_idx = idx
            # Check for "unit price" or "unit cost"
            if ('unit' in header_clean and 'price' in header_clean) or ('unit' in header_clean and 'cost' in header_clean):
                if unit_price_idx < 0:
                    unit_price_idx = idx
            # Check for "total price" or "total" or "ext price" (extended price)
            if (('total' in header_clean and 'price' in header_clean) or
                ('total' in header_clean and not 'unit' in header_clean) or
                ('ext' in header_clean and 'price' in header_clean)):
                if total_price_idx < 0:
                    total_price_idx = idx

            cost_score = sum(10 if kw in header_lower else 0 for kw in cost_keywords)
            if cost_score > best_cost_score:
                best_cost_score = cost_score
                cost_idx = idx

            desc_score = sum(10 if kw in header_lower else 0 for kw in desc_keywords)
            if desc_score > best_desc_score:
                best_desc_score = desc_score
                desc_idx = idx

        # Vendor-specific column preferences
        if self.vendor == 'IDD':
            # IDD: prefer "price" column over "amount" column
            # st.info(f"🔍 IDD vendor detected - analyzing columns:")
            # st.info(f"   - All headers: {headers}")
            # st.info(f"   - Price column found at index: {price_idx if price_idx >= 0 else 'NOT FOUND'} {f'(header: {headers[price_idx]})' if price_idx >= 0 else ''}")
            # st.info(f"   - Amount column found at index: {amount_idx if amount_idx >= 0 else 'NOT FOUND'} {f'(header: {headers[amount_idx]})' if amount_idx >= 0 else ''}")
            # st.info(f"   - Cost column (auto-detected) at index: {cost_idx} {f'(header: {headers[cost_idx]})' if cost_idx >= 0 else ''}")
            # st.info(f"   - Qty column (auto-detected) at index: {qty_idx} {f'(header: {headers[qty_idx]})' if qty_idx >= 0 else ''}")

            if price_idx >= 0:
                old_cost_idx = cost_idx
                cost_idx = price_idx
                # if old_cost_idx != price_idx:
                #     st.success(f"📊 IDD vendor: Overriding cost column from index {old_cost_idx} ('{headers[old_cost_idx]}') to 'Price' column at index {price_idx} ('{headers[price_idx]}')")
                # else:
                #     st.success(f"✅ IDD vendor: Already using 'Price' column (index {price_idx}, '{headers[price_idx]}') for cost")
            # else:
                # st.warning(f"⚠️ IDD vendor: 'Price' column not found. Using auto-detected cost column (index {cost_idx}). Verify this is correct!")

        elif self.vendor in ['Bulova', 'Seiko', 'Citizen', 'Casio']:
            # Bulova, Seiko, Citizen, Casio: prefer "unit price" over "total price"
            # st.info(f"🔍 {self.vendor} vendor detected - analyzing columns:")
            # st.info(f"   - All headers: {headers}")
            # st.info(f"   - Unit Price column found at index: {unit_price_idx if unit_price_idx >= 0 else 'NOT FOUND'} {f'(header: {headers[unit_price_idx]})' if unit_price_idx >= 0 else ''}")
            # st.info(f"   - Total Price column found at index: {total_price_idx if total_price_idx >= 0 else 'NOT FOUND'} {f'(header: {headers[total_price_idx]})' if total_price_idx >= 0 else ''}")
            # st.info(f"   - Cost column (auto-detected) at index: {cost_idx} {f'(header: {headers[cost_idx]})' if cost_idx >= 0 else ''}")

            if unit_price_idx >= 0:
                old_cost_idx = cost_idx
                cost_idx = unit_price_idx
                # if old_cost_idx != unit_price_idx:
                #     st.success(f"📊 {self.vendor} vendor: Overriding cost column from index {old_cost_idx} ('{headers[old_cost_idx]}') to 'Unit Price' column at index {unit_price_idx} ('{headers[unit_price_idx]}')")
                # else:
                #     st.success(f"✅ {self.vendor} vendor: Already using 'Unit Price' column (index {unit_price_idx}, '{headers[unit_price_idx]}') for cost")
            # else:
                # st.warning(f"⚠️ {self.vendor} vendor: 'Unit Price' column not found. Using auto-detected cost column (index {cost_idx}). Verify this is correct!")

        elif self.vendor == 'Invicta':
            # Invicta: specific column mappings
            # - "Invoiced" → Quantity
            # - "Item" → Vendor Style
            # - "Description" → Description
            # - "Amount" → Cost

            # Find Invoiced column for quantity
            invoiced_col = self.find_best_column_match(headers, ['invoiced'], 'quantity')
            if invoiced_col:
                try:
                    invoiced_idx = headers.index(invoiced_col)
                    qty_idx = invoiced_idx
                except:
                    pass

            # Find Item column for vendor style
            item_col = self.find_best_column_match(headers, ['item'], 'style')
            if item_col:
                try:
                    item_idx = headers.index(item_col)
                    style_idx = item_idx
                except:
                    pass

            # Find Amount column for cost
            amount_col = self.find_best_column_match(headers, ['amount'], 'cost')
            if amount_col:
                try:
                    amount_idx = headers.index(amount_col)
                    cost_idx = amount_idx
                except:
                    pass

            # Description should already be detected, but double-check
            if desc_idx < 0:
                desc_col = self.find_best_column_match(headers, ['description', 'desc'], 'description')
                if desc_col:
                    try:
                        desc_idx = headers.index(desc_col)
                    except:
                        pass

        elif self.vendor == 'Zenith':
            # Zenith: Prefer "Unit Price" over "Amount USD" or other amount columns
            if unit_price_idx >= 0:
                cost_idx = unit_price_idx

            # Zenith: Ensure we don't pick customer number columns as style columns
            # If the style column contains "customer" in its name, find a different column
            if style_idx >= 0 and style_idx < len(headers):
                current_style_header = str(headers[style_idx]).lower()
                if 'customer' in current_style_header or 'cust' in current_style_header:
                    # Found customer column - need to find actual style column
                    # Look for alternative columns that could be style
                    alternative_style_idx = -1
                    best_alt_score = 0

                    for idx, header in enumerate(headers):
                        if idx == style_idx:  # Skip the customer column
                            continue
                        header_lower = str(header).lower()
                        # Look for style/sku/item/product keywords
                        alt_score = sum(10 if kw in header_lower else 0 for kw in ['style', 'sku', 'item', 'product', 'model', 'reference'])
                        if alt_score > best_alt_score:
                            best_alt_score = alt_score
                            alternative_style_idx = idx

                    # If we found an alternative, use it
                    if alternative_style_idx >= 0:
                        style_idx = alternative_style_idx

        # Fallback: if we have a 4-column table and desc wasn't detected, assume last column is description
        if desc_idx < 0 and len(headers) == 4:
            desc_idx = 3

        return style_idx, qty_idx, cost_idx, desc_idx

    def extract_text_based_data(self, page):
        """Extract data from PDF text when tables aren't detected"""
        text = page.extract_text()
        if not text:
            return []

        lines = text.split('\n')

        data_rows = []
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue

            # Try tab-separated first (most common for structured data)
            if '\t' in line:
                parts = line.split('\t')
            else:
                # Try multiple spaces (2 or more)
                parts = re.split(r'\s{2,}', line)
                if len(parts) == 1:
                    # Fall back to single space split
                    parts = line.split()

            # Clean empty parts
            parts = [p.strip() for p in parts if p and p.strip()]

            has_letters = any(re.search(r'[A-Za-z]', str(p)) for p in parts)
            has_numbers = any(re.search(r'\d', str(p)) for p in parts)

            if has_letters and has_numbers and len(parts) >= 2:
                data_rows.append(parts)

        return data_rows

    def extract_with_ocr(self, pdf_bytes):
        """Extract text from scanned PDF using OCR"""
        if not OCR_AVAILABLE:
            return []

        try:
            # Convert PDF pages to images
            images = convert_from_bytes(pdf_bytes, dpi=300)

            all_data_rows = []
            for image in images:
                # Perform OCR on the image
                text = pytesseract.image_to_string(image)

                if not text:
                    continue

                lines = text.split('\n')

                for line in lines:
                    line = line.strip()
                    if not line or len(line) < 5:
                        continue

                    # Try tab-separated first
                    if '\t' in line:
                        parts = line.split('\t')
                    else:
                        # Try multiple spaces
                        parts = re.split(r'\s{2,}', line)
                        if len(parts) == 1:
                            parts = line.split()

                    # Clean empty parts
                    parts = [p.strip() for p in parts if p and p.strip()]

                    has_letters = any(re.search(r'[A-Za-z]', str(p)) for p in parts)
                    has_numbers = any(re.search(r'\d', str(p)) for p in parts)

                    if has_letters and has_numbers and len(parts) >= 2:
                        all_data_rows.append(parts)

            return all_data_rows
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return []

    def read_pdf(self, file):
        try:
            data = []
            is_scanned = False

            with pdfplumber.open(file) as pdf:
                # Check if PDF is scanned by trying to extract text from first page
                if len(pdf.pages) > 0:
                    first_page_text = pdf.pages[0].extract_text()
                    if not first_page_text or len(first_page_text.strip()) < 10:
                        is_scanned = True

            # If PDF is scanned, try OCR
            if is_scanned:
                if OCR_AVAILABLE:
                    st.info("📷 Detected scanned PDF. Using OCR to extract text...")
                    file.seek(0)
                    pdf_bytes = file.read()
                    text_rows = self.extract_with_ocr(pdf_bytes)

                    if text_rows:
                        # Process OCR results using simple columnar format
                        for row_parts in text_rows:
                            if len(row_parts) >= 3:
                                potential_style = row_parts[0]
                                potential_qty = row_parts[1] if len(row_parts) > 1 else ""
                                potential_cost = row_parts[2] if len(row_parts) > 2 else ""
                                potential_desc = ' '.join(row_parts[3:]) if len(row_parts) > 3 else ""

                                # Validate
                                style_valid = (len(potential_style) >= 3 and
                                             re.search(r'[A-Za-z0-9]', potential_style))
                                qty_valid = (not potential_qty or
                                           re.match(r'^\d+\.?\d*$', potential_qty.replace(',', '')))
                                cost_valid = re.match(r'^\d+\.?\d*$', potential_cost.replace(',', '').replace('$', ''))

                                if style_valid and qty_valid and cost_valid:
                                    vendor_style = re.sub(r'[^\w\-/]', '', potential_style)
                                    quantity = re.sub(r'[^\d.]', '', potential_qty) if potential_qty else ""
                                    cost = re.sub(r'[^\d.]', '', potential_cost)
                                    description = potential_desc if potential_desc else 'New SKU, please add description.'

                                    # Remove tariff-related text from description
                                    if description and description != 'New SKU, please add description.':
                                        tariff_patterns = [
                                            r'\btariff[s]?\b',
                                            r'\bduty\b',
                                            r'\bcustom[s]?\s+duty\b',
                                            r'\bimport\s+tax\b'
                                        ]
                                        for pattern in tariff_patterns:
                                            description = re.sub(pattern, '', description, flags=re.IGNORECASE)
                                        # Clean up extra spaces after removal
                                        description = ' '.join(description.split())
                                        # If description becomes empty after removal, use default
                                        if not description:
                                            description = 'New SKU, please add description.'

                                    # Convert quantity to integer if it's a whole number
                                    if quantity:
                                        try:
                                            qty_float = float(quantity)
                                            if qty_float == int(qty_float):
                                                quantity = str(int(qty_float))
                                        except:
                                            pass

                                    # Skip if missing quantity or cost
                                    if not quantity or not cost:
                                        continue

                                    # Skip invoice metadata rows (invoice numbers, totals, etc.)
                                    vendor_style_upper = vendor_style.upper()
                                    if vendor_style_upper.startswith('INV') and len(vendor_style) <= 15:
                                        # Likely invoice number (e.g., INV223997)
                                        continue

                                    # Skip if description looks like a date (MM/DD/YYYY pattern)
                                    if description and re.match(r'\d{1,2}/\d{1,2}/\d{4}', description):
                                        continue

                                    try:
                                        cost = f"{float(cost):.2f}"
                                    except:
                                        pass

                                    data.append({
                                        'Vendor Style #': vendor_style,
                                        'Quantity': quantity,
                                        'Cost': cost,
                                        'Description': description
                                    })
                        return data
                else:
                    st.error("❌ This is a scanned PDF, but OCR is not available. Please install: pip install pytesseract pdf2image")
                    st.info("💡 You also need to install Tesseract OCR on your system. Visit: https://github.com/tesseract-ocr/tesseract")
                    return []

            # If not scanned, use regular processing
            file.seek(0)

            # Track incomplete rows that span across pages
            pending_row = None

            with pdfplumber.open(file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # For Invicta, skip table extraction entirely and use text extraction
                    if self.vendor == 'Invicta':
                        text = page.extract_text()
                        if text:
                            lines = text.split('\n')
                            for line in lines:
                                line = line.strip()
                                parts = line.split()
                                if len(parts) >= 7:
                                    try:
                                        ordered = parts[0]
                                        invoiced = parts[1]
                                        item = parts[2]

                                        # Check if first 3 parts look valid
                                        item_looks_like_po = '-' in item and item.count('-') >= 2
                                        item_is_valid = (len(item) <= 12 and
                                                       not item_looks_like_po and
                                                       (item.isalnum() or item.isdigit()))

                                        if ordered.isdigit() and invoiced.isdigit() and item_is_valid:
                                            amount = parts[-1].replace('$', '').replace(',', '')
                                            rate = parts[-2].replace('$', '').replace(',', '')

                                            # Parse description
                                            desc_parts = []
                                            if len(parts) > 4:
                                                start_idx = 4
                                                if len(parts) > 5 and parts[5] == item:
                                                    start_idx = 6
                                                desc_parts = parts[start_idx:-2]

                                            description = ' '.join(desc_parts).strip() if desc_parts else 'New SKU, please add description.'

                                            try:
                                                float(amount)
                                                data.append({
                                                    'Vendor Style #': item,
                                                    'Quantity': invoiced,
                                                    'Cost': amount,
                                                    'Description': description
                                                })
                                            except:
                                                pass
                                    except:
                                        pass
                        continue  # Skip to next page, don't do table extraction

                    # For Zenith, use text extraction if tables aren't detected properly
                    if self.vendor == 'Zenith':
                        text = page.extract_text()
                        if text:
                            lines = text.split('\n')
                            for line in lines:
                                line = line.strip()
                                parts = line.split()

                                # Product line format: Sr# OrderNum StyleSKU Description Qty Unit UnitPrice AmountUSD
                                # Example: 1 100001562 JNB7356S-55FY70 RING # 1.00 PCS 484.24 484.24
                                if len(parts) >= 7:
                                    try:
                                        # Check if first part is a serial number (digit)
                                        if parts[0].isdigit():
                                            sr_num = parts[0]
                                            order_num = parts[1]
                                            style_sku = parts[2]

                                            # Skip if style looks like customer number
                                            if re.match(r'^P\d+$', style_sku, re.IGNORECASE):
                                                continue

                                            # Last two should be unit_price and amount_usd
                                            amount_usd = parts[-1].replace('$', '').replace(',', '')
                                            unit_price = parts[-2].replace('$', '').replace(',', '')

                                            # Find quantity and unit (PCS, PR, etc)
                                            # Work backwards from prices to find quantity
                                            qty = None
                                            unit = None
                                            desc_end_idx = len(parts) - 2  # Default: everything before unit_price

                                            # parts[-3] should be unit (PCS, PR), parts[-4] should be quantity
                                            if len(parts) >= 4:
                                                potential_unit = parts[-3]
                                                potential_qty = parts[-4]

                                                # Check if potential_qty is a number (might have decimal)
                                                try:
                                                    float(potential_qty)
                                                    qty = potential_qty
                                                    unit = potential_unit
                                                    desc_end_idx = len(parts) - 4
                                                except:
                                                    pass

                                            # Description is between style_sku and quantity
                                            desc_parts = parts[3:desc_end_idx]
                                            description = ' '.join(desc_parts).strip() if desc_parts else 'New SKU, please add description.'

                                            if style_sku and qty and unit_price:
                                                try:
                                                    float(unit_price)
                                                    data.append({
                                                        'Vendor Style #': style_sku,
                                                        'Quantity': qty,
                                                        'Cost': unit_price,  # Use unit_price, not amount_usd
                                                        'Description': description
                                                    })
                                                except:
                                                    pass
                                    except:
                                        pass
                        continue  # Skip to next page, don't do table extraction

                    tables = page.extract_tables()
                    data_extracted_from_tables = False  # Track if we actually got data from tables

                    if tables:
                        for table in tables:
                            if not table or len(table) < 2:
                                continue

                            # Zenith-specific: Skip tables with mostly None headers (address tables)
                            if self.vendor == 'Zenith':
                                none_count = sum(1 for h in table[0] if h is None or str(h).lower() == 'none')
                                if none_count > len(table[0]) * 0.5:  # More than 50% None headers
                                    continue

                            style_idx, qty_idx, cost_idx, desc_idx = self.smart_column_detection_pdf(table[0])

                            # Debug: Show table structure for first table
                            # if page_num == 0 and len([t for t in tables if t]) == 1:  # First table only
                            #     st.info(f"📋 Table structure detected:\n- Headers: {table[0]}\n- Style column: {style_idx}, Qty column: {qty_idx}, Cost column: {cost_idx} (will extract from: '{table[0][cost_idx] if cost_idx >= 0 and cost_idx < len(table[0]) else 'INVALID'}')\n- Sample row: {table[1] if len(table) > 1 else 'N/A'}")

                            # Skip tables that don't have at least style column AND (quantity or cost) detected
                            if style_idx < 0 or (qty_idx < 0 and cost_idx < 0):
                                continue

                            for row_idx, row in enumerate(table[1:]):
                                if not row or all(not cell for cell in row):
                                    continue

                                # If we have a pending row from previous page, try to merge
                                if pending_row:
                                    # Check what the pending row is missing
                                    pending_style = str(pending_row[style_idx]).strip() if style_idx >= 0 and style_idx < len(pending_row) and pending_row[style_idx] else ""
                                    pending_qty = str(pending_row[qty_idx]).strip() if qty_idx >= 0 and qty_idx < len(pending_row) and pending_row[qty_idx] else ""
                                    pending_cost = str(pending_row[cost_idx]).strip() if cost_idx >= 0 and cost_idx < len(pending_row) and pending_row[cost_idx] else ""

                                    # Check current row
                                    current_style = str(row[style_idx]).strip() if style_idx >= 0 and style_idx < len(row) and row[style_idx] else ""
                                    current_qty = str(row[qty_idx]).strip() if qty_idx >= 0 and qty_idx < len(row) and row[qty_idx] else ""
                                    current_cost = str(row[cost_idx]).strip() if cost_idx >= 0 and cost_idx < len(row) and row[cost_idx] else ""

                                    # Check if current row is actually a header row (not data)
                                    # Header rows contain keywords like "UOM", "Qty", "Price", etc. instead of actual data
                                    header_keywords_check = ['uom', 'unit', 'ea', 'each', 'qty', 'quantity', 'price', 'cost', 'amount',
                                                            'style', 'sku', 'description', 'desc', 'total', 'subtotal']
                                    current_row_is_header = False
                                    if current_qty and current_qty.lower() in header_keywords_check:
                                        current_row_is_header = True
                                        # st.info(f"   - Current row appears to be a header (qty='{current_qty}'), not continuation data")
                                    if current_cost and current_cost.lower() in header_keywords_check:
                                        current_row_is_header = True
                                        # st.info(f"   - Current row appears to be a header (cost='{current_cost}'), not continuation data")

                                    # Determine if this is a continuation:
                                    # - Pending has style but missing qty/cost
                                    # - Current row has qty/cost (the missing data)
                                    # - Current row either has no style OR a different/duplicate style
                                    # - Current row is NOT a header row
                                    is_continuation = False
                                    if pending_style and (not pending_qty or not pending_cost):
                                        if (current_qty or current_cost) and not current_row_is_header:
                                            # This looks like continuation data
                                            is_continuation = True

                                    if is_continuation:
                                        # This looks like a continuation - merge with pending row
                                        # st.info(f"🔗 Merging cross-page row (page {page_num + 1}): {pending_style}")

                                        # Merge data: fill missing fields from pending row with current row data
                                        # BUT: Keep the vendor style from pending row (first/top style number)
                                        # ALSO: Skip header-like values when merging
                                        header_keywords = ['uom', 'unit', 'ea', 'each', 'qty', 'quantity', 'price', 'cost', 'amount',
                                                         'style', 'sku', 'description', 'desc', 'total', 'subtotal']

                                        for i, cell in enumerate(row):
                                            if cell and str(cell).strip():
                                                cell_str = str(cell).strip()
                                                cell_lower = cell_str.lower()

                                                # Skip the style column - we want to keep the pending row's style
                                                if i == style_idx:
                                                    continue  # Don't overwrite the original style

                                                # Skip header-like values (e.g., "UOM", "EA", etc.)
                                                if cell_lower in header_keywords:
                                                    # st.info(f"   - Skipping header-like value in merge: '{cell_str}' at column {i}")
                                                    continue

                                                # Fill in missing data for other columns
                                                if i < len(pending_row):
                                                    if not pending_row[i] or not str(pending_row[i]).strip():
                                                        pending_row[i] = cell

                                        # Use the completed row (with original style from pending row)
                                        row = pending_row
                                        # Extract data using the FIRST style (from pending row), not from merged row
                                        vendor_style = pending_style  # Use the original style from first row
                                        quantity = str(row[qty_idx]).strip() if qty_idx >= 0 and qty_idx < len(row) and row[qty_idx] else ""
                                        cost = str(row[cost_idx]).strip() if cost_idx >= 0 and cost_idx < len(row) and row[cost_idx] else ""
                                        description = str(row[desc_idx]).strip() if desc_idx >= 0 and desc_idx < len(row) and row[desc_idx] else ""
                                        pending_row = None
                                        # if self.vendor == 'YGI':
                                        #     st.info(f"🔍 YGI Debug - Merged row:\n   - Raw vendor style: '{vendor_style}'\n   - Qty: {quantity}, Cost: {cost}")
                                        # st.success(f"✅ Merged row - Using style from first row: {vendor_style}, Qty: {quantity}, Cost: {cost}")

                                        # Skip to data processing (don't re-extract vendor_style)
                                        # Continue with cleaning and validation below
                                    else:
                                        # Not a continuation, the pending row couldn't be completed
                                        # st.warning(f"⚠️ Could not merge pending row: {pending_style} - current row doesn't look like continuation")
                                        pending_row = None
                                        # Extract data normally for this row
                                        vendor_style = str(row[style_idx]).strip() if style_idx >= 0 and style_idx < len(row) and row[style_idx] else ""
                                        quantity = str(row[qty_idx]).strip() if qty_idx >= 0 and qty_idx < len(row) and row[qty_idx] else ""
                                        cost = str(row[cost_idx]).strip() if cost_idx >= 0 and cost_idx < len(row) and row[cost_idx] else ""
                                        description = str(row[desc_idx]).strip() if desc_idx >= 0 and desc_idx < len(row) and row[desc_idx] else ""
                                else:
                                    # No pending row, extract data normally
                                    vendor_style = str(row[style_idx]).strip() if style_idx >= 0 and style_idx < len(row) and row[style_idx] else ""
                                    quantity = str(row[qty_idx]).strip() if qty_idx >= 0 and qty_idx < len(row) and row[qty_idx] else ""
                                    cost = str(row[cost_idx]).strip() if cost_idx >= 0 and cost_idx < len(row) and row[cost_idx] else ""
                                    description = str(row[desc_idx]).strip() if desc_idx >= 0 and desc_idx < len(row) and row[desc_idx] else ""

                                # Ensure all extracted values are strings
                                vendor_style = str(vendor_style) if vendor_style else ""
                                quantity = str(quantity) if quantity else ""
                                cost = str(cost) if cost else ""
                                description = str(description) if description else ""

                                # Debug: For Invicta vendor, show what's being extracted
                                if self.vendor == 'Invicta' and vendor_style:
                                    st.info(f"🔍 Invicta Debug - Row {row_idx} extraction:\n   - Full row data: {row}\n   - Style column index: {style_idx}\n   - Qty column index: {qty_idx}\n   - Cost column index: {cost_idx}\n   - Raw vendor style: '{vendor_style}'\n   - Raw quantity: '{quantity}'\n   - Raw cost: '{cost}'\n   - All headers: {table[0]}")

                                # Jewelry Depo-specific: Handle multi-line cells by splitting into separate rows
                                if self.vendor == 'Jewelry Depo' and vendor_style and '\n' in vendor_style:
                                    # Split all fields by newlines
                                    style_lines = [line.strip() for line in vendor_style.split('\n') if line.strip()]
                                    qty_lines = [line.strip() for line in quantity.split('\n') if line.strip()]
                                    cost_lines = [line.strip() for line in cost.split('\n') if line.strip()]
                                    desc_lines_raw = [line.strip() for line in description.split('\n') if line.strip()]

                                    # Filter out unwanted description lines (FROM TEXT, PO numbers)
                                    desc_lines = []
                                    for desc in desc_lines_raw:
                                        # Skip "FROM TEXT" lines
                                        if desc.upper() == 'FROM TEXT':
                                            continue
                                        # Skip PO number lines (e.g., "25-2708", "PO # 25-2695", "PO# 25-2708", etc.)
                                        # Pattern 1: Just digits-digits
                                        if re.match(r'^\d+-\d+$', desc):
                                            continue
                                        # Pattern 2: "PO #" or "PO#" followed by digits-digits
                                        if re.match(r'^PO\s*#?\s*\d+-\d+$', desc, re.IGNORECASE):
                                            continue
                                        # Keep this description
                                        desc_lines.append(desc)

                                    # Process each line as a separate product
                                    # Note: We'll skip the normal processing below and continue to next row
                                    for i in range(len(style_lines)):
                                        item_style = style_lines[i] if i < len(style_lines) else ""
                                        item_qty = qty_lines[i] if i < len(qty_lines) else ""
                                        item_cost = cost_lines[i] if i < len(cost_lines) else ""
                                        item_desc = desc_lines[i] if i < len(desc_lines) else ""

                                        # Clean up the individual values
                                        item_style = ' '.join(item_style.split())
                                        item_style = re.sub(r'[^\w\-/]', '', item_style)

                                        # Jewelry Depo: Transform style number format
                                        # From: 005-9mm9 → To: 005TR9M9
                                        # From: 518-8mm11 → To: 518TR8M11
                                        # Pattern: ItemNum-Sizemm[#]RingSize → ItemNumTRSizeMRingSize (# is optional)
                                        style_match = re.match(r'^(\d+)-(\d+)mm#?([\d.]+)$', item_style, re.IGNORECASE)
                                        if style_match:
                                            item_num = style_match.group(1)  # e.g., "005" or "518"
                                            size = style_match.group(2)       # e.g., "9" or "8"
                                            ring_size = style_match.group(3)  # e.g., "9" or "11" or "8.5"
                                            # Transform to: ItemNumTRSizeMRingSize
                                            item_style = f"{item_num}TR{size}M{ring_size}"

                                        # Clean quantity
                                        if item_qty:
                                            item_qty = re.sub(r'[^\d.]', '', item_qty)
                                            try:
                                                qty_float = float(item_qty)
                                                if qty_float == int(qty_float):
                                                    item_qty = str(int(qty_float))
                                            except:
                                                pass

                                        # Clean cost
                                        if item_cost:
                                            item_cost = re.sub(r'[^\d.]', '', item_cost)
                                            cost_match = re.search(r'\d+\.?\d*', item_cost)
                                            if cost_match:
                                                item_cost = cost_match.group()
                                            try:
                                                item_cost = f"{float(item_cost):.2f}"
                                            except:
                                                pass

                                        # Set default description if empty
                                        if not item_desc or item_desc.lower() in ['nan', 'none']:
                                            item_desc = 'New SKU, please add description.'

                                        # Validate and add the row
                                        if item_style and item_qty and item_cost:
                                            data.append({
                                                'Vendor Style #': item_style,
                                                'Quantity': item_qty,
                                                'Cost': item_cost,
                                                'Description': item_desc
                                            })
                                            data_extracted_from_tables = True

                                    # Skip the normal processing below for this row
                                    continue

                                # Seiko-specific fix: Check if model number is split across two columns
                                # Seiko invoices sometimes have "Customer SKU Mo" and "del No" as separate columns
                                if self.vendor == 'Seiko' and vendor_style:
                                    # Check if the next column exists and contains part of the model number
                                    next_col_idx = style_idx + 1
                                    if next_col_idx < len(row) and row[next_col_idx]:
                                        next_col_value = str(row[next_col_idx]).strip()
                                        # Check if vendor_style is very short (likely incomplete) and next column has data
                                        if len(vendor_style) <= 3 and next_col_value and len(next_col_value) <= 10:
                                            # Likely split model number - concatenate them
                                            vendor_style = vendor_style + next_col_value
                                            # st.info(f"🔧 Seiko: Merged split model number - '{row[style_idx]}' + '{next_col_value}' = '{vendor_style}'")

                                # Clean up vendor style - if cell contains multiple lines, take the FIRST one (top style number)
                                if vendor_style:
                                    # Split by newlines to handle cells with multiple style numbers stacked vertically
                                    style_lines = [line.strip() for line in vendor_style.split('\n') if line.strip()]
                                    if len(style_lines) > 1:
                                        # st.info(f"📦 Multiple style numbers in cell: {style_lines} - Using first: {style_lines[0]}")
                                        vendor_style = style_lines[0]  # Always use the TOP/FIRST style number

                                    vendor_style = ' '.join(vendor_style.split())
                                    # Remove non-alphanumeric chars except hyphens and slashes
                                    vendor_style = re.sub(r'[^\w\-/]', '', vendor_style)

                                # Clean up quantity - extract just the number
                                if quantity:
                                    quantity = re.sub(r'[^\d.]', '', quantity)
                                    # Convert to integer if it's a whole number
                                    try:
                                        qty_float = float(quantity)
                                        if qty_float == int(qty_float):
                                            quantity = str(int(qty_float))
                                    except:
                                        pass

                                # Clean up cost - extract just the number
                                if cost:
                                    # Remove all non-numeric chars except decimal point
                                    cost = re.sub(r'[^\d.]', '', cost)
                                    # If multiple decimals or other issues, try to get first valid number
                                    cost_match = re.search(r'\d+\.?\d*', cost)
                                    if cost_match:
                                        cost = cost_match.group()
                                    # Round to 2 decimal places
                                    try:
                                        cost = f"{float(cost):.2f}"
                                    except:
                                        pass

                                # Clean up description - remove newlines and extra spaces
                                if description:
                                    description = ' '.join(description.split())

                                    # Remove tariff-related text from description
                                    tariff_patterns = [
                                        r'\btariff[s]?\b',
                                        r'\bduty\b',
                                        r'\bcustom[s]?\s+duty\b',
                                        r'\bimport\s+tax\b'
                                    ]
                                    for pattern in tariff_patterns:
                                        description = re.sub(pattern, '', description, flags=re.IGNORECASE)

                                    # Clean up extra spaces after removal
                                    description = ' '.join(description.split())

                                if not description or description.lower() in ['nan', 'none']:
                                    description = 'New SKU, please add description.'

                                # BRAND NAME SWAP: If vendor style is a brand name, swap with product code from description
                                brand_names = ['citizen', 'bulova', 'seiko', 'casio', 'timex']

                                # Check if vendor_style is a brand name
                                if vendor_style and vendor_style.lower() in brand_names:
                                    # If description exists and contains a product code, extract it
                                    if description and description != 'New SKU, please add description.':
                                        desc_words = description.split()
                                        if desc_words:
                                            # Check if first word looks like a product code (letters + numbers)
                                            first_word = desc_words[0]
                                            if re.search(r'[A-Za-z]', first_word) and re.search(r'\d', first_word):
                                                # Perform the swap
                                                vendor_style = first_word
                                                description = ' '.join(desc_words[1:]) if len(desc_words) > 1 else 'New SKU, please add description.'
                                            else:
                                                # No valid product code in description, skip this row (it's probably a subtotal)
                                                continue
                                        else:
                                            # Description is empty after splitting, skip
                                            continue
                                    else:
                                        # Brand name but no description, skip this row
                                        continue

                                # Skip empty or invalid vendor styles
                                if not vendor_style or len(vendor_style) < 2:
                                    continue

                                # Skip vendor styles that are all numeric (likely IDs, not product codes)
                                if vendor_style.replace('-', '').replace('/', '').isdigit():
                                    continue

                                # Vendor-specific filtering
                                # if self.vendor == 'YGI':
                                #     st.info(f"🔍 YGI Debug - Checking vendor filter:\n   - Cleaned vendor style: '{vendor_style}'\n   - Will check against patterns")
                                if self.should_skip_vendor_style(vendor_style):
                                    # st.info(f"⏭️ Skipping vendor-filtered code ({self.vendor}): {vendor_style}")
                                    continue

                                # Zenith-specific: Filter out customer numbers
                                # Customer numbers typically follow pattern like P149, P001, etc.
                                if self.vendor == 'Zenith':
                                    # Check if it looks like a customer number: P followed by digits
                                    if re.match(r'^P\d+$', vendor_style, re.IGNORECASE):
                                        continue

                                vendor_style_lower = vendor_style.lower()

                                # Skip header-like values (including column headers that look like data)
                                header_keywords = ['nan', 'none', 'style', 'sku', 'item', 'product', 'model', 'no', 'modelno',
                                                 'shipped', 'ordered', 'backordered', 'quantity', 'price', 'cost', 'description', 'ins', 'insurance']
                                if vendor_style_lower in header_keywords:
                                    continue

                                # Final safety check: skip if vendor style is STILL a brand name (shouldn't happen after swap)
                                if vendor_style_lower in brand_names:
                                    continue

                                # Skip rows with keywords that indicate non-product lines
                                skip_keywords = ['insurance', 'charge', 'shipping', 'freight', 'tax', 'tariff', 'duty',
                                               'subtotal', 'total', 'sales', 'sale', 'discount', 'remit', 'balance',
                                               'payment', 'invoice', 'pursuant', 'article', 'chapter', 'division', 'summary']
                                if any(keyword in vendor_style_lower for keyword in skip_keywords):
                                    continue

                                # Skip if cost is invalid (0.00 or empty after cleaning)
                                if cost and (float(cost) if cost.replace('.', '').isdigit() else -1) == 0.0:
                                    continue

                                # Remove tariff-related text from description
                                if description and description.strip():
                                    tariff_patterns = [
                                        r'\btariff[s]?\b',
                                        r'\bduty\b',
                                        r'\bcustom[s]?\s+duty\b',
                                        r'\bimport\s+tax\b'
                                    ]
                                    for pattern in tariff_patterns:
                                        description = re.sub(pattern, '', description, flags=re.IGNORECASE)
                                    # Clean up extra spaces after removal
                                    description = ' '.join(description.split())

                                # Ensure description is never empty
                                if not description or description.strip() == '':
                                    description = 'New SKU, please add description.'

                                # Check if row is incomplete (might continue on next page)
                                if vendor_style and (not quantity or not cost):
                                    # Save as pending for next page
                                    # st.warning(f"⏸️ Incomplete row saved (page {page_num + 1}): {vendor_style} (missing: {'qty' if not quantity else ''}{' and ' if not quantity and not cost else ''}{'cost' if not cost else ''})")
                                    pending_row = row
                                    continue

                                # Skip row if still missing quantity or cost after merge
                                if not quantity or not cost:
                                    # st.warning(f"❌ Skipping row after cleaning (page {page_num + 1}): {vendor_style} - Qty: '{quantity}', Cost: '{cost}' (one or both empty)")
                                    continue

                                # Append the data
                                # st.success(f"✔️ Added row: {vendor_style} - Qty: {quantity}, Cost: {cost}")
                                data.append({
                                    'Vendor Style #': vendor_style,
                                    'Quantity': quantity,
                                    'Cost': cost,
                                    'Description': description
                                })
                                data_extracted_from_tables = True  # Mark that we got data from tables

                    # If no tables were found OR tables were found but yielded no data, try text extraction
                    if not tables or not data_extracted_from_tables:
                        text_rows = self.extract_text_based_data(page)

                        # If we have a pending row, try to merge with first text row
                        if pending_row and text_rows and len(text_rows) > 0:
                            first_row = text_rows[0]
                            # Check if first row looks like continuation (no vendor style at start)
                            if len(first_row) > 0:
                                first_part = first_row[0]
                                # If first part doesn't look like vendor style (no letters+numbers combo)
                                has_alpha = bool(re.search(r'[A-Za-z]', first_part))
                                has_digit = bool(re.search(r'\d', first_part))
                                if not (has_alpha and has_digit):
                                    # Looks like continuation - merge
                                    if isinstance(pending_row, list):
                                        # Convert pending_row to text format and merge
                                        text_rows[0] = list(pending_row) + first_row
                                    pending_row = None

                        if text_rows:
                            # Track which rows were successfully processed
                            processed_row_indices = set()

                            # First, try simple tab-separated format (VendorStyle\tQty\tCost\tDescription)
                            for row_idx, row_parts in enumerate(text_rows):
                                # Check if this looks like simple columnar data
                                if len(row_parts) >= 3:
                                    # Find the SKU dynamically - look for first part with letters AND numbers
                                    sku_idx = -1
                                    for i, part in enumerate(row_parts):
                                        part = str(part)  # Ensure part is a string
                                        has_alpha = bool(re.search(r'[A-Za-z]', part))
                                        has_digit = bool(re.search(r'\d', part))
                                        if has_alpha and has_digit and len(part) >= 3:
                                            sku_idx = i
                                            break

                                    # If no SKU found, skip to complex pattern matching
                                    if sku_idx < 0:
                                        continue

                                    # Look for quantity and cost after the SKU
                                    # They should be numeric values (with optional decimals)
                                    potential_style = str(row_parts[sku_idx])
                                    potential_qty = ""
                                    potential_cost = ""
                                    potential_desc_parts = []

                                    numeric_values = []
                                    for j in range(sku_idx + 1, len(row_parts)):
                                        part = str(row_parts[j])  # Ensure part is a string
                                        if re.match(r'^\d+\.?\d*$', part.replace(',', '').replace('$', '')):
                                            numeric_values.append(part.replace('$', '').replace(',', ''))
                                        else:
                                            # Non-numeric parts are description
                                            potential_desc_parts.append(part)

                                    # Heuristic: If we have multiple numeric values, quantity is often one of the smaller ones
                                    # and cost is often one of the larger ones, or second-to-last
                                    if len(numeric_values) >= 2:
                                        # First numeric is usually quantity, UNLESS it looks like a price (>= $10)
                                        # For watch brands, if first value is >= $10, it's likely a price, not quantity
                                        first_val_float = float(numeric_values[0]) if numeric_values[0] else 0
                                        if self.vendor in ['Bulova', 'Seiko', 'Citizen', 'Casio'] and first_val_float >= 10.0:
                                            # First value looks like a price, not quantity - might be missing qty in source
                                            # Skip this row or look for qty elsewhere
                                            # st.warning(f"⚠️ First numeric value ({numeric_values[0]}) looks like price (>= $10), not quantity. Row may be malformed.")
                                            # Try to find a small value (< 10) that could be quantity
                                            potential_qty = ""
                                            for val in numeric_values:
                                                try:
                                                    if float(val) < 10.0 and float(val) >= 1.0:
                                                        potential_qty = val
                                                        # st.info(f"   - Found potential quantity: {potential_qty}")
                                                        break
                                                except:
                                                    pass
                                            if not potential_qty:
                                                # No valid quantity found, skip this row
                                                # st.warning(f"   - No valid quantity found, skipping row")
                                                continue
                                        else:
                                            potential_qty = numeric_values[0]

                                        # Vendor-specific cost selection logic
                                        if self.vendor == 'IDD':
                                            # IDD: prefer unit price over total amount
                                            # Strategy: Find first price value > $10 (to skip small values like 1.00, 0.00)
                                            # Unit prices are typically substantial amounts
                                            potential_cost = None
                                            for val in numeric_values[1:]:  # Skip first (quantity)
                                                try:
                                                    float_val = float(val)
                                                    # Look for first price >= $10 (reasonable unit price threshold)
                                                    if float_val >= 10.0:
                                                        potential_cost = val
                                                        break
                                                except:
                                                    pass

                                            # Fallback: if no price >= $10, use first non-zero value
                                            if not potential_cost:
                                                for val in numeric_values[1:]:
                                                    try:
                                                        if float(val) > 0:
                                                            potential_cost = val
                                                            break
                                                    except:
                                                        pass

                                            # Last fallback
                                            if not potential_cost:
                                                potential_cost = numeric_values[1] if len(numeric_values) >= 2 else numeric_values[-1]

                                            # if len(numeric_values) >= 2:
                                            #     st.info(f"🔍 IDD text extraction - All numeric values: {numeric_values}")
                                            #     st.info(f"   - Prices found (after qty): {numeric_values[1:]}")
                                            #     st.info(f"   - Selected unit price (first >= $10): {potential_cost}")
                                        elif self.vendor in ['Bulova', 'Seiko', 'Citizen', 'Casio']:
                                            # Watch brands: prefer unit price over total price
                                            # Strategy: Find first price value >= $10 (to skip small values like 1.00)
                                            potential_cost = None
                                            for val in numeric_values[1:]:  # Skip first (quantity)
                                                try:
                                                    float_val = float(val)
                                                    # Look for first price >= $10 (reasonable unit price threshold)
                                                    if float_val >= 10.0:
                                                        potential_cost = val
                                                        break
                                                except:
                                                    pass

                                            # Fallback: if no price >= $10, use first non-zero value
                                            if not potential_cost:
                                                for val in numeric_values[1:]:
                                                    try:
                                                        if float(val) > 0:
                                                            potential_cost = val
                                                            break
                                                    except:
                                                        pass

                                            # Last fallback
                                            if not potential_cost:
                                                potential_cost = numeric_values[1] if len(numeric_values) >= 2 else numeric_values[-1]

                                            # if len(numeric_values) >= 2:
                                            #     st.info(f"🔍 {self.vendor} text extraction - All numeric values: {numeric_values}")
                                            #     st.info(f"   - Prices found (after qty): {numeric_values[1:]}")
                                            #     st.info(f"   - Selected unit price (first >= $10): {potential_cost}")
                                        else:
                                            # Default behavior: For cost, prefer larger values over smaller ones
                                            # Strategy: Use the largest value that looks like a price (> 1.0)
                                            largest_price = None
                                            for val in numeric_values[1:]:  # Skip first (quantity)
                                                try:
                                                    float_val = float(val)
                                                    if float_val >= 1.0:  # Prices are usually >= 1
                                                        if largest_price is None or float_val > float(largest_price):
                                                            largest_price = val
                                                except:
                                                    pass

                                            # If we found a large price, use it; otherwise use last numeric value
                                            if largest_price:
                                                potential_cost = largest_price
                                            else:
                                                potential_cost = numeric_values[-1]  # Last value as fallback
                                    elif len(numeric_values) == 1:
                                        # Only one numeric value - assume it's the cost
                                        potential_cost = numeric_values[0]

                                    potential_desc = ' '.join(potential_desc_parts) if potential_desc_parts else ""

                                    # Validate style (alphanumeric, at least 3 chars, but not too long)
                                    # Skip if vendor style is unreasonably long (likely extracted text, not a SKU)
                                    style_valid = (len(potential_style) >= 3 and
                                                 len(potential_style) <= 50 and  # SKUs are typically <= 50 chars
                                                 re.search(r'[A-Za-z0-9]', potential_style))

                                    # Validate quantity (numeric or empty)
                                    qty_valid = (not potential_qty or
                                               re.match(r'^\d+\.?\d*$', potential_qty.replace(',', '')))

                                    # Validate cost (numeric with optional decimal)
                                    cost_valid = potential_cost and re.match(r'^\d+\.?\d*$', potential_cost.replace(',', '').replace('$', ''))

                                    if style_valid and qty_valid and cost_valid:
                                        # Clean up the values
                                        vendor_style = re.sub(r'[^\w\-/]', '', potential_style)
                                        quantity = re.sub(r'[^\d.]', '', potential_qty) if potential_qty else ""
                                        cost = re.sub(r'[^\d.]', '', potential_cost)
                                        description = potential_desc if potential_desc else 'New SKU, please add description.'

                                        # Remove tariff-related text from description
                                        if description and description != 'New SKU, please add description.':
                                            tariff_patterns = [
                                                r'\btariff[s]?\b',
                                                r'\bduty\b',
                                                r'\bcustom[s]?\s+duty\b',
                                                r'\bimport\s+tax\b'
                                            ]
                                            for pattern in tariff_patterns:
                                                description = re.sub(pattern, '', description, flags=re.IGNORECASE)
                                            # Clean up extra spaces after removal
                                            description = ' '.join(description.split())
                                            # If description becomes empty after removal, use default
                                            if not description:
                                                description = 'New SKU, please add description.'

                                        # Convert quantity to integer if it's a whole number
                                        if quantity:
                                            try:
                                                qty_float = float(quantity)
                                                if qty_float == int(qty_float):
                                                    quantity = str(int(qty_float))
                                            except:
                                                pass

                                        # Skip if missing quantity or cost
                                        if not quantity or not cost:
                                            continue

                                        # For watch brands: Skip if quantity and cost are identical (malformed data)
                                        # This indicates the row is missing actual quantity data
                                        if self.vendor in ['Bulova', 'Seiko', 'Citizen', 'Casio']:
                                            try:
                                                if float(quantity) == float(cost):
                                                    # st.warning(f"⏭️ Skipping {vendor_style} - quantity ({quantity}) equals cost ({cost}), row is malformed")
                                                    continue
                                            except:
                                                pass

                                        # For watch brands: Skip if quantity looks like a price (decimal value >= $10)
                                        # Quantities should typically be whole numbers or small decimals (< 10)
                                        if self.vendor in ['Bulova', 'Seiko', 'Citizen', 'Casio']:
                                            try:
                                                qty_float = float(quantity)
                                                # If qty has a decimal point AND is >= $10, it's likely a price, not quantity
                                                if '.' in quantity and qty_float >= 10.0:
                                                    # st.warning(f"⏭️ Skipping {vendor_style} - quantity ({quantity}) looks like price (decimal >= $10)")
                                                    continue
                                            except:
                                                pass

                                        # For watch brands: Skip rows where cost is suspiciously low (likely missing price data)
                                        # Rows with cost = 1.00 are usually missing actual pricing
                                        if self.vendor in ['Bulova', 'Seiko', 'Citizen', 'Casio']:
                                            try:
                                                if float(cost) < 5.0:  # Unit prices for watches are typically >= $5
                                                    # st.warning(f"⏭️ Skipping {vendor_style} - cost too low ({cost}), likely missing price data")
                                                    continue
                                            except:
                                                pass

                                        # Skip invoice metadata rows (invoice numbers, totals, etc.)
                                        vendor_style_upper = vendor_style.upper()
                                        if vendor_style_upper.startswith('INV') and len(vendor_style) <= 15:
                                            # Likely invoice number (e.g., INV223997)
                                            # st.info(f"⏭️ Skipping invoice metadata: {vendor_style}")
                                            continue

                                        # Skip if description looks like a date (MM/DD/YYYY pattern)
                                        if description and re.match(r'\d{1,2}/\d{1,2}/\d{4}', description):
                                            # st.info(f"⏭️ Skipping date row: {vendor_style}")
                                            continue

                                        # Vendor-specific filtering
                                        if self.should_skip_vendor_style(vendor_style):
                                            # st.info(f"⏭️ Skipping vendor-filtered code ({self.vendor}): {vendor_style}")
                                            continue

                                        # Round cost
                                        try:
                                            cost = f"{float(cost):.2f}"
                                        except:
                                            pass

                                        # st.success(f"✔️ Added row (text): {vendor_style} - Qty: {quantity}, Cost: {cost}")
                                        data.append({
                                            'Vendor Style #': vendor_style,
                                            'Quantity': quantity,
                                            'Cost': cost,
                                            'Description': description
                                        })
                                        processed_row_indices.add(row_idx)  # Mark as processed
                                        continue

                            # Fall back to complex pattern matching for unstructured text
                            for row_idx, row_parts in enumerate(text_rows):
                                # Skip rows that were already processed by simple format
                                if row_idx in processed_row_indices:
                                    continue
                                vendor_style = ""
                                quantity = ""
                                cost = ""
                                description = ""

                                price_fields = []
                                qty_fields = []
                                style_idx = -1
                                ea_idx = -1

                                for i, part in enumerate(row_parts):
                                    part = str(part).strip()

                                    # More flexible pattern for vendor style codes
                                    # Matches: AB1234, AB1234-56X, AB-1234, 1234AB, ABC, etc.
                                    # Must have at least one letter and one number, OR be 3+ alphanumeric chars
                                    has_letter = bool(re.search(r'[A-Z]', part.upper()))
                                    has_number = bool(re.search(r'\d', part))
                                    is_alphanumeric = re.match(r'^[A-Z0-9\-/]+$', part.upper())

                                    if is_alphanumeric and len(part) >= 3:
                                        if (has_letter and has_number) or (has_letter and len(part) >= 4):
                                            if not vendor_style:  # Only take the first match
                                                vendor_style = part
                                                style_idx = i
                                    elif part.upper() == 'EA':
                                        ea_idx = i
                                    elif re.match(r'^\$?\d+\.?\d*$', part.replace(',', '')):
                                        if '$' in part or '.' in part:
                                            price_fields.append(part.replace('$', ''))
                                        elif re.match(r'^\d+$', part):
                                            qty_fields.append(part)

                                if vendor_style and price_fields:
                                    # Skip summary rows (subtotal, total, grand total, etc.)
                                    vendor_style_lower = vendor_style.lower()
                                    skip_keywords = ['insurance', 'charge', 'shipping', 'freight', 'tax', 'tariff', 'duty',
                                                    'subtotal', 'total', 'sales', 'sale', 'discount', 'remit', 'balance',
                                                    'payment', 'invoice', 'pursuant', 'article', 'chapter', 'division',
                                                    'summary', 'grand', 'amount']

                                    if any(keyword in vendor_style_lower for keyword in skip_keywords):
                                        continue

                                    # Skip if cost is 0.00
                                    cost_value = price_fields[-2] if len(price_fields) >= 2 else price_fields[-1]
                                    if cost_value and (float(cost_value) if cost_value.replace('.', '').isdigit() else -1) == 0.0:
                                        continue

                                    if qty_fields:
                                        quantity = qty_fields[0]
                                    cost = price_fields[-2] if len(price_fields) >= 2 else price_fields[-1]

                                    # Round cost to 2 decimal places
                                    try:
                                        cost = f"{float(cost):.2f}"
                                    except:
                                        pass

                                    if style_idx >= 0 and ea_idx >= 0:
                                        desc_parts = row_parts[style_idx + 1:ea_idx]
                                        description = ' '.join(desc_parts)
                                    elif style_idx >= 0:
                                        desc_end = min(style_idx + 8, len(row_parts))
                                        desc_parts = []
                                        for part in row_parts[style_idx + 1:desc_end]:
                                            if not re.match(r'^\$?\d+\.?\d*$', part.replace(',', '')) and part.upper() != 'EA':
                                                desc_parts.append(part)
                                        description = ' '.join(desc_parts)

                                    # Remove tariff-related text from description
                                    if description:
                                        tariff_patterns = [
                                            r'\btariff[s]?\b',
                                            r'\bduty\b',
                                            r'\bcustom[s]?\s+duty\b',
                                            r'\bimport\s+tax\b'
                                        ]
                                        for pattern in tariff_patterns:
                                            description = re.sub(pattern, '', description, flags=re.IGNORECASE)
                                        # Clean up extra spaces after removal
                                        description = ' '.join(description.split())

                                    if not description:
                                        description = 'New SKU, please add description.'

                                    # BRAND NAME SWAP: If vendor style is a brand name, swap with product code from description
                                    brand_names = ['citizen', 'bulova', 'seiko', 'casio', 'timex']

                                    # Check if vendor_style is a brand name
                                    if vendor_style and vendor_style.lower() in brand_names:
                                        # If description exists and contains a product code, extract it
                                        if description and description != 'New SKU, please add description.':
                                            desc_words = description.split()
                                            if desc_words:
                                                # Check if first word looks like a product code (letters + numbers)
                                                first_word = desc_words[0]
                                                if re.search(r'[A-Za-z]', first_word) and re.search(r'\d', first_word):
                                                    # Perform the swap
                                                    vendor_style = first_word
                                                    description = ' '.join(desc_words[1:]) if len(desc_words) > 1 else 'New SKU, please add description.'
                                                else:
                                                    # No valid product code in description, skip this row (it's probably a subtotal)
                                                    continue
                                            else:
                                                # Description is empty after splitting, skip
                                                continue
                                        else:
                                            # Brand name but no description, skip this row
                                            continue

                                    # Final safety check: skip if vendor style is STILL a brand name
                                    if vendor_style.lower() in brand_names:
                                        continue

                                    # Ensure description is never empty
                                    if not description or description.strip() == '':
                                        description = 'New SKU, please add description.'

                                    # Vendor-specific filtering
                                    if self.should_skip_vendor_style(vendor_style):
                                        # st.info(f"⏭️ Skipping vendor-filtered code ({self.vendor}): {vendor_style}")
                                        continue

                                    # Skip row if quantity or cost is missing
                                    if not quantity or not cost:
                                        continue

                                    # st.success(f"✔️ Added row (complex): {vendor_style} - Qty: {quantity}, Cost: {cost}")
                                    data.append({
                                        'Vendor Style #': vendor_style,
                                        'Quantity': quantity,
                                        'Cost': cost,
                                        'Description': description
                                    })

            # Process any remaining pending row at the end
            if pending_row:
                # Try to extract what we can from the incomplete row
                if isinstance(pending_row, list) and len(pending_row) > 0:
                    # Assume first column is vendor style
                    vendor_style = str(pending_row[0]).strip() if pending_row[0] else ""
                    if vendor_style and len(vendor_style) >= 3:
                        st.warning(f"⚠️ Incomplete row at end of document: {vendor_style} (missing quantity or cost)")

            return data
        except Exception as e:
            import traceback
            st.error(f"Error details: {traceback.format_exc()}")
            raise Exception(f"Error reading PDF file: {str(e)}")


def main():
    st.set_page_config(
        page_title="Invoice Import Tool",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 Invoice Import Tool")
    st.markdown("""
    Upload an Excel or PDF invoice file to extract vendor style, quantity, cost, and description data.
    The application will automatically detect the correct columns and export to Excel format!
    """)

    st.divider()

    # Vendor selection
    col1, col2 = st.columns([1, 3])
    with col1:
        vendor = st.selectbox(
            "Select Vendor",
            options=['None', 'YGI', 'DNG', 'Rolex', 'Seiko', 'Citizen', 'Bulova', 'Casio', 'IDD', 'Invicta', 'Zenith', 'Jewelry Depo'],
            help="Select vendor to apply vendor-specific filters (e.g., skip secondary identifiers)"
        )
        if vendor == 'None':
            vendor = None

    # Show active filters if vendor selected
    if vendor:
        with col2:
            importer_temp = InvoiceImporter(vendor=vendor)
            if vendor in importer_temp.vendor_filters:
                filter_desc = importer_temp.vendor_filters[vendor]['description']
                st.info(f"✅ **{vendor}:** {filter_desc}")

    uploaded_file = st.file_uploader(
        "Choose an invoice file (Excel or PDF)",
        type=['xlsx', 'xls', 'pdf'],
        help="Upload an invoice file in Excel (.xlsx, .xls) or PDF format"
    )

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if st.button("🚀 Process File", type="primary"):
            # Clear any caching
            st.cache_data.clear()

            importer = InvoiceImporter(vendor=vendor)

            with st.spinner("Processing file..."):
                try:
                    if file_ext in ['.xlsx', '.xls']:
                        data = importer.read_excel(uploaded_file)
                    elif file_ext == '.pdf':
                        data = importer.read_pdf(uploaded_file)
                    else:
                        st.error("❌ Unsupported file format")
                        return

                    if not data:
                        st.warning("⚠️ No data could be extracted from the file.")

                        # DEBUG: Show what happened during extraction
                        st.write("DEBUG: Re-processing to show details...")
                        if file_ext == '.pdf':
                            uploaded_file.seek(0)
                            with pdfplumber.open(uploaded_file) as pdf:
                                for page_num, page in enumerate(pdf.pages[:1]):
                                    tables = page.extract_tables()
                                    for table_num, table in enumerate(tables):
                                        if table and len(table) >= 2:
                                            st.write(f"**Table {table_num + 1} Headers:** {table[0]}")
                                            style_idx, qty_idx, cost_idx, desc_idx = importer.smart_column_detection_pdf(table[0])
                                            st.write(f"- Style column index: {style_idx} = {table[0][style_idx] if style_idx >= 0 else 'NOT FOUND'}")
                                            st.write(f"- Qty column index: {qty_idx} = {table[0][qty_idx] if qty_idx >= 0 else 'NOT FOUND'}")
                                            st.write(f"- Cost column index: {cost_idx} = {table[0][cost_idx] if cost_idx >= 0 else 'NOT FOUND'}")
                                            st.write(f"- Desc column index: {desc_idx} = {table[0][desc_idx] if desc_idx >= 0 else 'NOT FOUND'}")
                                            if style_idx >= 0:
                                                for row_num, row in enumerate(table[1:3]):
                                                    if row:
                                                        vendor_style_raw = row[style_idx] if style_idx < len(row) else "OUT OF BOUNDS"
                                                        st.write(f"- Row {row_num + 1} vendor style (raw): '{vendor_style_raw}'")

                        # Show diagnostic information for troubleshooting
                        if file_ext == '.pdf':
                            with st.expander("🔍 Show full diagnostic information"):
                                try:
                                    with pdfplumber.open(uploaded_file) as pdf:
                                        st.write(f"**PDF Pages:** {len(pdf.pages)}")

                                        for page_num, page in enumerate(pdf.pages[:2]):  # Check first 2 pages
                                            st.write(f"\n**Page {page_num + 1}:**")

                                            tables = page.extract_tables()
                                            st.write(f"- Tables found: {len(tables)}")

                                            # Show table contents
                                            if tables:
                                                for table_num, table in enumerate(tables[:3]):  # First 3 tables
                                                    st.write(f"\n**Table {table_num + 1}:**")
                                                    if table and len(table) > 0:
                                                        st.write(f"  - Rows: {len(table)}, Columns: {len(table[0]) if table[0] else 0}")
                                                        st.write(f"  - Headers: {table[0]}")
                                                        # Show first 3 data rows
                                                        for row_num, row in enumerate(table[1:4]):
                                                            st.write(f"  - Row {row_num + 1}: {row}")

                                            text = page.extract_text()
                                            st.write(f"\n**Text extraction result:**")
                                            if text:
                                                st.write(f"- Text extracted: YES (length: {len(text)} characters)")
                                                lines = text.split('\n')[:30]  # First 30 lines
                                                st.write(f"- Total text lines: {len(text.split(chr(10)))}")
                                                st.write(f"- Non-empty lines in first 30: {sum(1 for line in lines if line.strip())}")
                                                st.write("\n**Sample lines (first 30):**")
                                                for i, line in enumerate(lines):
                                                    if line.strip():
                                                        st.code(f"{i}: {line}")
                                                    else:
                                                        st.text(f"{i}: (empty line)")
                                            else:
                                                st.write("- Text extracted: NO - This is likely an image-based (scanned) PDF")
                                                st.warning("⚠️ This PDF appears to be image-based. Text cannot be extracted without OCR.")
                                except Exception as e:
                                    st.error(f"Could not read diagnostic info: {str(e)}")

                        st.info("**Possible reasons:**\n"
                               "- The file doesn't contain recognizable table structures\n"
                               "- Column headers don't match expected patterns (style, sku, qty, quantity, cost, price, etc.)\n"
                               "- For PDFs: The file may be image-based (scanned) rather than text-based\n"
                               "- Vendor style codes don't have letters and numbers, or are too short (< 3 characters)\n"
                               "- No price information found on the same line as vendor styles")
                        return

                    st.divider()
                    st.subheader("📋 Extracted Data Preview")

                    output_df = pd.DataFrame(data)
                    st.dataframe(output_df, use_container_width=True)

                    # Calculate totals
                    total_quantity = 0
                    total_cost = 0.0

                    for item in data:
                        # Parse quantity
                        qty_str = str(item.get('Quantity', '')).replace(',', '')
                        qty = 0
                        if qty_str and qty_str.replace('.', '').isdigit():
                            qty = int(float(qty_str))
                            total_quantity += qty

                        # Parse cost (unit price)
                        cost_str = str(item.get('Cost', '')).replace(',', '').replace('$', '')
                        if cost_str and cost_str.replace('.', '').isdigit():
                            unit_cost = float(cost_str)
                            # Multiply unit cost by quantity for line total
                            total_cost += unit_cost * qty

                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(data))
                    with col2:
                        st.metric("Total Quantity", f"{total_quantity:,}")
                    with col3:
                        st.metric("Total Cost", f"${total_cost:,.2f}")

                    st.divider()
                    st.subheader("💾 Download Results")

                    output = BytesIO()
                    output_df.to_excel(output, index=False, engine='openpyxl')
                    output.seek(0)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"invoice_import_{timestamp}.xlsx"

                    st.download_button(
                        label="📥 Download Excel File",
                        data=output,
                        file_name=output_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )

                    st.success(f"✅ File processed successfully! {len(data)} rows extracted.")

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    with st.expander("Show error details"):
                        st.exception(e)

    else:
        st.info("👆 Please upload a file to get started")

    st.divider()
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        ### Instructions:
        1. **Select your vendor** (optional) - Choose from the dropdown to apply vendor-specific filters
        2. **Upload your invoice file** - Select an Excel (.xlsx, .xls) or PDF file
        3. **Click 'Process File'** - The application will automatically detect columns
        4. **Review the data** - Check the extracted data in the preview table
        5. **Download results** - Click the download button to save as Excel

        ### Vendor Selection:
        - **YGI**: Filters out Amazon ASINs (B0/X00 codes) and U-codes
        - **IDD**: Prefers "Price" column over "Amount" column
        - **Casio/Bulova/Seiko/Citizen**: Prefers "Unit Price" over "Total Price"
        - **Invicta**: Maps "Invoiced"→Quantity, "Item"→Style, "Amount"→Cost
        - **Zenith**: Prefers "Unit Price" over "Amount USD", filters customer numbers (P### pattern)
        - **Jewelry Depo**: Standard column detection
        - **None**: No vendor-specific filters applied

        ### What gets extracted:
        - **Vendor Style #**: SKU, item code, or product number
        - **Quantity**: Number of items ordered
        - **Cost**: Unit cost or price
        - **Description**: Product description (when available)

        ### Important:
        ⚠️ **Rows without quantity OR cost are automatically skipped**

        ### Supported formats:
        - ✅ Excel files with table headers
        - ✅ PDF files with structured tables
        - ✅ PDF files with text-based invoices
        - 📷 Scanned/image PDFs (requires OCR setup - see INSTALL_OCR.md)
        """)

        if not OCR_AVAILABLE:
            st.warning("📷 OCR not available. To process scanned PDFs, run: `install_ocr.bat` or see INSTALL_OCR.md")


if __name__ == "__main__":
    main()

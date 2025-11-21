import streamlit as st
import pandas as pd
import pdfplumber
import os
from datetime import datetime
import re
from io import BytesIO


class InvoiceImporter:
    def __init__(self):
        pass

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

                if not description or description.lower() in ['nan', 'none']:
                    description = 'New SKU, please add description.'

                if vendor_style and vendor_style.lower() not in ['nan', 'none', 'style', 'sku', 'item', 'product']:
                    data.append({
                        'Vendor Style #': vendor_style,
                        'Quantity': quantity if quantity and quantity.lower() != 'nan' else '',
                        'Cost': cost if cost and cost.lower() != 'nan' else '',
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

        for idx, header in enumerate(headers):
            header_lower = str(header).lower()

            style_score = sum(10 if kw in header_lower else 0 for kw in style_keywords)
            if style_score > best_style_score:
                best_style_score = style_score
                style_idx = idx

            qty_score = sum(10 if kw in header_lower else 0 for kw in qty_keywords)
            if qty_score > best_qty_score:
                best_qty_score = qty_score
                qty_idx = idx

            cost_score = sum(10 if kw in header_lower else 0 for kw in cost_keywords)
            if cost_score > best_cost_score:
                best_cost_score = cost_score
                cost_idx = idx

            desc_score = sum(10 if kw in header_lower else 0 for kw in desc_keywords)
            if desc_score > best_desc_score:
                best_desc_score = desc_score
                desc_idx = idx

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

            parts = re.split(r'\s{2,}|\t', line)
            if len(parts) == 1:
                parts = line.split()

            has_letters = any(re.search(r'[A-Za-z]', str(p)) for p in parts)
            has_numbers = any(re.search(r'\d', str(p)) for p in parts)

            if has_letters and has_numbers and len(parts) >= 2:
                data_rows.append(parts)

        return data_rows

    def read_pdf(self, file):
        try:
            data = []

            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()

                    if tables:
                        for table in tables:
                            if not table or len(table) < 2:
                                continue

                            style_idx, qty_idx, cost_idx, desc_idx = self.smart_column_detection_pdf(table[0])

                            # Skip tables that don't have at least style column AND (quantity or cost) detected
                            if style_idx < 0 or (qty_idx < 0 and cost_idx < 0):
                                continue

                            for row in table[1:]:
                                if not row or all(not cell for cell in row):
                                    continue

                                vendor_style = str(row[style_idx]).strip() if style_idx >= 0 and style_idx < len(row) and row[style_idx] else ""
                                quantity = str(row[qty_idx]).strip() if qty_idx >= 0 and qty_idx < len(row) and row[qty_idx] else ""
                                cost = str(row[cost_idx]).strip() if cost_idx >= 0 and cost_idx < len(row) and row[cost_idx] else ""
                                description = str(row[desc_idx]).strip() if desc_idx >= 0 and desc_idx < len(row) and row[desc_idx] else ""

                                # Clean up vendor style - remove newlines and special chars
                                if vendor_style:
                                    vendor_style = ' '.join(vendor_style.split())
                                    # Remove non-alphanumeric chars except hyphens and slashes
                                    vendor_style = re.sub(r'[^\w\-/]', '', vendor_style)

                                # Clean up quantity - extract just the number
                                if quantity:
                                    quantity = re.sub(r'[^\d.]', '', quantity)

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

                                vendor_style_lower = vendor_style.lower()

                                # Skip header-like values
                                if vendor_style_lower in ['nan', 'none', 'style', 'sku', 'item', 'product', 'model', 'no', 'modelno']:
                                    continue

                                # Final safety check: skip if vendor style is STILL a brand name (shouldn't happen after swap)
                                if vendor_style_lower in brand_names:
                                    continue

                                # Skip rows with keywords that indicate non-product lines
                                skip_keywords = ['insurance', 'charge', 'shipping', 'freight', 'tax', 'subtotal', 'total',
                                               'sales', 'sale', 'discount', 'remit', 'balance', 'payment', 'invoice',
                                               'pursuant', 'article', 'chapter', 'division', 'summary']
                                if any(keyword in vendor_style_lower for keyword in skip_keywords):
                                    continue

                                # Skip if cost is invalid (0.00 or empty after cleaning)
                                if cost and (float(cost) if cost.replace('.', '').isdigit() else -1) == 0.0:
                                    continue

                                # Ensure description is never empty
                                if not description or description.strip() == '':
                                    description = 'New SKU, please add description.'

                                # Append the data
                                data.append({
                                    'Vendor Style #': vendor_style,
                                    'Quantity': quantity if quantity else '',
                                    'Cost': cost if cost else '',
                                    'Description': description
                                })
                    else:
                        text_rows = self.extract_text_based_data(page)

                        if text_rows:

                            for row_parts in text_rows:
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
                                    skip_keywords = ['insurance', 'charge', 'shipping', 'freight', 'tax', 'subtotal', 'total',
                                                    'sales', 'sale', 'discount', 'remit', 'balance', 'payment', 'invoice',
                                                    'pursuant', 'article', 'chapter', 'division', 'summary', 'grand', 'amount']

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

                                    data.append({
                                        'Vendor Style #': vendor_style,
                                        'Quantity': quantity,
                                        'Cost': cost,
                                        'Description': description
                                    })

            return data
        except Exception as e:
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

            importer = InvoiceImporter()

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
                                            if text:
                                                lines = text.split('\n')[:20]  # First 20 lines
                                                st.write(f"\n- Text lines: {len(text.split(chr(10)))}")
                                                st.write("- Sample lines:")
                                                for i, line in enumerate(lines):
                                                    if line.strip():
                                                        st.code(f"{i}: {line[:100]}")
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
        1. **Upload your invoice file** - Select an Excel (.xlsx, .xls) or PDF file
        2. **Click 'Process File'** - The application will automatically detect columns
        3. **Review the data** - Check the extracted data in the preview table
        4. **Download results** - Click the download button to save as Excel

        ### What gets extracted:
        - **Vendor Style #**: SKU, item code, or product number
        - **Quantity**: Number of items ordered
        - **Cost**: Unit cost or price
        - **Description**: Product description (when available)

        ### Supported formats:
        - Excel files with table headers
        - PDF files with structured tables
        - PDF files with text-based invoices
        """)


if __name__ == "__main__":
    main()

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

        if best_match:
            st.write(f"✓ Matched {column_type}: '{best_match}' (score: {best_score})")

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
            st.write("⚙️ No columns detected by keywords, attempting pattern analysis...")
            for col in df.columns:
                sample_values = df[col].dropna().head(10)

                has_alphanumeric = any(bool(re.search(r'[a-zA-Z]', str(v))) and
                                      bool(re.search(r'\d', str(v))) for v in sample_values)
                has_numeric = all(str(v).replace('.', '').replace(',', '').replace('-', '').isdigit()
                                 for v in sample_values if str(v).strip())

                if not style_col and has_alphanumeric and len(sample_values) > 0:
                    style_col = col
                    st.write(f"✓ Pattern matched style column: {col}")
                elif not qty_col and has_numeric and not style_col == col:
                    if all(float(str(v).replace(',', '')) < 10000 for v in sample_values if str(v).strip()):
                        qty_col = col
                        st.write(f"✓ Pattern matched quantity column: {col}")
                elif not cost_col and has_numeric and col != qty_col:
                    cost_col = col
                    st.write(f"✓ Pattern matched cost column: {col}")

        return style_col, qty_col, cost_col, desc_col

    def read_excel(self, file):
        try:
            df_raw = pd.read_excel(file, header=None)

            st.write(f"📊 Initial read - {len(df_raw)} rows and {len(df_raw.columns)} columns")

            header_row = None
            for idx in range(min(15, len(df_raw))):
                row_values = [str(v).lower() for v in df_raw.iloc[idx] if pd.notna(v)]
                if len(row_values) < 2:
                    continue

                non_numeric_count = sum(1 for v in row_values if not str(v).replace('.', '').replace(',', '').isdigit())

                if non_numeric_count >= 2:
                    header_row = idx
                    st.write(f"✓ Found header row at index {idx}")
                    break

            if header_row is not None:
                df = pd.read_excel(file, header=header_row)
            else:
                df = pd.read_excel(file)
                st.write("⚙️ Using default header row")

            st.write(f"📋 Found {len(df)} rows and {len(df.columns)} columns")
            with st.expander("View detected column names"):
                st.write(list(df.columns))

            style_col, qty_col, cost_col, desc_col = self.smart_column_detection(df)

            if not style_col:
                st.warning("⚠️ Could not auto-detect style column")
            if not qty_col:
                st.warning("⚠️ Could not auto-detect quantity column")
            if not cost_col:
                st.warning("⚠️ Could not auto-detect cost column")
            if not desc_col:
                st.info("ℹ️ Could not auto-detect description column")

            data = []
            for idx, row in df.iterrows():
                vendor_style = str(row[style_col]).strip() if style_col and pd.notna(row[style_col]) else ""
                quantity = str(row[qty_col]).strip() if qty_col and pd.notna(row[qty_col]) else ""
                cost = str(row[cost_col]).strip() if cost_col and pd.notna(row[cost_col]) else ""
                description = str(row[desc_col]).strip() if desc_col and pd.notna(row[desc_col]) else ""

                if not description or description.lower() in ['nan', 'none']:
                    description = 'New SKU, please add description.'

                if vendor_style and vendor_style.lower() not in ['nan', 'none', 'style', 'sku', 'item', 'product']:
                    data.append({
                        'Vendor Style #': vendor_style,
                        'Quantity': quantity if quantity and quantity.lower() != 'nan' else '',
                        'Cost': cost if cost and cost.lower() != 'nan' else '',
                        'Description': description
                    })

            st.success(f"✅ Total rows extracted: {len(data)}")
            return data
        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")

    def smart_column_detection_pdf(self, headers):
        """Intelligently detect columns for PDF tables"""
        style_keywords = ['style', 'sku', 'item', 'product', 'prod', 'part', 'number',
                         'code', 'article', 'model', 'ref', 'reference', 'material']
        qty_keywords = ['qty', 'quantity', 'amount', 'count', 'units', 'ord', 'ordered',
                       'pieces', 'pcs', 'pack', 'carton']
        cost_keywords = ['cost', 'price', 'rate', 'total', 'value', 'amount', 'unit',
                        'unitprice', 'each', 'perpcs', 'charge']

        style_idx = -1
        qty_idx = -1
        cost_idx = -1
        best_style_score = 0
        best_qty_score = 0
        best_cost_score = 0

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

        return style_idx, qty_idx, cost_idx

    def extract_text_based_data(self, page):
        """Extract data from PDF text when tables aren't detected"""
        text = page.extract_text()
        if not text:
            return []

        lines = text.split('\n')
        st.write(f"📄 Extracted {len(lines)} lines of text from PDF")

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
                st.write(f"📄 PDF has {len(pdf.pages)} pages")

                for page_num, page in enumerate(pdf.pages):
                    st.write(f"⚙️ Processing page {page_num + 1}")
                    tables = page.extract_tables()

                    if tables:
                        st.write(f"✓ Found {len(tables)} tables on page {page_num + 1}")

                        for table_num, table in enumerate(tables):
                            if not table or len(table) < 2:
                                continue

                            style_idx, qty_idx, cost_idx = self.smart_column_detection_pdf(table[0])

                            for row in table[1:]:
                                if not row or all(not cell for cell in row):
                                    continue

                                vendor_style = str(row[style_idx]).strip() if style_idx >= 0 and style_idx < len(row) and row[style_idx] else ""
                                quantity = str(row[qty_idx]).strip() if qty_idx >= 0 and qty_idx < len(row) and row[qty_idx] else ""
                                cost = str(row[cost_idx]).strip() if cost_idx >= 0 and cost_idx < len(row) and row[cost_idx] else ""

                                if vendor_style and vendor_style.lower() not in ['nan', 'none', 'style', 'sku', 'item', 'product']:
                                    data.append({
                                        'Vendor Style #': vendor_style,
                                        'Quantity': quantity if quantity and quantity.lower() != 'nan' else '',
                                        'Cost': cost if cost and cost.lower() != 'nan' else '',
                                        'Description': 'New SKU, please add description.'
                                    })
                    else:
                        st.write("⚙️ No tables found, trying text extraction...")
                        text_rows = self.extract_text_based_data(page)

                        if text_rows:
                            st.write(f"✓ Found {len(text_rows)} potential data rows in text")

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

                                    if re.match(r'^[A-Z]{2}[0-9]{4}', part):
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
                                    if qty_fields:
                                        quantity = qty_fields[0]
                                    cost = price_fields[-2] if len(price_fields) >= 2 else price_fields[-1]

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

                                    data.append({
                                        'Vendor Style #': vendor_style,
                                        'Quantity': quantity,
                                        'Cost': cost,
                                        'Description': description
                                    })

            st.success(f"✅ Total rows extracted from PDF: {len(data)}")
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
    The application will automatically detect the correct columns and export to Excel format.
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
            importer = InvoiceImporter()

            with st.spinner("Processing file..."):
                try:
                    st.subheader("📊 Processing Details")

                    if file_ext in ['.xlsx', '.xls']:
                        st.info("📊 Detected Excel file")
                        data = importer.read_excel(uploaded_file)
                    elif file_ext == '.pdf':
                        st.info("📄 Detected PDF file")
                        data = importer.read_pdf(uploaded_file)
                    else:
                        st.error("❌ Unsupported file format")
                        return

                    if not data:
                        st.warning("⚠️ No data could be extracted from the file.")
                        return

                    st.divider()
                    st.subheader("📋 Extracted Data Preview")

                    output_df = pd.DataFrame(data)
                    st.dataframe(output_df, use_container_width=True)

                    st.metric("Total Rows", len(data))

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

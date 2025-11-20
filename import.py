import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import pandas as pd
import pdfplumber
import os
from datetime import datetime
import re


class InvoiceImporter:
    def __init__(self, root):
        self.root = root
        self.root.title("Invoice Import Tool")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        self.selected_file = None
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        title_label = ttk.Label(
            main_frame,
            text="Invoice Import Tool",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=20)

        instructions = ttk.Label(
            main_frame,
            text="Select an Excel or PDF file to import invoice data.\n"
                 "The output will contain: Vendor Style #, Quantity, Cost, and Description.",
            font=("Arial", 10),
            justify=tk.CENTER
        )
        instructions.grid(row=1, column=0, columnspan=2, pady=10)

        self.file_label = ttk.Label(
            main_frame,
            text="No file selected",
            font=("Arial", 9),
            foreground="gray"
        )
        self.file_label.grid(row=2, column=0, columnspan=2, pady=10)

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)

        select_btn = ttk.Button(
            button_frame,
            text="Select File",
            command=self.select_file,
            width=20
        )
        select_btn.grid(row=0, column=0, padx=5)

        process_btn = ttk.Button(
            button_frame,
            text="Process & Export",
            command=self.process_file,
            width=20
        )
        process_btn.grid(row=0, column=1, padx=5)

        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            length=400
        )
        self.progress.grid(row=4, column=0, columnspan=2, pady=10)

        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            font=("Arial", 9),
            foreground="green"
        )
        self.status_label.grid(row=5, column=0, columnspan=2, pady=5)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Invoice File",
            filetypes=[
                ("Supported Files", "*.xlsx *.xls *.pdf"),
                ("Excel Files", "*.xlsx *.xls"),
                ("PDF Files", "*.pdf"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            self.selected_file = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Selected: {filename}", foreground="black")
            self.status_label.config(text="File selected - ready to process", foreground="green")

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
            print(f"DEBUG: Best match for {column_type}: '{best_match}' (score: {best_score})")

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
            print("DEBUG: No columns detected, attempting pattern analysis...")
            for col in df.columns:
                sample_values = df[col].dropna().head(10)

                has_alphanumeric = any(bool(re.search(r'[a-zA-Z]', str(v))) and
                                      bool(re.search(r'\d', str(v))) for v in sample_values)
                has_numeric = all(str(v).replace('.', '').replace(',', '').replace('-', '').isdigit()
                                 for v in sample_values if str(v).strip())

                if not style_col and has_alphanumeric and len(sample_values) > 0:
                    style_col = col
                    print(f"DEBUG: Pattern matched style column: {col}")
                elif not qty_col and has_numeric and not style_col == col:
                    if all(float(str(v).replace(',', '')) < 10000 for v in sample_values if str(v).strip()):
                        qty_col = col
                        print(f"DEBUG: Pattern matched quantity column: {col}")
                elif not cost_col and has_numeric and col != qty_col:
                    cost_col = col
                    print(f"DEBUG: Pattern matched cost column: {col}")

        return style_col, qty_col, cost_col, desc_col

    def read_excel(self, file_path):
        try:
            df_raw = pd.read_excel(file_path, header=None)

            print(f"DEBUG: Initial read - {len(df_raw)} rows and {len(df_raw.columns)} columns")

            header_row = None
            for idx in range(min(15, len(df_raw))):
                row_values = [str(v).lower() for v in df_raw.iloc[idx] if pd.notna(v)]
                if len(row_values) < 2:
                    continue

                print(f"DEBUG: Row {idx} sample: {row_values[:5]}")

                non_numeric_count = sum(1 for v in row_values if not str(v).replace('.', '').replace(',', '').isdigit())

                if non_numeric_count >= 2:
                    header_row = idx
                    print(f"DEBUG: Found likely header row at index {idx}")
                    break

            if header_row is not None:
                df = pd.read_excel(file_path, header=header_row)
            else:
                df = pd.read_excel(file_path)
                print("DEBUG: Using default header row")

            print(f"DEBUG: Found {len(df)} rows and {len(df.columns)} columns")
            print(f"DEBUG: Column names: {list(df.columns)}")

            style_col, qty_col, cost_col, desc_col = self.smart_column_detection(df)

            if not style_col:
                print("WARNING: Could not auto-detect style column")
            if not qty_col:
                print("WARNING: Could not auto-detect quantity column")
            if not cost_col:
                print("WARNING: Could not auto-detect cost column")
            if not desc_col:
                print("WARNING: Could not auto-detect description column")

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

            print(f"DEBUG: Total rows extracted: {len(data)}")
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
            header_clean = re.sub(r'[^a-z0-9]', '', header_lower)

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
        print(f"DEBUG: Extracted {len(lines)} lines of text")

        for i, line in enumerate(lines[:10]):
            print(f"DEBUG: Sample line {i}: {line[:100]}")

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
                print(f"DEBUG: Text line matched: {parts}")

        return data_rows

    def read_pdf(self, file_path):
        try:
            data = []

            with pdfplumber.open(file_path) as pdf:
                print(f"DEBUG: PDF has {len(pdf.pages)} pages")

                for page_num, page in enumerate(pdf.pages):
                    print(f"DEBUG: Processing page {page_num + 1}")
                    tables = page.extract_tables()
                    print(f"DEBUG: Found {len(tables)} tables on page {page_num + 1}")

                    if tables:
                        for table_num, table in enumerate(tables):
                            if not table or len(table) < 2:
                                print(f"DEBUG: Skipping table {table_num} - too small")
                                continue

                            print(f"DEBUG: Table {table_num} has {len(table)} rows and {len(table[0]) if table[0] else 0} columns")
                            print(f"DEBUG: Headers: {table[0]}")

                            style_idx, qty_idx, cost_idx = self.smart_column_detection_pdf(table[0])

                            if style_idx >= 0:
                                print(f"DEBUG: Matched style column at index {style_idx}: {table[0][style_idx]}")
                            if qty_idx >= 0:
                                print(f"DEBUG: Matched quantity column at index {qty_idx}: {table[0][qty_idx]}")
                            if cost_idx >= 0:
                                print(f"DEBUG: Matched cost column at index {cost_idx}: {table[0][cost_idx]}")

                            for row_num, row in enumerate(table[1:]):
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
                        print("DEBUG: No tables found, trying text extraction...")
                        text_rows = self.extract_text_based_data(page)

                        if text_rows:
                            print(f"DEBUG: Found {len(text_rows)} potential data rows in text")

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
                                        print(f"DEBUG: Found vendor style pattern: {part}")
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
                                    print(f"DEBUG: Extracted - Style: {vendor_style}, Qty: {quantity}, Cost: {cost}, Desc: {description}")

            print(f"DEBUG: Total rows extracted from PDF: {len(data)}")
            return data
        except Exception as e:
            raise Exception(f"Error reading PDF file: {str(e)}")

    def process_file(self):
        if not self.selected_file:
            messagebox.showwarning("No File Selected", "Please select a file first.")
            return

        self.progress.start()
        self.status_label.config(text="Processing file...", foreground="orange")
        self.root.update()

        try:
            file_ext = os.path.splitext(self.selected_file)[1].lower()

            if file_ext in ['.xlsx', '.xls']:
                data = self.read_excel(self.selected_file)
            elif file_ext == '.pdf':
                data = self.read_pdf(self.selected_file)
            else:
                raise Exception("Unsupported file format")

            if not data:
                messagebox.showwarning("No Data", "No data could be extracted from the file.")
                self.progress.stop()
                self.status_label.config(text="No data found", foreground="orange")
                return

            output_df = pd.DataFrame(data)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"invoice_import_{timestamp}.xlsx"
            output_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                initialfile=output_filename,
                filetypes=[("Excel Files", "*.xlsx")]
            )

            if output_path:
                output_df.to_excel(output_path, index=False)

                self.progress.stop()
                self.status_label.config(text="Export completed successfully!", foreground="green")
                messagebox.showinfo(
                    "Success",
                    f"File processed successfully!\n\n"
                    f"Rows extracted: {len(data)}\n"
                    f"Saved to: {os.path.basename(output_path)}"
                )
            else:
                self.progress.stop()
                self.status_label.config(text="Export cancelled", foreground="gray")

        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="Error occurred", foreground="red")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


def main():
    root = tk.Tk()
    app = InvoiceImporter(root)
    root.mainloop()


if __name__ == "__main__":
    main()

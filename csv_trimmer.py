import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
import numpy as np
import os

class CSVTrimmer:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV Visual Trimmer")
        self.root.geometry("1200x800")
        
        # Data variables
        self.df = None
        self.current_file = None
        self.selected_column = None
        self.crop_start = None
        self.crop_end = None
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Load CSV File", command=self.load_file).grid(row=0, column=0, padx=(0, 10))
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Column selection frame
        column_frame = ttk.LabelFrame(main_frame, text="Column Selection", padding="5")
        column_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(column_frame, text="Select column to plot:").grid(row=0, column=0, padx=(0, 10))
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(column_frame, textvariable=self.column_var, state="readonly")
        self.column_combo.grid(row=0, column=1, padx=(0, 10))
        self.column_combo.bind('<<ComboboxSelected>>', self.on_column_selected)
        
        ttk.Button(column_frame, text="Plot", command=self.plot_data).grid(row=0, column=2)
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Data Visualization", padding="5")
        plot_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Trimming Controls", padding="5")
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Crop range display
        ttk.Label(control_frame, text="Selected range:").grid(row=0, column=0, padx=(0, 10))
        self.range_label = ttk.Label(control_frame, text="No range selected")
        self.range_label.grid(row=0, column=1, padx=(0, 20))
        
        # Manual range entry
        ttk.Label(control_frame, text="Start (s):").grid(row=0, column=2, padx=(0, 5))
        self.start_var = tk.StringVar()
        self.start_entry = ttk.Entry(control_frame, textvariable=self.start_var, width=10)
        self.start_entry.grid(row=0, column=3, padx=(0, 10))
        
        ttk.Label(control_frame, text="End (s):").grid(row=0, column=4, padx=(0, 5))
        self.end_var = tk.StringVar()
        self.end_entry = ttk.Entry(control_frame, textvariable=self.end_var, width=10)
        self.end_entry.grid(row=0, column=5, padx=(0, 10))
        
        ttk.Button(control_frame, text="Set Range", command=self.set_manual_range).grid(row=0, column=6, padx=(0, 10))
        ttk.Button(control_frame, text="Clear Selection", command=self.clear_selection).grid(row=0, column=7, padx=(0, 10))
        ttk.Button(control_frame, text="Crop & Save", command=self.crop_and_save).grid(row=0, column=8)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))
        status_frame.columnconfigure(0, weight=1)
        
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.grid(row=0, column=0, sticky=(tk.W,))
        
        # Initialize span selector (will be created when plotting)
        self.span_selector = None
        
    def load_file(self):
        """Load CSV file and populate column selection"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="c:/Users/Alvaro/Documents/GitHub/Human-Activity-Recognition_UsingML/data"
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.current_file = file_path
                
                # Validate required columns
                if 'seconds_elapsed' not in self.df.columns:
                    messagebox.showerror("Error", "CSV file must contain 'seconds_elapsed' column")
                    return
                
                # Get available columns (excluding first two: time and seconds_elapsed)
                available_columns = [col for col in self.df.columns if col not in ['time', 'seconds_elapsed']]
                
                if not available_columns:
                    messagebox.showerror("Error", "No plottable columns found (excluding time and seconds_elapsed)")
                    return
                
                # Update GUI
                self.file_label.config(text=os.path.basename(file_path))
                self.column_combo['values'] = available_columns
                self.column_combo.set('')  # Clear selection
                
                # Clear previous plot
                self.ax.clear()
                self.canvas.draw()
                
                # Reset crop range
                self.crop_start = None
                self.crop_end = None
                self.update_range_display()
                
                self.status_label.config(text=f"Loaded {len(self.df)} rows from {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                
    def on_column_selected(self, event=None):
        """Handle column selection"""
        self.selected_column = self.column_var.get()
        
    def plot_data(self):
        """Plot selected column against seconds_elapsed"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return
            
        if not self.selected_column:
            messagebox.showwarning("Warning", "Please select a column to plot")
            return
            
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Plot data
            x_data = self.df['seconds_elapsed']
            y_data = self.df[self.selected_column]
            
            self.ax.plot(x_data, y_data, 'b-', linewidth=1, alpha=0.8)
            self.ax.set_xlabel('Seconds Elapsed')
            self.ax.set_ylabel(self.selected_column)
            self.ax.set_title(f'{self.selected_column} vs Time')
            self.ax.grid(True, alpha=0.3)
            
            # Create span selector for interactive selection
            self.span_selector = SpanSelector(
                self.ax, 
                self.on_span_select, 
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='red'),
                interactive=True
            )
            
            # Refresh canvas
            self.canvas.draw()
            
            self.status_label.config(text=f"Plotted {self.selected_column}. Click and drag to select time range.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot data: {str(e)}")
            
    def on_span_select(self, xmin, xmax):
        """Handle span selection on plot"""
        self.crop_start = xmin
        self.crop_end = xmax
        self.update_range_display()
        
        # Update entry fields
        self.start_var.set(f"{xmin:.3f}")
        self.end_var.set(f"{xmax:.3f}")
        
    def set_manual_range(self):
        """Set crop range manually from entry fields"""
        try:
            start = float(self.start_var.get())
            end = float(self.end_var.get())
            
            if start >= end:
                messagebox.showerror("Error", "Start time must be less than end time")
                return
                
            if self.df is not None:
                min_time = self.df['seconds_elapsed'].min()
                max_time = self.df['seconds_elapsed'].max()
                
                if start < min_time or end > max_time:
                    messagebox.showwarning("Warning", f"Range should be between {min_time:.3f} and {max_time:.3f}")
                    
            self.crop_start = start
            self.crop_end = end
            self.update_range_display()
            
            # Update plot selection if span selector exists
            if self.span_selector:
                self.span_selector.set_visible(True)
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
            
    def clear_selection(self):
        """Clear crop range selection"""
        self.crop_start = None
        self.crop_end = None
        self.start_var.set('')
        self.end_var.set('')
        self.update_range_display()
        
        # Clear span selector
        if self.span_selector:
            self.span_selector.set_visible(False)
            self.canvas.draw()
            
    def update_range_display(self):
        """Update the range display label"""
        if self.crop_start is not None and self.crop_end is not None:
            duration = self.crop_end - self.crop_start
            self.range_label.config(text=f"{self.crop_start:.3f}s to {self.crop_end:.3f}s (duration: {duration:.3f}s)")
        else:
            self.range_label.config(text="No range selected")
            
    def crop_and_save(self):
        """Crop data to selected range and save to original file"""
        if self.df is None:
            messagebox.showwarning("Warning", "Please load a CSV file first")
            return
            
        if self.crop_start is None or self.crop_end is None:
            messagebox.showwarning("Warning", "Please select a time range to crop")
            return
            
        try:
            # Filter data within the selected range
            mask = (self.df['seconds_elapsed'] >= self.crop_start) & (self.df['seconds_elapsed'] <= self.crop_end)
            cropped_df = self.df[mask].copy()
            
            if len(cropped_df) == 0:
                messagebox.showwarning("Warning", "No data points found in the selected range")
                return
                
            # Reset seconds_elapsed to start from the minimum value in the cropped range
            min_seconds = cropped_df['seconds_elapsed'].min()
            cropped_df['seconds_elapsed'] = cropped_df['seconds_elapsed'] - min_seconds
            
            # Confirm save operation
            result = messagebox.askyesno(
                "Confirm Save", 
                f"This will overwrite the original file with {len(cropped_df)} rows.\n"
                f"Original file had {len(self.df)} rows.\n\n"
                f"Do you want to continue?"
            )
            
            if result:
                # Save to original file
                cropped_df.to_csv(self.current_file, index=False)
                
                # Update current dataframe
                self.df = cropped_df
                
                # Clear selection
                self.clear_selection()
                
                # Replot if a column was selected
                if self.selected_column:
                    self.plot_data()
                    
                self.status_label.config(text=f"File saved successfully. {len(cropped_df)} rows remaining.")
                messagebox.showinfo("Success", "File has been cropped and saved successfully!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {str(e)}")

def main():
    root = tk.Tk()
    app = CSVTrimmer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
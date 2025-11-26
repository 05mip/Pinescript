import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import pandas as pd
import numpy as np
import os
import glob

class VisualLabelingTool:
    def __init__(self):
        self.current_labels = {}
        self.current_symbol = None
        self.current_data = None
        self.fig = None
        self.ax = None
        self.current_mode = 2  # 0=Sell, 2=Buy, -1=Unlabel (removed Hold mode)
        self.zoom_start = 0
        self.zoom_window = 100  # Show 100 data points at a time
        self.data_folder = "data_30m"
        self.current_file_index = 0
        self.csv_files = []
        
    def get_csv_files(self):
        """Get all CSV files from the data_30m folder"""
        pattern = os.path.join(self.data_folder, "*.csv")
        self.csv_files = glob.glob(pattern)
        self.csv_files.sort()  # Sort for consistent ordering
        print(f"Found {len(self.csv_files)} CSV files to process")
        return self.csv_files
    
    def start_labeling_all(self, zoom_window=100):
        """Start labeling all files in the data_30m folder"""
        self.zoom_window = zoom_window
        self.get_csv_files()
        
        if not self.csv_files:
            print(f"No CSV files found in {self.data_folder}")
            return
        
        print("Starting to label all files. Press 'n' to go to next file, 'p' for previous file.")
        self.current_file_index = 0
        self.start_labeling_current_file()
    
    def start_labeling_current_file(self):
        """Start labeling the current file"""
        if self.current_file_index >= len(self.csv_files):
            print("All files processed!")
            return
        
        csv_file = self.csv_files[self.current_file_index]
        symbol = os.path.basename(csv_file).replace('_30m.csv', '')
        
        print(f"\nProcessing file {self.current_file_index + 1}/{len(self.csv_files)}: {symbol}")
        
        # Load data from CSV file
        self.current_data = self.load_data_from_csv(csv_file)
        if self.current_data is None:
            print(f"Could not load data for {symbol}")
            self.current_file_index += 1
            self.start_labeling_current_file()
            return
        
        self.current_symbol = symbol
        
        # Initialize labels as None (will be filled automatically)
        self.current_labels = {}
        
        # Create the plot
        self.create_plot()
    
    def load_data_from_csv(self, csv_file):
        """Load data from a CSV file"""
        try:
            # Read CSV with proper datetime parsing
            df = pd.read_csv(csv_file, skiprows=2)  # Skip the header rows
            
            # The first column contains datetime but has no name, so we need to handle it differently
            # Get the first column as datetime
            datetime_col = df.iloc[:, 0]
            df['Datetime'] = pd.to_datetime(datetime_col)
            
            # Get the data columns (skip the first column which was datetime)
            data_df = df.iloc[:, 1:]
            data_df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']  # Set proper column names
            data_df['Datetime'] = df['Datetime']
            data_df.set_index('Datetime', inplace=True)
            
            # Convert to numeric, handling any non-numeric values
            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
            
            # Drop any rows with NaN values
            data_df = data_df.dropna()
            
            if len(data_df) == 0:
                print(f"No valid data found in {csv_file}")
                return None
            
            print(f"Loaded {len(data_df)} data points from {csv_file}")
            return data_df
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            return None
    
    def start_labeling(self, symbol, zoom_window=100):
        """Start the visual labeling process for a single symbol"""
        self.current_symbol = symbol
        self.zoom_window = zoom_window
        
        # Load data from CSV file
        csv_file = os.path.join(self.data_folder, f"{symbol}_30m.csv")
        self.current_data = self.load_data_from_csv(csv_file)
        if self.current_data is None:
            print(f"Could not load data for {symbol}")
            return
        
        # Initialize labels as None (will be filled automatically)
        self.current_labels = {}
        
        # Create the plot
        self.create_plot()
        
    def create_plot(self):
        """Create the interactive plot"""
        # Clear any existing plot
        plt.close('all')
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        
        # Create buttons
        self.create_buttons()
        
        # Plot initial data
        self.update_plot()
        
        # Connect click events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.show()
    
    def create_buttons(self):
        """Create control buttons"""
        # Button positions
        button_width = 0.08
        button_height = 0.04
        
        # Mode buttons (Buy, Sell, and Unlabel)
        ax_sell = plt.axes([0.02, 0.85, button_width, button_height])
        ax_buy = plt.axes([0.02, 0.80, button_width, button_height])
        ax_unlabel = plt.axes([0.02, 0.75, button_width, button_height])
        
        self.btn_sell = Button(ax_sell, 'Sell (0)', color='red', hovercolor='lightcoral')
        self.btn_buy = Button(ax_buy, 'Buy (2)', color='green', hovercolor='lightgreen')
        self.btn_unlabel = Button(ax_unlabel, 'Unlabel (X)', color='gray', hovercolor='lightgray')
        
        # Navigation buttons
        ax_prev = plt.axes([0.02, 0.65, button_width, button_height])
        ax_next = plt.axes([0.02, 0.60, button_width, button_height])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        
        # File navigation buttons
        ax_prev_file = plt.axes([0.02, 0.50, button_width, button_height])
        ax_next_file = plt.axes([0.02, 0.45, button_width, button_height])
        
        self.btn_prev_file = Button(ax_prev_file, 'Prev File')
        self.btn_next_file = Button(ax_next_file, 'Next File')
        
        # Save button
        ax_save = plt.axes([0.02, 0.35, button_width, button_height])
        self.btn_save = Button(ax_save, 'Save Labels')
        
        # Connect button events
        self.btn_sell.on_clicked(lambda x: self.set_mode(0))
        self.btn_buy.on_clicked(lambda x: self.set_mode(2))
        self.btn_unlabel.on_clicked(lambda x: self.set_mode(-1))
        self.btn_prev.on_clicked(lambda x: self.navigate(-1))
        self.btn_next.on_clicked(lambda x: self.navigate(1))
        self.btn_prev_file.on_clicked(lambda x: self.navigate_file(-1))
        self.btn_next_file.on_clicked(lambda x: self.navigate_file(1))
        self.btn_save.on_clicked(lambda x: self.save_labels())
    
    def set_mode(self, mode):
        """Set the current labeling mode"""
        self.current_mode = mode
        mode_names = ['SELL', 'UNLABEL', 'BUY']
        mode_index = 0 if mode == 0 else 2 if mode == 2 else 1
        print(f"Mode set to: {mode_names[mode_index]}")
        
        # Update button colors to show current mode
        self.btn_sell.color = 'red' if mode == 0 else 'lightgray'
        self.btn_buy.color = 'green' if mode == 2 else 'lightgray'
        self.btn_unlabel.color = 'gray' if mode == -1 else 'lightgray'
        
        self.fig.canvas.draw()
    
    def navigate(self, direction):
        """Navigate through the data"""
        new_start = self.zoom_start + (direction * self.zoom_window // 2)
        new_start = max(0, min(new_start, len(self.current_data) - self.zoom_window))
        
        if new_start != self.zoom_start:
            self.zoom_start = new_start
            self.update_plot()
    
    def navigate_file(self, direction):
        """Navigate to next/previous file"""
        new_index = self.current_file_index + direction
        if 0 <= new_index < len(self.csv_files):
            self.current_file_index = new_index
            plt.close(self.fig)  # Close current plot
            self.start_labeling_current_file()
    
    def update_plot(self):
        """Update the plot with current data window"""
        self.ax.clear()
        
        # Get current data window
        end_idx = min(self.zoom_start + self.zoom_window, len(self.current_data))
        window_data = self.current_data.iloc[self.zoom_start:end_idx]
        
        # Plot candlestick-style price data
        for i, (timestamp, row) in enumerate(window_data.iterrows()):
            color = 'green' if row['Close'] > row['Open'] else 'red'
            
            # Plot high-low line
            self.ax.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1)
            
            # Plot open-close box
            height = abs(row['Close'] - row['Open'])
            bottom = min(row['Open'], row['Close'])
            
            rect = patches.Rectangle((i-0.3, bottom), 0.6, height, 
                                   facecolor=color, alpha=0.7, edgecolor='black')
            self.ax.add_patch(rect)
            
            # Color background based on label
            label = self.current_labels.get(timestamp, None)
            if label == 0:  # Sell
                bg_color = 'lightcoral'
            elif label == 2:  # Buy
                bg_color = 'lightgreen'
            elif label == -1: # Unlabeled
                bg_color = 'lightyellow'
            else:  # Hold or unlabeled
                bg_color = 'lightyellow'
            
            self.ax.axvspan(i-0.5, i+0.5, alpha=0.3, color=bg_color)
        
        # Set labels and title
        file_info = f"File {self.current_file_index + 1}/{len(self.csv_files)}" if self.csv_files else ""
        self.ax.set_title(f'{self.current_symbol} - Visual Labeling Tool {file_info}\n'
                         f'Showing {self.zoom_start} to {end_idx} of {len(self.current_data)} points')
        self.ax.set_xlabel('Time Index')
        self.ax.set_ylabel('Price')
        
        # Add legend
        legend_elements = [
            patches.Patch(color='lightcoral', label='Sell (0)'),
            patches.Patch(color='lightyellow', label='Hold (auto)'),
            patches.Patch(color='lightgreen', label='Buy (2)')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')
        
        # Add instructions
        instructions = ("Click on candlesticks to label them\n"
                       "Use buttons or keys: 0=Sell, 2=Buy, X=Unlabel\n"
                       "Arrow keys: navigate, S=Save\n"
                       "N/P: next/prev file")
        self.ax.text(0.02, 0.02, instructions, transform=self.ax.transAxes, 
                    fontsize=10, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return
        
        # Get the clicked index
        clicked_idx = int(round(event.xdata))
        
        if 0 <= clicked_idx < self.zoom_window:
            actual_idx = self.zoom_start + clicked_idx
            if actual_idx < len(self.current_data):
                timestamp = self.current_data.index[actual_idx]
                old_label = self.current_labels.get(timestamp, None)
                
                if self.current_mode == -1:
                    # Unlabel mode - remove the label
                    if timestamp in self.current_labels:
                        del self.current_labels[timestamp]
                        print(f"Unlabeled {timestamp} (was {['SELL', 'HOLD', 'BUY'][old_label] if old_label is not None else 'unlabeled'})")
                    else:
                        print(f"{timestamp} was already unlabeled")
                else:
                    # Buy or Sell mode
                    self.current_labels[timestamp] = self.current_mode
                    label_names = ['SELL', 'HOLD', 'BUY']
                    print(f"Labeled {timestamp} as {label_names[self.current_mode]} (was {label_names[old_label] if old_label is not None else 'unlabeled'})")
                
                self.update_plot()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == '0':
            self.set_mode(0)
        elif event.key == '2':
            self.set_mode(2)
        elif event.key == 'x' or event.key == 'X':
            self.set_mode(-1)
        elif event.key == 'left':
            self.navigate(-1)
        elif event.key == 'right':
            self.navigate(1)
        elif event.key == 's':
            self.save_labels()
        elif event.key == 'n':
            self.navigate_file(1)
        elif event.key == 'p':
            self.navigate_file(-1)
    
    def fill_hold_positions(self):
        """Automatically fill hold positions between buy and sell signals"""
        if not self.current_labels:
            return
        
        # Get all timestamps in order
        timestamps = sorted(self.current_data.index)
        buy_positions = []
        sell_positions = []
        
        # Find all buy and sell positions
        for timestamp in timestamps:
            label = self.current_labels.get(timestamp)
            if label == 2:  # Buy
                buy_positions.append(timestamp)
            elif label == 0:  # Sell
                sell_positions.append(timestamp)
        
        # Fill hold positions between buy and sell
        for i, buy_time in enumerate(buy_positions):
            # Find the next sell after this buy
            next_sell = None
            for sell_time in sell_positions:
                if sell_time > buy_time:
                    next_sell = sell_time
                    break
            
            if next_sell:
                # Fill all positions between buy and sell as hold (1)
                for timestamp in timestamps:
                    if buy_time < timestamp < next_sell:
                        self.current_labels[timestamp] = 1
        
        print(f"Automatically filled hold positions between {len(buy_positions)} buy and {len(sell_positions)} sell signals")
    
    def save_labels(self):
        """Save the current labels to CSV"""
        if not self.current_labels:
            print("No labels to save")
            return
        
        # Fill hold positions automatically
        self.fill_hold_positions()
        
        # Convert to DataFrame
        labels_df = pd.DataFrame(list(self.current_labels.items()), 
                                columns=['Datetime', 'Label'])
        labels_df.set_index('Datetime', inplace=True)
        labels_df = labels_df.sort_index()  # Sort by datetime
        
        # Save to CSV
        filename = f"manual_labels_{self.current_symbol.replace('-', '_')}_visual.csv"
        labels_df.to_csv(filename)
        
        print(f"Saved {len(labels_df)} labels to {filename}")
        
        # Show label distribution
        label_counts = labels_df['Label'].value_counts()
        print(f"Label distribution: Sell={label_counts.get(0, 0)}, Hold={label_counts.get(1, 0)}, Buy={label_counts.get(2, 0)}")


# Example usage
if __name__ == "__main__":
    # Create the visual labeling tool
    labeling_tool = VisualLabelingTool()
    
    # Start labeling all files in data_30m folder
    labeling_tool.start_labeling_all(zoom_window=250)
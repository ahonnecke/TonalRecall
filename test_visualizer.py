import tkinter as tk
from tkinter import ttk
import time
from audio_detector import BassAudioDetector, list_audio_devices

# Bass guitar standard tuning frequencies (E1, A1, D2, G2)
BASS_FREQUENCIES = {
    'E1': 41.20,
    'A1': 55.00,
    'D2': 73.42,
    'G2': 98.00
}

class BassVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bass Frequency Visualizer")
        self.root.geometry("800x600")
        
        # Setup styles
        style = ttk.Style()
        style.configure("Green.Horizontal.TProgressbar", 
                       background='green',
                       troughcolor='black')
        
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Device selection frame
        self.device_frame = ttk.LabelFrame(self.main_frame, text="Audio Input", padding="5")
        self.device_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.device_var = tk.StringVar(value="Auto-detect")
        self.device_menu = ttk.Combobox(self.device_frame, 
                                      textvariable=self.device_var, 
                                      state="readonly")
        self.device_menu.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        self.refresh_btn = ttk.Button(self.device_frame, 
                                    text="Refresh", 
                                    command=self.refresh_devices)
        self.refresh_btn.grid(row=0, column=1, padx=5)
        
        # Frequency display
        self.freq_label = ttk.Label(self.main_frame, 
                                  text="Frequency: --", 
                                  font=('TkDefaultFont', 24))
        self.freq_label.grid(row=1, column=0, columnspan=2, pady=20)
        
        # Note display
        self.note_label = ttk.Label(self.main_frame, 
                                  text="Note: --", 
                                  font=('TkDefaultFont', 24))
        self.note_label.grid(row=2, column=0, columnspan=2, pady=20)
        
        # Confidence bar
        self.conf_frame = ttk.LabelFrame(self.main_frame, text="Confidence", padding="5")
        self.conf_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.conf_bar = ttk.Progressbar(self.conf_frame, 
                                      mode='determinate',
                                      style="Green.Horizontal.TProgressbar",
                                      length=300)
        self.conf_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5)
        
        # Reference frequencies
        self.ref_frame = ttk.LabelFrame(self.main_frame, text="Reference Notes", padding="5")
        self.ref_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        for i, (note, freq) in enumerate(BASS_FREQUENCIES.items()):
            label = ttk.Label(self.ref_frame, 
                            text=f"{note}: {freq:.1f} Hz",
                            font=('TkDefaultFont', 12))
            label.grid(row=i, column=0, sticky=tk.W, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, 
                                  textvariable=self.status_var, 
                                  relief=tk.SUNKEN)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.detector = None
        self.running = False
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        
        self.refresh_devices()
    
    def refresh_devices(self):
        """Refresh the list of audio devices"""
        devices = list_audio_devices()
        choices = ["Auto-detect"]
        choices.extend([f"ID: {id} - {name}" for id, name in devices])
        self.device_menu['values'] = choices
        if self.device_var.get() not in choices:
            self.device_var.set("Auto-detect")
    
    def get_selected_device(self):
        """Get the selected device ID"""
        if self.device_var.get() == "Auto-detect":
            return None
        # Extract device ID from the string "ID: X - Device Name"
        return int(self.device_var.get().split(" - ")[0].split(": ")[1])
    
    def get_note_name(self, frequency):
        """Get the closest note name based on frequency"""
        if frequency <= 0:
            return "No signal"
        
        closest_note = min(BASS_FREQUENCIES.items(), 
                          key=lambda x: abs(x[1] - frequency))
        diff = abs(closest_note[1] - frequency)
        
        # If within 2 Hz, consider it a match
        if diff <= 2:
            return f"{closest_note[0]} ({closest_note[1]:.1f} Hz)"
        return f"{frequency:.1f} Hz"
    
    def update_display(self):
        """Update the frequency display"""
        if self.running and self.detector:
            freq, conf = self.detector.get_frequency()
            
            # Update frequency and note display
            self.freq_label['text'] = f"Frequency: {freq:.1f} Hz"
            self.note_label['text'] = f"Note: {self.get_note_name(freq)}"
            
            # Update confidence bar
            self.conf_bar['value'] = conf * 100
            
            # Check for errors
            error = self.detector.get_error()
            if error:
                self.status_var.set(f"Error: {error}")
                self.running = False
            else:
                self.status_var.set("Running")
            
            # Schedule next update
            self.root.after(50, self.update_display)
    
    def start_detection(self):
        """Start audio detection"""
        if not self.running:
            device_id = self.get_selected_device()
            self.detector = BassAudioDetector(device=device_id)
            if self.detector.start():
                self.running = True
                self.status_var.set("Running")
                self.update_display()
            else:
                self.status_var.set(f"Error: {self.detector.get_error()}")
    
    def stop_detection(self):
        """Stop audio detection"""
        if self.running and self.detector:
            self.detector.stop()
            self.running = False
            self.status_var.set("Stopped")
    
    def run(self):
        """Run the visualizer"""
        # Add start/stop button
        self.control_btn = ttk.Button(self.device_frame, 
                                    text="Start", 
                                    command=self.start_detection)
        self.control_btn.grid(row=0, column=2, padx=5)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start the main loop
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_detection()
        self.root.destroy()

if __name__ == "__main__":
    visualizer = BassVisualizer()
    visualizer.run()

import sys
import os
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QFileDialog, QTextEdit,
                             QMessageBox, QComboBox, QGroupBox, QProgressBar,
                             QRadioButton, QButtonGroup, QTabWidget)
from PyQt6.QtCore import Qt
import numpy as np


class LSBSteganography:
    @staticmethod
    def get_max_capacity(image_path: str, lsb_bits: int) -> int:
        """Calculate maximum capacity in bytes for given LSB bits"""
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            width, height = img.size
            total_pixels = width * height
            # Each pixel can store 3 * lsb_bits bits (R, G, B channels)
            max_bits = total_pixels * 3 * lsb_bits
            # Subtract bits for markers and null terminator
            return (max_bits // 8) - 1 - 22
        except Exception:
            return 0

    @staticmethod
    def encode(image_path: str, message: str, output_path: str, lsb_bits: int) -> bool:
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            encoded = img.copy()
            width, height = img.size
            
            # Add start and end markers
            message = f"zxy_start{message}zxy_end"
            
            # Add null terminator
            message += chr(0)
            
            # Convert message to binary with variable bit encoding
            bits = ''.join(f'{ord(c):08b}' for c in message)
            
            # Create mask for clearing LSB bits
            mask = (0xFF << lsb_bits) & 0xFF
            
            data_index = 0
            bit_length = len(bits)
            
            for y in range(height):
                for x in range(width):
                    if data_index >= bit_length:
                        encoded.save(output_path)
                        return True
                    
                    r, g, b = img.getpixel((x, y))
                    
                    # Process each color channel
                    channels = [r, g, b]
                    new_channels = []
                    
                    for channel_value in channels:
                        if data_index < bit_length:
                            # Extract bits to embed (up to lsb_bits)
                            bits_to_embed = bits[data_index:data_index + lsb_bits].ljust(lsb_bits, '0')
                            # Clear LSB bits and set new ones
                            new_value = (channel_value & mask) | int(bits_to_embed, 2)
                            new_channels.append(new_value)
                            data_index += lsb_bits
                        else:
                            new_channels.append(channel_value)
                    
                    encoded.putpixel((x, y), tuple(new_channels))
            
            encoded.save(output_path)
            return True
        except Exception as e:
            print(f"Encoding error: {e}")
            return False

    @staticmethod
    def decode(image_path: str, lsb_bits: int) -> str:
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            width, height = img.size
            
            # Create mask for extracting LSB bits
            mask = (1 << lsb_bits) - 1
            
            bits = ''
            for y in range(height):
                for x in range(width):
                    r, g, b = img.getpixel((x, y))
                    # Extract LSB bits from each channel
                    bits += format(r & mask, f'0{lsb_bits}b')
                    bits += format(g & mask, f'0{lsb_bits}b')
                    bits += format(b & mask, f'0{lsb_bits}b')
            
            # Convert bits to characters
            chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
            decoded_string = ''
            for c in chars:
                if len(c) == 8:  # Ensure we have complete byte
                    byte = int(c, 2)
                    if byte == 0:
                        break
                    decoded_string += chr(byte)

            # Find markers
            start_marker = "zxy_start"
            end_marker = "zxy_end"
            start_index = decoded_string.find(start_marker)
            end_index = decoded_string.find(end_marker)

            if start_index != -1 and end_index != -1:
                return decoded_string[start_index + len(start_marker):end_index]
            
            return ''
        except Exception as e:
            print(f"Decoding error: {e}")
            return ''

    @staticmethod
    def calculate_psnr(original_path: str, stego_path: str) -> float:
        """Calculate PSNR between original and stego images"""
        try:
            orig = np.array(Image.open(original_path))
            stego = np.array(Image.open(stego_path))
            
            mse = np.mean((orig - stego) ** 2)
            if mse == 0:
                return float('inf')
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return psnr
        except Exception:
            return 0.0

    @staticmethod
    def calculate_ssim(original_path: str, stego_path: str) -> float:
        """Calculate simplified SSIM approximation"""
        try:
            # Simplified SSIM calculation
            orig = np.array(Image.open(original_path)).astype(np.float64)
            stego = np.array(Image.open(stego_path)).astype(np.float64)
            
            # Basic similarity measure
            diff = np.abs(orig - stego)
            similarity = 1.0 - (np.mean(diff) / 255.0)
            return similarity
        except Exception:
            return 0.0


class CapacityConverter:
    @staticmethod
    def bytes_to_human_readable(bytes_value: int) -> str:
        """Convert bytes to human readable format (B, KB, MB, GB)"""
        if bytes_value < 1024:
            return f"{bytes_value} B"
        elif bytes_value < 1024**2:
            return f"{bytes_value/1024:.2f} KB"
        elif bytes_value < 1024**3:
            return f"{bytes_value/(1024**2):.2f} MB"
        else:
            return f"{bytes_value/(1024**3):.2f} GB"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced LSB Steganography")
        self.setGeometry(100, 100, 800, 700)

        # Create tab widget for different sections
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        self.create_steganography_tab()
        self.create_quality_analysis_tab()

        self.encode_image_path = None
        self.decode_image_path = None
        self.stego_image_path = None

    def create_steganography_tab(self):
        """Create the main steganography tab"""
        stego_widget = QWidget()
        vbox = QVBoxLayout(stego_widget)

        # LSB Options
        options_group = QGroupBox("LSB Options")
        options_layout = QHBoxLayout()
        
        self.lsb_combo = QComboBox()
        self.lsb_combo.addItems(["1 LSB", "2 LSB", "3 LSB", "4 LSB"])
        self.lsb_combo.setCurrentIndex(0)
        self.lsb_combo.currentTextChanged.connect(self.update_capacity)
        
        options_layout.addWidget(QLabel("LSB Bits:"))
        options_layout.addWidget(self.lsb_combo)
        options_group.setLayout(options_layout)
        vbox.addWidget(options_group)

        # Encode section
        encode_group = QGroupBox("Encode Message into Image")
        encode_layout = QVBoxLayout()
        
        hbox_encode = QHBoxLayout()
        self.encode_image_label = QLabel("No image selected")
        hbox_encode.addWidget(self.encode_image_label)
        btn_select_encode_image = QPushButton("Select Image")
        btn_select_encode_image.clicked.connect(self.select_encode_image)
        hbox_encode.addWidget(btn_select_encode_image)
        encode_layout.addLayout(hbox_encode)

        # Capacity information
        capacity_layout = QVBoxLayout()
        self.encode_capacity_label = QLabel("Maximum capacity: 0 bytes")
        self.message_info_label = QLabel("Message size: 0 bytes | Remaining: 0 bytes")
        self.capacity_percentage_label = QLabel("Usage: 0%")
        capacity_layout.addWidget(self.encode_capacity_label)
        capacity_layout.addWidget(self.message_info_label)
        capacity_layout.addWidget(self.capacity_percentage_label)
        encode_layout.addLayout(capacity_layout)

        self.encode_text = QTextEdit()
        self.encode_text.setPlaceholderText("Enter secret message here...")
        self.encode_text.setMaximumHeight(150)
        self.encode_text.textChanged.connect(self.update_message_info)  # Connect to text change
        encode_layout.addWidget(self.encode_text)

        btn_encode = QPushButton("Encode & Save")
        btn_encode.clicked.connect(self.encode_and_save)
        encode_layout.addWidget(btn_encode)
        
        encode_group.setLayout(encode_layout)
        vbox.addWidget(encode_group)

        # Decode section
        decode_group = QGroupBox("Decode Message from Image")
        decode_layout = QVBoxLayout()
        
        hbox_decode = QHBoxLayout()
        self.decode_image_label = QLabel("No image selected")
        hbox_decode.addWidget(self.decode_image_label)
        btn_select_decode_image = QPushButton("Select Image")
        btn_select_decode_image.clicked.connect(self.select_decode_image)
        hbox_decode.addWidget(btn_select_decode_image)
        decode_layout.addLayout(hbox_decode)

        self.decode_text = QTextEdit()
        self.decode_text.setReadOnly(True)
        self.decode_text.setMaximumHeight(150)
        decode_layout.addWidget(self.decode_text)

        btn_decode = QPushButton("Decode Message")
        btn_decode.clicked.connect(self.decode_message)
        decode_layout.addWidget(btn_decode)
        
        decode_group.setLayout(decode_layout)
        vbox.addWidget(decode_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        vbox.addWidget(self.progress_bar)

        self.tab_widget.addTab(stego_widget, "Steganography")

    def create_quality_analysis_tab(self):
        """Create the quality analysis tab"""
        quality_widget = QWidget()
        vbox = QVBoxLayout(quality_widget)

        # Image selection for quality analysis
        selection_group = QGroupBox("Select Images for Quality Analysis")
        selection_layout = QVBoxLayout()
        
        # Original image selection
        original_layout = QHBoxLayout()
        self.original_image_label = QLabel("No original image selected")
        original_layout.addWidget(self.original_image_label)
        btn_select_original = QPushButton("Select Original Image")
        btn_select_original.clicked.connect(self.select_original_image)
        original_layout.addWidget(btn_select_original)
        selection_layout.addLayout(original_layout)
        
        # Stego image selection
        stego_layout = QHBoxLayout()
        self.stego_image_label = QLabel("No stego image selected")
        stego_layout.addWidget(self.stego_image_label)
        btn_select_stego = QPushButton("Select Stego Image")
        btn_select_stego.clicked.connect(self.select_stego_image)
        stego_layout.addWidget(btn_select_stego)
        selection_layout.addLayout(stego_layout)
        
        selection_group.setLayout(selection_layout)
        vbox.addWidget(selection_group)

        # Analysis controls
        control_layout = QHBoxLayout()
        self.analyze_quality_btn = QPushButton("Analyze Image Quality")
        self.analyze_quality_btn.clicked.connect(self.analyze_quality_separate)
        self.analyze_quality_btn.setEnabled(False)
        control_layout.addWidget(self.analyze_quality_btn)
        
        self.clear_analysis_btn = QPushButton("Clear Results")
        self.clear_analysis_btn.clicked.connect(self.clear_quality_results)
        control_layout.addWidget(self.clear_analysis_btn)
        vbox.addLayout(control_layout)

        # Results display
        self.quality_results = QTextEdit()
        self.quality_results.setReadOnly(True)
        self.quality_results.setPlaceholderText("Quality analysis results will appear here...")
        vbox.addWidget(self.quality_results)

        # Progress bar for quality analysis
        self.quality_progress_bar = QProgressBar()
        self.quality_progress_bar.setVisible(False)
        vbox.addWidget(self.quality_progress_bar)

        self.tab_widget.addTab(quality_widget, "Quality Analysis")

        # Store paths for quality analysis
        self.original_analysis_path = None
        self.stego_analysis_path = None

    def get_selected_lsb_bits(self) -> int:
        """Get the number of LSB bits selected by user"""
        text = self.lsb_combo.currentText()
        return int(text.split()[0])

    def update_capacity(self):
        """Update the capacity display when image or LSB option changes"""
        if self.encode_image_path:
            lsb_bits = self.get_selected_lsb_bits()
            capacity = LSBSteganography.get_max_capacity(self.encode_image_path, lsb_bits)
            capacity_display = CapacityConverter.bytes_to_human_readable(capacity)
            self.encode_capacity_label.setText(f"Maximum capacity: {capacity_display}")
            self.update_message_info()  # Update message info when capacity changes

    def update_message_info(self):
        """Update message size and remaining capacity information"""
        if self.encode_image_path:
            # Get current message size
            message = self.encode_text.toPlainText()
            message_bytes = len(message.encode('utf-8'))
            
            # Get maximum capacity
            lsb_bits = self.get_selected_lsb_bits()
            max_capacity = LSBSteganography.get_max_capacity(self.encode_image_path, lsb_bits)
            
            # Calculate remaining capacity
            remaining = max_capacity - message_bytes
            
            # Update message info label
            message_display = CapacityConverter.bytes_to_human_readable(message_bytes)
            remaining_display = CapacityConverter.bytes_to_human_readable(remaining)
            self.message_info_label.setText(f"Message size: {message_display} | Remaining: {remaining_display}")
            
            # Update percentage usage
            if max_capacity > 0:
                percentage = (message_bytes / max_capacity) * 100
                self.capacity_percentage_label.setText(f"Usage: {percentage:.1f}%")
                
                # Change color based on usage
                if percentage > 90:
                    self.capacity_percentage_label.setStyleSheet("color: red; font-weight: bold;")
                elif percentage > 75:
                    self.capacity_percentage_label.setStyleSheet("color: orange; font-weight: bold;")
                else:
                    self.capacity_percentage_label.setStyleSheet("color: white;")
            else:
                self.capacity_percentage_label.setText("Usage: 0%")
                self.capacity_percentage_label.setStyleSheet("color: black;")
        else:
            self.message_info_label.setText("Message size: 0 bytes | Remaining: 0 bytes")
            self.capacity_percentage_label.setText("Usage: 0%")
            self.capacity_percentage_label.setStyleSheet("color: black;")

    def select_encode_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.bmp)")
        if path:
            self.encode_image_path = path
            self.encode_image_label.setText(os.path.basename(path))
            self.update_capacity()

    def encode_and_save(self):
        if not self.encode_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
        message = self.encode_text.toPlainText()
        if not message:
            QMessageBox.warning(self, "Warning", "Message is empty.")
            return
            
        lsb_bits = self.get_selected_lsb_bits()
        capacity = LSBSteganography.get_max_capacity(self.encode_image_path, lsb_bits)
        
        # Check if message fits
        message_bytes = len(message.encode('utf-8'))
        if message_bytes > capacity:
            QMessageBox.warning(self, "Warning", 
                              f"Message too large! Message: {message_bytes} bytes, Capacity: {capacity} bytes")
            return
            
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Encoded Image", "", "PNG (*.png)")
        if output_path:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            
            success = LSBSteganography.encode(self.encode_image_path, message, output_path, lsb_bits)
            
            self.progress_bar.setVisible(False)
            
            if success:
                self.stego_image_path = output_path
                QMessageBox.information(self, "Success", 
                                      f"Message encoded successfully using {lsb_bits} LSB bits.\n"
                                      f"Saved as: {os.path.basename(output_path)}")
            else:
                QMessageBox.critical(self, "Error", "Failed to encode message.")

    def select_decode_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.bmp)")
        if path:
            self.decode_image_path = path
            self.decode_image_label.setText(os.path.basename(path))

    def decode_message(self):
        if not self.decode_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
        lsb_bits = self.get_selected_lsb_bits()
        message = LSBSteganography.decode(self.decode_image_path, lsb_bits)
        if message:
            self.decode_text.setPlainText(message)
        else:
            QMessageBox.critical(self, "Error", "Failed to decode message or no message found.")

    # Quality Analysis Methods
    def select_original_image(self):
        """Select original image for quality analysis"""
        path, _ = QFileDialog.getOpenFileName(self, "Select Original Image", "", "Images (*.png *.bmp *.jpg *.jpeg)")
        if path:
            self.original_analysis_path = path
            self.original_image_label.setText(os.path.basename(path))
            self.check_analysis_ready()

    def select_stego_image(self):
        """Select stego image for quality analysis"""
        path, _ = QFileDialog.getOpenFileName(self, "Select Stego Image", "", "Images (*.png *.bmp *.jpg *.jpeg)")
        if path:
            self.stego_analysis_path = path
            self.stego_image_label.setText(os.path.basename(path))
            self.check_analysis_ready()

    def check_analysis_ready(self):
        """Enable analysis button if both images are selected"""
        self.analyze_quality_btn.setEnabled(
            self.original_analysis_path is not None and 
            self.stego_analysis_path is not None
        )

    def analyze_quality_separate(self):
        """Analyze quality between two selected images"""
        if not self.original_analysis_path or not self.stego_analysis_path:
            QMessageBox.warning(self, "Warning", "Please select both original and stego images.")
            return
            
        try:
            self.quality_progress_bar.setVisible(True)
            self.quality_progress_bar.setRange(0, 0)
            
            # Calculate quality metrics
            psnr = LSBSteganography.calculate_psnr(self.original_analysis_path, self.stego_analysis_path)
            ssim = LSBSteganography.calculate_ssim(self.original_analysis_path, self.stego_analysis_path)
            
            self.quality_progress_bar.setVisible(False)
            
            # Display results
            results = f"Image Quality Analysis Results:\n"
            results += f"{'='*50}\n"
            results += f"Original Image: {os.path.basename(self.original_analysis_path)}\n"
            results += f"Stego Image: {os.path.basename(self.stego_analysis_path)}\n"
            results += f"{'='*50}\n\n"
            
            results += f"PSNR (Peak Signal-to-Noise Ratio): {psnr:.2f} dB\n"
            results += f"SSIM (Structural Similarity Index): {ssim:.4f}\n\n"
            
            # Quality assessment
            results += "Quality Assessment:\n"
            if psnr == float('inf'):
                results += "• PSNR: Images are identical\n"
            elif psnr > 40:
                results += "• PSNR: Excellent quality (imperceptible changes)\n"
            elif psnr > 30:
                results += "• PSNR: Good quality (minor changes)\n"
            elif psnr > 20:
                results += "• PSNR: Fair quality (noticeable changes)\n"
            else:
                results += "• PSNR: Poor quality (significant changes)\n"
            
            if ssim > 0.95:
                results += "• SSIM: Very high similarity\n"
            elif ssim > 0.90:
                results += "• SSIM: High similarity\n"
            elif ssim > 0.80:
                results += "• SSIM: Moderate similarity\n"
            else:
                results += "• SSIM: Low similarity\n"
            
            results += f"\nDetailed Metrics:\n"
            results += f"- Higher PSNR values indicate better quality (ideal > 30 dB)\n"
            results += f"- SSIM ranges from 0-1, where 1 means identical images\n"
            
            self.quality_results.setPlainText(results)
            
        except Exception as e:
            self.quality_progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to analyze quality: {str(e)}")

    def clear_quality_results(self):
        """Clear quality analysis results"""
        self.quality_results.clear()
        self.original_analysis_path = None
        self.stego_analysis_path = None
        self.original_image_label.setText("No original image selected")
        self.stego_image_label.setText("No stego image selected")
        self.analyze_quality_btn.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

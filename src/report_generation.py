import os, time, tempfile
from PIL import Image, ImageDraw
from fpdf import FPDF
import platform
import subprocess
import matplotlib.pyplot as plt
from config import config
# Add this at the top of both files
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive

# Missing Person Detection PDF Report
def export_to_pdf(detections, pdf_filename="Output/detections.pdf", ref_filenames=None):
    """
    Export detection detections to a PDF report with improved formatting.
    Each detection includes the video filename, detection time, similarity score,
    and an image preview with bounding box and dominant color.
    """
    # Create PDF with appropriate settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Set up metadata
    pdf.set_title("Missing Person Detection Report")
    pdf.set_author("Missing Person Detection System")

    # Add cover page
    pdf.add_page()
    pdf.set_font("Arial", "B", size=27)
    pdf.cell(0, 80, "Missing Person Detection", 0, 1, 'C')
    pdf.cell(0, 20, "Report", 0, 1, 'C')

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, f"Total Detections: {len(detections)}", 0, 1, 'C')

    # Add timestamp
    pdf.set_font("Arial", "I", size=10)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated: {current_time}", 0, 1, 'C')

    # Add reference images section
    if ref_filenames:
        pdf.add_page()
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(0, 10, "Reference Images Used", 0, 1, 'C')

        # Add description
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, "The following reference images were used to identify the missing person:", 0, 1, 'L')

        # Calculate layout for reference images
        images_per_row = 2
        margin = 10
        image_width = (pdf.w - 2 * margin) / images_per_row - 10
        image_height = image_width * 0.75  # Aspect ratio

        # Add each reference image
        x_pos = margin
        y_pos = pdf.get_y() + 5

        for i, ref_filename in enumerate(ref_filenames):
            # Position images in a grid
            if i > 0 and i % images_per_row == 0:
                x_pos = margin
                y_pos += image_height + 20

            # Add the image
            pdf.image(ref_filename, x=x_pos, y=y_pos, w=image_width, h=image_height)

            # Add filename caption
            pdf.set_xy(x_pos, y_pos + image_height + 2)
            pdf.set_font("Arial", "I", size=8)
            display_name = os.path.basename(ref_filename)
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."
            pdf.cell(image_width, 10, display_name, 0, 1, 'C')

            x_pos += image_width + 10

    # Add detection summary table
    if detections:
        pdf.add_page()
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(0, 10, "Detections Summary", 0, 1, 'C')

        # Create header row
        pdf.set_font("Arial", "B", size=10)
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(70, 8, "Video", 1, 0, 'C', True)
        pdf.cell(30, 8, "Time (s)", 1, 0, 'C', True)
        pdf.cell(30, 8, "Similarity", 1, 0, 'C', True)
        pdf.cell(60, 8, "Dominant Color", 1, 1, 'C', True)

        # Fill data rows
        pdf.set_font("Arial", size=9)
        for det in detections:
            # Truncate filename if too long
            filename = det['video_filename']
            if len(filename) > 30:
                filename = filename[:27] + "..."

            pdf.cell(70, 8, filename, 1, 0, 'L')
            pdf.cell(30, 8, f"{det['time']:.2f}", 1, 0, 'C')

            # Color the similarity cell based on confidence level
            similarity = det['similarity']
            if similarity > 0.8:
                pdf.set_fill_color(150, 255, 150)  # Green for high confidence
            elif similarity > 0.7:
                pdf.set_fill_color(255, 255, 150)  # Yellow for medium confidence
            else:
                pdf.set_fill_color(255, 200, 200)  # Light red for lower confidence

            pdf.cell(30, 8, f"{similarity:.2f}", 1, 0, 'C', True)

            # Reset fill color
            pdf.set_fill_color(255, 255, 255)

            # Get color values
            r, g, b = det['dominant_color']

            # Add color box and text
            pdf.cell(40, 8, f"RGB: {r},{g},{b}", 1, 0, 'L')
            pdf.set_fill_color(r, g, b)
            pdf.cell(20, 8, "", 1, 1, 'C', True)

            # Reset fill color
            pdf.set_fill_color(255, 255, 255)

    # Add confidence visualization
    pdf.add_page()
    pdf.set_font("Arial", "B", size=14)
    pdf.cell(0, 10, "Confidence Analysis", 0, 1, 'C')
    add_confidence_visualization(pdf, detections, 'face')

    # Add detailed detections
    images_per_page = 4  # Reduced from 6 to allow more space
    current_image = 0

    for idx, det in enumerate(detections):
        # Start a new page for first image or when page is full
        if current_image == 0:
            pdf.add_page()
            pdf.set_font("Arial", "B", size=14)
            pdf.cell(0, 10, f"Detection Details", 0, 1, 'C')
            current_image = 0

        # Calculate position with more spacing
        row = current_image // 2
        col = current_image % 2

        # Base positions with more space between items
        x_start = 10 + col * 95
        y_start = 30 + row * 120

        # Create a box around the entire detection
        pdf.set_draw_color(100, 100, 100)
        pdf.rect(x_start, y_start, 90, 110)

        # Process and add the image
        img = Image.fromarray(det['frame_img'])
        draw = ImageDraw.Draw(img)
        draw.rectangle(det['box'], outline="red", width=3)
        img_resized = img.resize((400, 300))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_filename = tmp.name
            img_resized.save(tmp_filename)

        # Add the image with proper positioning
        pdf.image(tmp_filename, x_start + 5, y_start + 5, w=80, h=60)
        os.remove(tmp_filename)

        # Add detection information below the image
        y_text = y_start + 70
        pdf.set_xy(x_start + 5, y_text)

        # Add video filename and timestamp with proper styling
        pdf.set_font("Arial", "B", size=10)
        pdf.cell(30, 5, "Video:", 0, 0, 'L')
        pdf.set_font("Arial", size=10)
        pdf.cell(50, 5, f"{det['video_filename']}", 0, 2, 'R')

        y_text = y_start + 70 + 10
        pdf.set_xy(x_start + 5, y_text)

        pdf.set_font("Arial", "B", size=9)
        pdf.cell(20, 10, "Timestamp:", 0, 0, 'L')
        pdf.set_font("Arial", size=9)
        pdf.cell(30, 10, f"{det['time']:.2f}s", 0, 0, 'L')

        pdf.set_font("Arial", "B", size=9)
        pdf.cell(20, 10, "Similarity:", 0, 0, 'L')
        pdf.set_font("Arial", size=9)
        pdf.cell(10, 10, f"{det['similarity']:.2f}", 0, 1, 'R')

        y_text = y_start + 70 + 20
        pdf.set_xy(x_start + 5, y_text)

        # Add dominant color information
        pdf.set_font("Arial", "B", size=9)
        pdf.cell(50, 10, "Clothing Color:", 0, 0, 'L')

        # Add color swatch
        r, g, b = det['dominant_color']
        pdf.set_fill_color(r, g, b)
        pdf.rect(x_start + 35, y_text + 2.5 , 50, 5, style='F')

        # Increment counter for positioning
        current_image += 1

        # Reset after filling a page
        if current_image >= images_per_page:
            current_image = 0

    # Add information footer
    pdf.set_y(-25)
    pdf.set_font("Arial", "I", size=8)
    pdf.cell(0, 10, "Missing Person Detection System - Confidential Report", 0, 0, 'C')
    pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, 'R')

    # Save the PDF
    pdf.output(pdf_filename)
    print(f"PDF saved as {pdf_filename}")
    # Open the PDF with the default PDF viewer
    try:
        import platform
        import subprocess
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', pdf_filename))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(pdf_filename)
        else:                                   # Linux
            subprocess.call(('xdg-open', pdf_filename))
    except:
        print("Could not open PDF automatically. Please open it manually.")

# Violence Detection PDF Report  
def export_violence_report(detections, video_filename, pdf_filename="Output/violence_detections.pdf"):
    """Create a PDF report for violence detections with improved formatting"""
    if not detections:
        print(f"No violence detected in {video_filename}")
        return

    # Create PDF with larger margins to avoid overlap
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Set up metadata
    pdf.set_title("Violence Detection Report")
    pdf.set_author("Violence Detection System")

    # Add cover page
    pdf.add_page()
    pdf.set_font("Arial", "B", size=27)
    pdf.cell(0, 80, "Violence Detection", 0, 1, 'C')
    pdf.cell(0, 20, "Report", 0, 1, 'C')

    # Add video information
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, f"Video File: {os.path.basename(video_filename)}", 0, 1, 'C')

    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, f"Total Detections: {len(detections)}", 0, 1, 'C')

    # Add timestamp
    pdf.set_font("Arial", "I", size=10)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated: {current_time}", 0, 1, 'C')


    pdf.ln(10)

    # Add confidence visualization
    pdf.add_page()
    pdf.set_font("Arial", "B", size=14)
    pdf.cell(0, 10, "Confidence Analysis", 0, 1, 'C')
    add_confidence_visualization(pdf, detections, 'violence')
    
    pdf.add_page()
    # Add each detection with proper spacing
    for idx, det in enumerate(detections):
        # Create a new page for each detection except the first one
        if idx > 0:
            pdf.add_page()

        # Detection header with background color
        pdf.set_fill_color(220, 220, 220)  # Light gray background
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, f"Detection #{idx+1}", 1, 1, 'L', fill=True)
        pdf.ln(5)

        # Save thumbnail to temp file
        thumbnail = Image.fromarray(det['thumbnail'])
        thumbnail_resized = thumbnail.resize((320, 240))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_filename = tmp.name
            thumbnail_resized.save(tmp_filename)

        # Add thumbnail to PDF - centered
        image_width = 120
        margin_left = (210 - image_width) / 2  # A4 width is 210mm
        pdf.image(tmp_filename, x=margin_left, y=pdf.get_y(), w=image_width)
        os.remove(tmp_filename)

        # Move cursor below the image
        pdf.ln(120)  # Adjust based on your image height

        # Add detection details in a table-like format
        pdf.set_font("Arial", "B", size=11)
        pdf.cell(40, 8, "Time:", 1, 0)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"{det['time']:.2f} seconds", 1, 1)

        pdf.set_font("Arial", "B", size=11)
        pdf.cell(40, 8, "Probability:", 1, 0)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"{det['probability']:.4f}", 1, 1)

        pdf.set_font("Arial", "B", size=11)
        pdf.cell(40, 8, "Frame Index:", 1, 0)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"{det['frame_idx']}", 1, 1)

        # Add severity indicator based on probability
        pdf.ln(10)
        pdf.set_font("Arial", "B", size=12)
        pdf.cell(0, 8, "Severity Level:", 0, 1)

        # Determine severity level based on probability
        if det['probability'] > 0.9:
            severity = "HIGH"
            r, g, b = 255, 0, 0  # Red
        elif det['probability'] > 0.8:
            severity = "MEDIUM"
            r, g, b = 255, 165, 0  # Orange
        else:
            severity = "LOW"
            r, g, b = 255, 255, 0  # Yellow

        pdf.set_fill_color(r, g, b)
        pdf.set_text_color(0 if severity == "LOW" else 255)
        pdf.cell(60, 10, f" {severity} ", 1, 1, 'C', fill=True)
        pdf.set_text_color(0)  # Reset text color to black

    # Add footer with timestamp
    pdf.set_y(-30)
    pdf.set_font("Arial", "I", size=8)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Report generated on {current_time}", 0, 0, 'C')

    pdf.output(pdf_filename)
    print(f"Violence detection report saved as {pdf_filename}")
    
    # Open the PDF with the default PDF viewer
    try:
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', pdf_filename))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(pdf_filename)
        else:                                   # Linux
            subprocess.call(('xdg-open', pdf_filename))
    except:
        print("Could not open PDF automatically. Please open it manually.")

def generate_combined_report(missing_detections, violence_detections, ref_filenames=None, 
                                     output_dir="Output"):
    """
    Generate a comprehensive report combining missing person and violence detections
    with enhanced formatting, visualizations, and performance statistics.
    
    Args:
        missing_detections: List of missing person detections
        violence_detections: List of violence detections
        ref_filenames: List of reference image filenames used for missing person detection
        output_dir: Directory to save the report
        performance_stats: Dictionary containing performance metrics (optional)
    
    Returns:
        Path to the generated report PDF
    """
    import os, time, tempfile
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw
    from collections import Counter
    import matplotlib
    matplotlib.use('Agg')  # Set the backend to non-interactive
    
    # Get actual performance stats instead of sample data
    from stats import calculate_performance_stats
    performance_stats = calculate_performance_stats()

    os.makedirs(output_dir, exist_ok=True)
    
    # Create PDF with appropriate settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Set up metadata
    pdf.set_title("CCTV Analysis System - Complete Report")
    pdf.set_author("CCTV Analysis System")
    
    # Add cover page
    pdf.add_page()
    pdf.set_font("Arial", "B", size=27)
    pdf.cell(0, 60, "CCTV Analysis System", 0, 1, 'C')
    pdf.cell(0, 20, "Comprehensive Report", 0, 1, 'C')
    
    # Add summary statistics
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, f"Missing Person Detections: {len(missing_detections)}", 0, 1, 'C')
    pdf.cell(0, 10, f"Violence Incidents: {len(violence_detections)}", 0, 1, 'C')
    
    # Add timestamp
    pdf.set_font("Arial", "I", size=10)
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 10, f"Generated: {current_time}", 0, 1, 'C')
    
    # Executive summary section
    pdf.add_page()
    pdf.set_font("Arial", "B", size=18)
    pdf.cell(0, 15, "Executive Summary", 0, 1, 'L')
    
    # Calculate unique videos
    missing_videos = set([det.get('video_filename', det.get('video_path', '')) for det in missing_detections])
    violence_videos = set([det.get('video_filename', det.get('video_path', '')) for det in violence_detections])
    all_videos = missing_videos.union(violence_videos)
    
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, 
        f"This report summarizes the analysis of {len(all_videos)} video files processed by the CCTV Analysis System. " +
        f"The system detected {len(missing_detections)} potential missing person instances across {len(missing_videos)} videos and " +
        f"{len(violence_detections)} potential violent incidents across {len(violence_videos)} videos.", 0, 'L')
    
    # Add detection statistics visualization
    if missing_detections or violence_detections:
        pdf.ln(5)
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, "Detection Statistics", 0, 1, 'L')
        
        # Create a chart for detection types
        fig, ax = plt.subplots(figsize=(10, 4))
        labels = ['Missing Person Detections', 'Violence Detections']
        values = [len(missing_detections), len(violence_detections)]
        colors = ['#3498db', '#e74c3c']  # Blue for missing, red for violence
        ax.bar(labels, values, color=colors)
        plt.title('Detection Summary')
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        # Save chart to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            chart_path = tmp.name
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Add chart to PDF
        pdf.image(chart_path, x=10, y=pdf.get_y(), w=180)
        pdf.ln(100)  # Space for the chart
        
        # Clean up
        os.remove(chart_path)
    
    # Add reference images section if available
    if ref_filenames:
        pdf.add_page()
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(0, 10, "Reference Images Used", 0, 1, 'C')

        # Add description
        pdf.set_font("Arial", size=10)
        pdf.cell(0, 10, "The following reference images were used to identify the missing person:", 0, 1, 'L')

        # Calculate layout for reference images
        images_per_row = 2
        margin = 10
        image_width = (pdf.w - 2 * margin) / images_per_row - 10
        image_height = image_width * 0.75  # Aspect ratio

        # Add each reference image
        x_pos = margin
        y_pos = pdf.get_y() + 5

        for i, ref_filename in enumerate(ref_filenames):
            # Position images in a grid
            if i > 0 and i % images_per_row == 0:
                x_pos = margin
                y_pos += image_height + 20

            # Add the image
            pdf.image(ref_filename, x=x_pos, y=y_pos, w=image_width, h=image_height)

            # Add filename caption
            pdf.set_xy(x_pos, y_pos + image_height + 2)
            pdf.set_font("Arial", "I", size=8)
            display_name = os.path.basename(ref_filename)
            if len(display_name) > 25:
                display_name = display_name[:22] + "..."
            pdf.cell(image_width, 10, display_name, 0, 1, 'C')

            x_pos += image_width + 10
    
    # Missing person detections section
    if missing_detections:
        pdf.add_page()
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(0, 10, "Missing Person Detections", 0, 1, 'L')
        
        # Add confidence visualization
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, "Confidence Analysis", 0, 1, 'C')
        add_confidence_visualization(pdf, missing_detections, 'face')
        
        # Summary table
        pdf.ln(10)
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, "Detections Summary", 0, 1, 'C')

        # Create header row
        pdf.set_font("Arial", "B", size=10)
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(70, 8, "Video", 1, 0, 'C', True)
        pdf.cell(30, 8, "Time (s)", 1, 0, 'C', True)
        pdf.cell(30, 8, "Similarity", 1, 0, 'C', True)
        pdf.cell(60, 8, "Dominant Color", 1, 1, 'C', True)

        # Fill data rows (show top 10 most confident detections)
        pdf.set_font("Arial", size=9)
        top_detections = sorted(missing_detections, key=lambda x: x.get('similarity', 0), reverse=True)[:10]
        
        for det in top_detections:
            # Truncate filename if too long
            filename = det.get('video_filename', os.path.basename(det.get('video_path', 'Unknown')))
            if len(filename) > 30:
                filename = filename[:27] + "..."

            pdf.cell(70, 8, filename, 1, 0, 'L')
            pdf.cell(30, 8, f"{det.get('time', 0):.2f}", 1, 0, 'C')

            # Color the similarity cell based on confidence level
            similarity = det.get('similarity', 0)
            if similarity > 0.8:
                pdf.set_fill_color(150, 255, 150)  # Green for high confidence
            elif similarity > 0.7:
                pdf.set_fill_color(255, 255, 150)  # Yellow for medium confidence
            else:
                pdf.set_fill_color(255, 200, 200)  # Light red for lower confidence

            pdf.cell(30, 8, f"{similarity:.2f}", 1, 0, 'C', True)

            # Reset fill color
            pdf.set_fill_color(255, 255, 255)

            # Get color values
            if 'dominant_color' in det:
                r, g, b = det['dominant_color']
                color_text = f"RGB: {r},{g},{b}"
            else:
                r, g, b = 200, 200, 200
                color_text = "N/A"

            # Add color box and text
            pdf.cell(40, 8, color_text, 1, 0, 'L')
            pdf.set_fill_color(r, g, b)
            pdf.cell(20, 8, "", 1, 1, 'C', True)
            pdf.set_fill_color(255, 255, 255)
        
        # Add detailed examples with images - FIXED SECTION
        pdf.add_page()
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, "High Confidence Examples", 0, 1, 'C')
        
        high_conf_detections = [det for det in missing_detections if det.get('similarity', 0) > 0.7]
        if high_conf_detections:
            max_examples = min(6, len(high_conf_detections))
            
            # Show 2 examples per page to avoid incomplete layouts
            examples_per_page = 2
            
            for i in range(max_examples):
                # Add new page for every 2 examples
                if i % examples_per_page == 0 and i > 0:
                    pdf.add_page()
                
                det = high_conf_detections[i]
                
                # Calculate vertical position based on example position on the page
                example_height = 100  # Fixed height for each example
                y_position = pdf.get_y()
                
                # Create a bordered box for the entire detection
                pdf.set_draw_color(100, 100, 100)
                pdf.rect(10, y_position, 190, example_height)
                
                # Process and add the image if available
                if 'frame_img' in det:
                    img = Image.fromarray(det['frame_img'])
                    
                    # Add bounding box if available
                    if 'box' in det:
                        draw = ImageDraw.Draw(img)
                        draw.rectangle(det['box'], outline="red", width=3)
                    
                    # Fixed aspect ratio and size
                    img_width = 80
                    img_height = 80
                    img_resized = img.resize((int(img_width * 3), int(img_height * 3))) if hasattr(img, 'resize') else img

                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_filename = tmp.name
                        img_resized.save(tmp_filename)

                    # Add the image with consistent positioning
                    pdf.image(tmp_filename, x=15, y=y_position + 10, w=img_width, h=img_height)
                    os.remove(tmp_filename)
                
                # Add detection information in a structured table format
                info_x = 110  # Consistent X position for text
                
                # Video filename with consistent positioning
                pdf.set_xy(info_x, y_position + 15)
                pdf.set_font("Arial", "B", size=10)
                pdf.cell(30, 8, "Video:", 0, 0, 'L')
                pdf.set_font("Arial", size=10)
                video_name = det.get('video_filename', os.path.basename(det.get('video_path', 'Unknown')))
                if len(video_name) > 20:
                    video_name = video_name[:17] + "..."
                pdf.cell(60, 8, video_name, 0, 1, 'L')
                
                # Timestamp with consistent positioning
                pdf.set_xy(info_x, y_position + 35)
                pdf.set_font("Arial", "B", size=10)
                pdf.cell(30, 8, "Time:", 0, 0, 'L')
                pdf.set_font("Arial", size=10)
                pdf.cell(60, 8, f"{det.get('time', 0):.2f}s", 0, 1, 'L')
                
                # Similarity score with consistent positioning
                pdf.set_xy(info_x, y_position + 55)
                pdf.set_font("Arial", "B", size=10)
                pdf.cell(30, 8, "Similarity:", 0, 0, 'L')
                pdf.set_font("Arial", size=10)
                pdf.cell(60, 8, f"{det.get('similarity', 0):.2f}", 0, 1, 'L')
                
                # Dominant color with consistent positioning
                if 'dominant_color' in det:
                    pdf.set_xy(info_x, y_position + 75)
                    pdf.set_font("Arial", "B", size=10)
                    pdf.cell(30, 8, "Color:", 0, 0, 'L')
                    r, g, b = det['dominant_color']
                    pdf.set_fill_color(r, g, b)
                    pdf.rect(info_x + 30, y_position + 77, 20, 5, style='F')
                    pdf.set_fill_color(255, 255, 255)
                
                # Adjust cursor position for next example with consistent spacing
                pdf.set_y(y_position + example_height + 10)  # 10px gap between examples
        else:
            pdf.multi_cell(0, 10, "No high confidence detections found.", 0, 'L')
    
    # Violence detections section
    if violence_detections:
        pdf.add_page()
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(0, 10, "Violence Detections", 0, 1, 'L')
        
        # Add confidence visualization
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, "Confidence Analysis", 0, 1, 'C')
        add_confidence_visualization(pdf, violence_detections, 'violence')
        
        # Summary information
        pdf.ln(10)
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, "Violence Detection Summary", 0, 1, 'C')
        
        # Create header row for summary table
        pdf.set_font("Arial", "B", size=10)
        pdf.set_fill_color(200, 200, 200)
        pdf.cell(70, 8, "Video", 1, 0, 'C', True)
        pdf.cell(30, 8, "Time (s)", 1, 0, 'C', True)
        pdf.cell(30, 8, "Probability", 1, 0, 'C', True)
        pdf.cell(60, 8, "Severity", 1, 1, 'C', True)
        
        # Fill data rows (show top 10 most probable violence detections)
        pdf.set_font("Arial", size=9)
        top_violence = sorted(violence_detections, key=lambda x: x.get('probability', 0), reverse=True)[:10]
        
        for det in top_violence:
            # Truncate filename if too long
            filename = det.get('video_filename', os.path.basename(det.get('video_path', 'Unknown')))
            if len(filename) > 30:
                filename = filename[:27] + "..."

            pdf.cell(70, 8, filename, 1, 0, 'L')
            pdf.cell(30, 8, f"{det.get('time', 0):.2f}", 1, 0, 'C')

            # Color the probability cell based on level
            probability = det.get('probability', 0)
            if probability > 0.9:
                pdf.set_fill_color(255, 0, 0)  # Red for high probability
                severity = "HIGH"
            elif probability > 0.8:
                pdf.set_fill_color(255, 165, 0)  # Orange for medium
                severity = "MEDIUM"
            else:
                pdf.set_fill_color(255, 255, 0)  # Yellow for lower
                severity = "LOW"

            pdf.cell(30, 8, f"{probability:.2f}", 1, 0, 'C', True)
            pdf.cell(60, 8, severity, 1, 1, 'C', True)
            pdf.set_fill_color(255, 255, 255)  # Reset fill color
        
        # Add detailed examples with images
        pdf.add_page()
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, "High Probability Violence Examples", 0, 1, 'C')
        
        high_prob_violence = [det for det in violence_detections if det.get('probability', 0) > 0.7]
        if high_prob_violence:
            max_examples = min(6, len(high_prob_violence))
            
            for i in range(max_examples):
                if i > 0:
                    pdf.add_page()
                
                det = high_prob_violence[i]
                
                # Detection header with background color
                pdf.set_fill_color(220, 220, 220)  # Light gray background
                pdf.set_font("Arial", "B", size=14)
                pdf.cell(0, 10, f"Violence Detection #{i+1}", 1, 1, 'L', fill=True)
                pdf.ln(5)
                
                # Add thumbnail if available
                if 'thumbnail' in det:
                    # Save thumbnail to temp file
                    thumbnail = Image.fromarray(det['thumbnail'])
                    thumbnail_resized = thumbnail.resize((320, 240)) if hasattr(thumbnail, 'resize') else thumbnail

                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_filename = tmp.name
                        thumbnail_resized.save(tmp_filename)

                    # Add thumbnail to PDF - centered
                    image_width = 120
                    margin_left = (210 - image_width) / 2  # A4 width is 210mm
                    pdf.image(tmp_filename, x=margin_left, y=pdf.get_y(), w=image_width)
                    os.remove(tmp_filename)
                    
                    # Move cursor below the image
                    pdf.ln(90)
                
                # Add detection details in a table-like format
                pdf.set_font("Arial", "B", size=11)
                pdf.cell(40, 8, "Video:", 1, 0)
                pdf.set_font("Arial", size=11)
                pdf.cell(0, 8, f"{det.get('video_filename', os.path.basename(det.get('video_path', 'Unknown')))}", 1, 1)
                
                pdf.set_font("Arial", "B", size=11)
                pdf.cell(40, 8, "Time:", 1, 0)
                pdf.set_font("Arial", size=11)
                pdf.cell(0, 8, f"{det.get('time', 0):.2f} seconds", 1, 1)

                pdf.set_font("Arial", "B", size=11)
                pdf.cell(40, 8, "Probability:", 1, 0)
                pdf.set_font("Arial", size=11)
                pdf.cell(0, 8, f"{det.get('probability', 0):.4f}", 1, 1)
                
                if 'frame_idx' in det:
                    pdf.set_font("Arial", "B", size=11)
                    pdf.cell(40, 8, "Frame Index:", 1, 0)
                    pdf.set_font("Arial", size=11)
                    pdf.cell(0, 8, f"{det['frame_idx']}", 1, 1)
                
                # Add severity indicator based on probability
                pdf.ln(10)
                pdf.set_font("Arial", "B", size=12)
                pdf.cell(0, 8, "Severity Level:", 0, 1)

                # Determine severity level based on probability
                probability = det.get('probability', 0)
                if probability > 0.9:
                    severity = "HIGH"
                    r, g, b = 255, 0, 0  # Red
                elif probability > 0.8:
                    severity = "MEDIUM"
                    r, g, b = 255, 165, 0  # Orange
                else:
                    severity = "LOW"
                    r, g, b = 255, 255, 0  # Yellow

                pdf.set_fill_color(r, g, b)
                pdf.set_text_color(0 if severity == "LOW" else 255)
                pdf.cell(60, 10, f" {severity} ", 1, 1, 'C', fill=True)
                pdf.set_text_color(0)  # Reset text color to black
        else:
            pdf.multi_cell(0, 10, "No high probability violence detections found.", 0, 'L')
    
    # Performance statistics section
    if performance_stats:
        pdf.add_page()
        pdf.set_font("Arial", "B", size=16)
        pdf.cell(0, 10, "Performance Statistics", 0, 1, 'L')
        
        # Processing time visualization if available
        if 'processing_times' in performance_stats:
            processing_times = performance_stats['processing_times']
            
            # Create visualization for processing times
            fig, ax = plt.subplots(figsize=(10, 4))
            videos = list(processing_times.keys())
            times = list(processing_times.values())
            
            # Only show up to 10 videos for readability
            if len(videos) > 10:
                videos = videos[:10]
                times = times[:10]
                
            # Truncate long video names
            videos = [v[-20:] if len(v) > 20 else v for v in videos]
            
            ax.barh(videos, times, color='#3498db')
            plt.title('Processing Time by Video')
            plt.xlabel('Time (seconds)')
            plt.grid(axis='x', alpha=0.3)
            
            # Save chart to temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                chart_path = tmp.name
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Add chart to PDF
            pdf.ln(5)
            pdf.set_font("Arial", "B", size=12)
            pdf.cell(0, 10, "Processing Times:", 0, 1, 'L')
            pdf.image(chart_path, x=10, y=pdf.get_y(), w=180)
            pdf.ln(120)  # Space for the chart
            
            # Clean up
            os.remove(chart_path)
        
        # Add overall metrics
        pdf.set_font("Arial", "B", size=14)
        pdf.cell(0, 10, "Overall Metrics", 0, 1, 'L')
        
        pdf.set_font("Arial", "B", size=10)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(100, 8, "Metric", 1, 0, 'C', True)
        pdf.cell(80, 8, "Value", 1, 1, 'C', True)
        
        # Add each metric
        pdf.set_font("Arial", size=10)
        for key, value in performance_stats.items():
            if key != 'processing_times':  # Skip processing times as we visualized it
                pdf.cell(100, 8, key.replace('_', ' ').title(), 1, 0, 'L')
                
                # Format the value based on its type
                if isinstance(value, float):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                    
                pdf.cell(80, 8, formatted_value, 1, 1, 'C')
    
    # Add information footer
    pdf.set_y(-25)
    pdf.set_font("Arial", "I", size=8)
    pdf.cell(0, 10, "CCTV Analysis System - Confidential Report", 0, 0, 'C')
    pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, 'R')
    
    # Save the PDF
    report_filename = os.path.join(output_dir, f"cctv_analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf")
    pdf.output(report_filename)
    print(f"Enhanced comprehensive report saved as {report_filename}")
    
    # Open the PDF with the default PDF viewer
    try:
        import platform
        import subprocess
        if platform.system() == 'Darwin':       # macOS
            subprocess.call(('open', report_filename))
        elif platform.system() == 'Windows':    # Windows
            os.startfile(report_filename)
        else:                                   # Linux
            subprocess.call(('xdg-open', report_filename))
    except:
        print("Could not open PDF automatically. Please open it manually.")
    
    return report_filename

def add_confidence_visualization(pdf, detections, detection_type):
    """
    Create and add a visualization of detection confidence levels.
    
    Args:
        pdf: FPDF object
        detections: List of detection dictionaries
        detection_type: 'face' for missing persons or 'violence' for violence detections
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import tempfile
    import os
    from collections import Counter
    
    # Get confidence values
    confidence_key = 'similarity' if detection_type == 'face' else 'probability'
    confidence_values = [det.get(confidence_key, 0) for det in detections]
    
    # Create histogram of confidence values
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Define bin edges
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    
    # Create histogram
    n, bins, patches = ax.hist(confidence_values, bins=bins, edgecolor='black', alpha=0.7)
    
    # Customize histogram colors based on confidence level
    cm = plt.cm.get_cmap('RdYlGn')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Normalize bin centers to [0, 1] for colormap
    col = bin_centers / max(1, max(bin_centers))
    
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # Add labels and title
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Detections')
    if detection_type == 'face':
        plt.title('Distribution of Missing Person Detection Confidence')
    else:
        plt.title('Distribution of Violence Detection Confidence')
    
    plt.grid(alpha=0.3)
    
    # Add a summary of confidence levels as text
    high_conf = sum(1 for x in confidence_values if x > 0.8)
    medium_conf = sum(1 for x in confidence_values if 0.6 < x <= 0.8)
    low_conf = sum(1 for x in confidence_values if x <= 0.6)
    
    summary_text = f"High (>0.8): {high_conf}\nMedium (0.6-0.8): {medium_conf}\nLow (<0.6): {low_conf}"
    plt.figtext(0.75, 0.70, summary_text, horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8))
    
    # Save chart to temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        chart_path = tmp.name
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Add chart to PDF
    pdf.image(chart_path, x=10, y=pdf.get_y(), w=180)
    pdf.ln(90)  # Space for the chart
    
    # Clean up
    os.remove(chart_path)
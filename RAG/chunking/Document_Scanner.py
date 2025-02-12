### From https://medium.com/@abhishekkhaiwale007/building-an-intelligent-document-scanner-advanced-computer-vision-for-document-processing-a1cf8958ca78

import cv2
import numpy as np
import pytesseract
from scipy.spatial import distance as dist
import imutils
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_local
import time

# Required installations:
# pip install opencv-python numpy pytesseract scipy imutils pillow scikit-image matplotlib
# You'll also need to install Tesseract OCR on your system

class DocumentScanner:
    def __init__(self):
        # Initialize OCR engine
        self.tesseract_config = r'--oem 3 --psm 6'
        
        # Initialize image enhancement parameters
        self.kernel_size = (5, 5)
        self.sigma = 1.0
        
        # Performance tracking
        self.processing_times = []
        self.start_time = None
        
    def preprocess_image(self, image):
        """Prepare image for document detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.kernel_size, self.sigma)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 75, 200)
        
        return edges
        
    def find_document_contour(self, edges):
        """Detect the document boundaries"""
        # Find contours in the edge map
        contours = cv2.findContours(
            edges.copy(),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(contours)
        
        # Sort contours by area in descending order
        contours = sorted(
            contours,
            key=cv2.contourArea,
            reverse=True
        )[:5]
        
        # Initialize document contour
        document_contour = None
        
        # Loop over contours
        for contour in contours:
            # Approximate the contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(
                contour,
                0.02 * perimeter,
                True
            )
            
            # If we have found a contour with four points,
            # we can assume we have found the document
            if len(approx) == 4:
                document_contour = approx
                break
                
        return document_contour
        
    def order_points(self, points):
        """Order points in top-left, top-right, bottom-right, bottom-left order"""
        # Initialize coordinates
        rect = np.zeros((4, 2), dtype="float32")
        
        # Get points sum and difference
        pts_sum = points.sum(axis=1)
        pts_diff = np.diff(points, axis=1)
        
        # Top-left point has smallest sum
        rect[0] = points[np.argmin(pts_sum)]
        # Bottom-right point has largest sum
        rect[2] = points[np.argmax(pts_sum)]
        # Top-right point has smallest difference
        rect[1] = points[np.argmin(pts_diff)]
        # Bottom-left point has largest difference
        rect[3] = points[np.argmax(pts_diff)]
        
        return rect
        
    def perspective_transform(self, image, points):
        """Apply perspective transform to obtain top-down view"""
        # Order points in standard order
        rect = self.order_points(
            points.reshape(4, 2).astype("float32")
        )
        (tl, tr, br, bl) = rect
        
        # Calculate width of new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(widthA), int(widthB))
        
        # Calculate height of new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(heightA), int(heightB))
        
        # Construct destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply perspective transform
        warped = cv2.warpPerspective(
            image,
            transform_matrix,
            (max_width, max_height)
        )
        
        return warped
        
    def enhance_document(self, image):
        """Enhance document image for better readability"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply adaptive thresholding
        threshold = threshold_local(gray, 11, offset=10)
        binary = (gray > threshold).astype("uint8") * 255
        
        # Apply unsharp masking for edge enhancement
        blurred = cv2.GaussianBlur(binary, (0, 0), 3)
        enhanced = cv2.addWeighted(binary, 1.5, blurred, -0.5, 0)
        
        return enhanced
        
    def extract_text(self, image):
        """Extract text from the document image"""
        # Ensure image is in correct format
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Extract text using Tesseract
        text = pytesseract.image_to_string(
            image,
            config=self.tesseract_config
        )
        
        return text
        
    def detect_text_regions(self, image):
        """Detect and highlight text regions in the document"""
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Configure Tesseract to output bounding boxes
        boxes = pytesseract.image_to_boxes(
            image,
            config=self.tesseract_config
        )
        
        # Create copy for visualization
        visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw boxes around text
        for box in boxes.splitlines():
            box = box.split()
            x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
            
            # Convert coordinates (OpenCV uses top-left origin)
            y = height - y
            h = height - h
            
            # Draw rectangle around character
            cv2.rectangle(
                visualization,
                (x, h),
                (w, y),
                (0, 255, 0),
                1
            )
            
        return visualization
        
    def process_document(self, image_path):
        """Process document image end-to-end"""
        # Start timing
        self.start_time = time.time()
        
        # Read image
        image = cv2.imread(image_path)
        original = image.copy()
        
        # Preprocess image
        edges = self.preprocess_image(image)
        
        # Find document contour
        document_contour = self.find_document_contour(edges)
        
        if document_contour is None:
            raise ValueError("No document found in image")
            
        # Apply perspective transform
        warped = self.perspective_transform(
            original,
            document_contour
        )
        
        # Enhance document
        enhanced = self.enhance_document(warped)
        
        # Extract text
        text = self.extract_text(enhanced)
        
        # Detect text regions
        text_visualization = self.detect_text_regions(enhanced)
        
        # Track processing time
        processing_time = time.time() - self.start_time
        self.processing_times.append(processing_time)
        
        return {
            'original': original,
            'edges': edges,
            'warped': warped,
            'enhanced': enhanced,
            'text_regions': text_visualization,
            'text': text,
            'processing_time': processing_time
        }
        
    def analyze_document_structure(self, image):
        """Analyze document structure and layout"""
        # Get document structure using Tesseract
        data = pytesseract.image_to_data(
            image,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Initialize structure analysis
        structure = {
            'paragraphs': [],
            'lines': [],
            'words': []
        }
        
        # Process structure data
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            if int(data['conf'][i]) > 60:  # Filter by confidence
                (x, y, w, h) = (
                    data['left'][i],
                    data['top'][i],
                    data['width'][i],
                    data['height'][i]
                )
                
                # Categorize by block type
                if data['level'][i] == 4:  # Paragraph
                    structure['paragraphs'].append({
                        'text': data['text'][i],
                        'bbox': (x, y, w, h)
                    })
                elif data['level'][i] == 5:  # Line
                    structure['lines'].append({
                        'text': data['text'][i],
                        'bbox': (x, y, w, h)
                    })
                elif data['level'][i] == 6:  # Word
                    structure['words'].append({
                        'text': data['text'][i],
                        'bbox': (x, y, w, h)
                    })
                    
        return structure

def main():
    # Create scanner instance
    scanner = DocumentScanner()
    
    # Process document
    results = scanner.process_document('document.jpg')
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(231)
    plt.imshow(cv2.cvtColor(results['original'], cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(232)
    plt.imshow(results['edges'], cmap='gray')
    plt.title('Edge Detection')
    
    plt.subplot(233)
    plt.imshow(cv2.cvtColor(results['warped'], cv2.COLOR_BGR2RGB))
    plt.title('Perspective Transform')
    
    plt.subplot(234)
    plt.imshow(results['enhanced'], cmap='gray')
    plt.title('Enhanced Document')
    
    plt.subplot(235)
    plt.imshow(cv2.cvtColor(
        results['text_regions'],
        cv2.COLOR_BGR2RGB
    ))
    plt.title('Text Regions')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Extracted Text:\n{results['text']}")
    print(f"Processing Time: {results['processing_time']:.2f} seconds")

if __name__ == "__main__":
    main()

----------------------------------------------------------------------------------
## Document Classification
def classify_document(self, image, text):
    """Classify document type based on content and layout"""
    def extract_features(self, image, text):
        """Extract features for classification"""
        # Layout features
        layout = self.analyze_document_structure(image)
        layout_features = {
            'n_paragraphs': len(layout['paragraphs']),
            'n_lines': len(layout['lines']),
            'text_density': len(text) / (image.shape[0] * image.shape[1])
        }
        
        # Content features
        content_features = {
            'has_date': bool(re.search(r'\d{1,2}/\d{1,2}/\d{4}', text)),
            'has_currency': bool(re.search(r'\$\d+\.?\d*', text)),
            'has_letterhead': self.detect_letterhead(image)
        }
        
        return {**layout_features, **content_features}
        
    def classify(self, features):
        """Classify document based on features"""
        # Implement classification logic
        if features['has_letterhead'] and features['text_density'] < 0.2:
            return 'Letter'
        elif features['has_currency'] and features['text_density'] > 0.3:
            return 'Invoice'
        elif features['has_date'] and features['n_paragraphs'] > 5:
            return 'Report'
        else:
            return 'General Document'
            
    # Extract features and classify
    features = extract_features(self, image, text)
    return classify(self, features)

def detect_letterhead(self, image):
    """Detect presence of letterhead in document"""
    # Analyze top portion of document
    top_section = image[:int(image.shape[0] * 0.2), :]
    
    # Apply text detection to top section
    top_text = pytesseract.image_to_data(
        top_section,
        config=self.tesseract_config,
        output_type=pytesseract.Output.DICT
    )
    
    # Check for company indicators
    has_logo = self.detect_logo(top_section)
    has_company_name = any(
        len(word) > 3 and word.isupper()
        for word in top_text['text']
        if isinstance(word, str)
    )
    
    return has_logo or has_company_name

def detect_logo(self, image):
    """Detect presence of logo in image section"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Find contours
    contours = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )[0]
    
    # Filter contours by size and shape
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 5000:  # Typical logo size range
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2:  # Typical logo aspect ratio
                return True
                
    return False

class DocumentProcessor:
    """Advanced document processing capabilities"""
    
    def __init__(self, scanner):
        self.scanner = scanner
        self.document_history = []
        
    def batch_process(self, image_paths):
        """Process multiple documents in batch"""
        results = []
        for path in image_paths:
            try:
                result = self.scanner.process_document(path)
                doc_type = self.scanner.classify_document(
                    result['enhanced'],
                    result['text']
                )
                result['document_type'] = doc_type
                results.append(result)
                self.document_history.append({
                    'path': path,
                    'type': doc_type,
                    'timestamp': time.time()
                })
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
                
        return results
        
    def generate_searchable_pdf(self, image, text):
        """Create searchable PDF from scanned document"""
        from fpdf import FPDF
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Add image
        image_path = 'temp_image.png'
        cv2.imwrite(image_path, image)
        pdf.image(image_path, x=10, y=10, w=190)
        
        # Add invisible text layer
        pdf.set_font('Arial', '', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.set_xy(10, 10)
        
        # Split text into lines and add to PDF
        lines = text.split('\n')
        for line in lines:
            pdf.cell(0, 5, line, ln=True)
            
        return pdf
        
    def extract_form_fields(self, image, text):
        """Extract structured data from form documents"""
        # Define common field patterns
        field_patterns = {
            'name': r'Name:?\s*([A-Za-z\s]+)',
            'date': r'Date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            'email': r'Email:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'phone': r'Phone:?\s*(\d{3}[-.]?\d{3}[-.]?\d{4})',
            'address': r'Address:?\s*([A-Za-z0-9\s,]+)'
        }
        
        # Extract fields using regex
        fields = {}
        for field_name, pattern in field_patterns.items():
            match = re.search(pattern, text)
            if match:
                fields[field_name] = match.group(1).strip()
                
        return fields
        
    def detect_tables(self, image):
        """Detect and extract tables from document"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal and vertical lines
        horizontal = np.copy(gray)
        vertical = np.copy(gray)
        
        # Specify size on horizontal and vertical lines
        cols = horizontal.shape[1]
        horizontal_size = cols // 30
        rows = vertical.shape[0]
        vertical_size = rows // 30
        
        # Create structure elements
        horizontalStructure = cv2.getStructuringElement(
            cv2.MORPH_RECT, (horizontal_size, 1)
        )
        verticalStructure = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, vertical_size)
        )
        
        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        
        # Combine horizontal and vertical lines
        table_mask = cv2.add(horizontal, vertical)
        
        # Find contours of table cells
        contours, _ = cv2.findContours(
            table_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Extract cell contents
        cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cell_image = gray[y:y+h, x:x+w]
            cell_text = pytesseract.image_to_string(
                cell_image,
                config=self.scanner.tesseract_config
            )
            cells.append({
                'position': (x, y, w, h),
                'text': cell_text.strip()
            })
            
        return cells
        
    def enhance_image_quality(self, image):
        """Advanced image enhancement techniques"""
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(image)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        return sharpened
        
    def validate_document(self, text, doc_type):
        """Validate document completeness and quality"""
        validation_results = {
            'is_complete': True,
            'issues': [],
            'confidence_score': 0.0
        }
        
        # Check text extraction quality
        if len(text.strip()) < 50:
            validation_results['is_complete'] = False
            validation_results['issues'].append(
                "Low text content - possible extraction failure"
            )
            
        # Check for required fields based on document type
        if doc_type == 'Invoice':
            required_patterns = {
                'invoice_number': r'Invoice\s*#?\s*([A-Za-z0-9-]+)',
                'date': r'Date:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                'amount': r'Total:?\s*\$?\s*(\d+\.?\d*)',
            }
            
            for field, pattern in required_patterns.items():
                if not re.search(pattern, text):
                    validation_results['is_complete'] = False
                    validation_results['issues'].append(
                        f"Missing required field: {field}"
                    )
                    
        # Calculate confidence score
        words = text.split()
        valid_words = sum(
            1 for word in words
            if len(word) > 2 and word.isalnum()
        )
        validation_results['confidence_score'] = valid_words / len(words)
        
        return validation_results

def main():
    # Create scanner instance
    scanner = DocumentScanner()
    processor = DocumentProcessor(scanner)
    
    # Example batch processing
    image_paths = ['doc1.jpg', 'doc2.jpg', 'doc3.jpg']
    results = processor.batch_process(image_paths)
    
    # Process and analyze results
    for result in results:
        # Validate document
        validation = processor.validate_document(
            result['text'],
            result['document_type']
        )
        
        if validation['is_complete']:
            # Generate searchable PDF
            pdf = processor.generate_searchable_pdf(
                result['enhanced'],
                result['text']
            )
            
            # Extract form fields if applicable
            if result['document_type'] in ['Form', 'Invoice']:
                fields = processor.extract_form_fields(
                    result['enhanced'],
                    result['text']
                )
                
            # Detect and extract tables
            tables = processor.detect_tables(result['enhanced'])
        else:
            print(f"Document validation failed: {validation['issues']}")

if __name__ == "__main__":
    main()


from flask import Flask, request, render_template, jsonify, url_for
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
from threading import Lock
import numpy as np
from skimage import measure
import scipy.ndimage as ndimage
import trimesh
import matplotlib.pyplot as plt
import os
import gc
import PyPDF2  # New import for PDF extraction

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit
app.config['UPLOAD_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}  # Allowed file extensions

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize QA model
model_name = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

class Generator:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if getattr(self, '_initialized', False):
            return
            
        self.model_id = "CompVis/stable-diffusion-v1-4"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        ).to(self.device)
        
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_model_cpu_offload()
        
        self._initialized = True
        self._generation_lock = Lock()

    def generate_image(self, prompt):
        with self._generation_lock:
            try:
                with torch.inference_mode():
                    result = self.pipe(prompt, num_inference_steps=20)
                
                if not result.images:
                    return None
                    
                image = result.images[0]
                image_path = os.path.join("static", "2d_generated_image.png")
                image.save(image_path)
                
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()
                
                return "2d_generated_image.png"
            except Exception as e:
                print(f"Error generating image: {e}")
                return None

    def create_3d_model(self, prompt, depth=0.3, smoothing=1.0):
        try:
            # First generate 2D image
            with torch.inference_mode():
                result = self.pipe(prompt, num_inference_steps=20)
            
            if not result.images:
                return None
            
            image = result.images[0]
            
            # Create volume from image
            gray_img = np.array(image.convert('L')) / 255.0
            volume = np.zeros((gray_img.shape[0], gray_img.shape[1], int(gray_img.shape[0] * depth)))
            
            for i in range(volume.shape[2]):
                depth_factor = 1 - (i / volume.shape[2])
                volume[:, :, i] = gray_img * depth_factor
            
            # Apply smoothing
            if smoothing > 0:
                volume = ndimage.gaussian_filter(volume, sigma=smoothing)
            
            # Create mesh
            verts, faces, normals, _ = measure.marching_cubes(volume, level=0.2)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            mesh.vertices -= mesh.center_mass
            mesh.vertices *= 50.0 / mesh.scale
            
            # Save 3D model
            model_path = os.path.join("static", "3d_output_model.obj")
            mesh.export(model_path)
            
            # Generate preview
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(
                mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2],
                triangles=mesh.faces,
                cmap='viridis',
                shade=True,
                alpha=0.9
            )
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=20, azim=45)
            plt.tight_layout()
            
            preview_path = os.path.join("static", "3d_model_preview.png")
            plt.savefig(preview_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                'model': '3d_output_model.obj',
                'preview': '3d_model_preview.png'
            }
        except Exception as e:
            print(f"Error generating 3D model: {e}")
            return None

# Initialize generator
generator = Generator()

def allowed_file(filename):
    """Check if uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(filepath):
    """Extract text from a PDF file."""
    try:
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    context = data.get('context', '').strip()
    question = data.get('question', '').strip()
    
    if not context or not question:
        return jsonify({"error": "Context and question are required"}), 400
    
    try:
        # Get answer from QA pipeline
        qa_result = qa_pipeline(question=question, context=context)
        
        return jsonify({
            'answer': qa_result['answer'],
            'score': qa_result['score']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Save the PDF temporarily
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_document.pdf')
        file.save(filename)
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(filename)
        
        # Optional: Remove the temporary file
        os.remove(filename)
        
        if extracted_text:
            return jsonify({
                'success': True,
                'text': extracted_text
            })
        else:
            return jsonify({"error": "Could not extract text from PDF"}), 500
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    try:
        image_path = generator.generate_image(prompt)
        if image_path:
            return jsonify({
                'success': True,
                'image_url': url_for('static', filename=image_path)
            })
        return jsonify({"error": "Failed to generate image"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_3d', methods=['POST'])
def generate_3d():
    data = request.json
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    try:
        result = generator.create_3d_model(prompt)
        if result:
            return jsonify({
                'success': True,
                'model_url': url_for('static', filename=result['model']),
                'preview_url': url_for('static', filename=result['preview'])
            })
        return jsonify({"error": "Failed to generate 3D model"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, port=5000)
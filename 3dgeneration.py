import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from diffusers import StableDiffusionPipeline
import trimesh
import matplotlib.pyplot as plt
from skimage import measure
import scipy.ndimage as ndimage
import torch
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import cv2
from pathlib import Path

class ImprovedText23DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Text to 3D Generator with Textures")
        
        # Configure styles
        self.root.configure(padx=20, pady=20)
        
        # Create output directory
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize UI components
        self._init_ui()
        
        # Initialize model with optimizations
        self.load_model()
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    def _init_ui(self):
        """Enhanced UI initialization with texture controls"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Settings", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(input_frame, text="Enter text prompt:").pack(anchor=tk.W)
        self.prompt_entry = ttk.Entry(input_frame, width=50)
        self.prompt_entry.pack(fill=tk.X, pady=(5, 10))
        
        # Advanced settings frame
        settings_frame = ttk.Frame(input_frame)
        settings_frame.pack(fill=tk.X)
        
        # Depth control
        ttk.Label(settings_frame, text="Depth:").pack(side=tk.LEFT)
        self.depth_var = tk.DoubleVar(value=0.3)
        self.depth_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                   variable=self.depth_var, orient=tk.HORIZONTAL)
        self.depth_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 20))
        
        # Smoothing control
        ttk.Label(settings_frame, text="Smoothing:").pack(side=tk.LEFT)
        self.smooth_var = tk.DoubleVar(value=1.0)
        self.smooth_scale = ttk.Scale(settings_frame, from_=0.0, to=2.0,
                                    variable=self.smooth_var, orient=tk.HORIZONTAL)
        self.smooth_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Texture settings frame
        texture_frame = ttk.LabelFrame(main_frame, text="Texture Settings", padding=10)
        texture_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Texture resolution control
        ttk.Label(texture_frame, text="Texture Resolution:").pack(side=tk.LEFT)
        self.texture_res_var = tk.StringVar(value="1024")
        texture_res_combo = ttk.Combobox(texture_frame, 
                                       textvariable=self.texture_res_var,
                                       values=["512", "1024", "2048", "4096"],
                                       width=10,
                                       state="readonly")
        texture_res_combo.pack(side=tk.LEFT, padx=(5, 20))
        
        # Texture quality control
        ttk.Label(texture_frame, text="Texture Quality:").pack(side=tk.LEFT)
        self.texture_quality_var = tk.DoubleVar(value=0.8)
        self.texture_quality_scale = ttk.Scale(texture_frame, from_=0.1, to=1.0,
                                             variable=self.texture_quality_var,
                                             orient=tk.HORIZONTAL)
        self.texture_quality_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Controls section
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.generate_button = ttk.Button(controls_frame, text="Generate 3D", 
                                        command=self.generate_3d)
        self.generate_button.pack(side=tk.LEFT)
        
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(controls_frame, textvariable=self.progress_var).pack(side=tk.LEFT, padx=(10, 0))
        
        # Display frame
        self.display_frame = ttk.LabelFrame(main_frame, text="Generated Model", padding=10)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

    def load_model(self):
        """Initialize the text-to-image model with optimizations"""
        try:
            model_id = "CompVis/stable-diffusion-v1-4"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                use_safetensors=True,
                variant="fp16",
                torch_dtype=torch.float16
            )
            
            if hasattr(self.pipe, 'enable_attention_slicing'):
                self.pipe.enable_attention_slicing()
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                self.pipe.enable_sequential_cpu_offload()
            self.pipe = self.pipe.to(device)
            
        except Exception as e:
            self.progress_var.set(f"Error loading model: {str(e)}")
            self.generate_button.configure(state="disabled")

    @lru_cache(maxsize=32)
    def _create_depth_matrix(self, depth, size):
        """Cache depth matrices for better performance"""
        return np.linspace(1, 0, int(size * depth))[:, np.newaxis, np.newaxis]

    def generate_texture_coordinates(self, vertices, projection='cylindrical'):
        """Generate texture coordinates based on vertex positions"""
        if projection == 'cylindrical':
            # Cylindrical projection
            vertices = vertices - vertices.mean(axis=0)
            r = np.sqrt(vertices[:, 0]**2 + vertices[:, 2]**2)
            theta = np.arctan2(vertices[:, 0], vertices[:, 2])
            
            # Normalize coordinates
            u = (theta + np.pi) / (2 * np.pi)
            v = (vertices[:, 1] - vertices[:, 1].min()) / (vertices[:, 1].max() - vertices[:, 1].min())
            
            return np.column_stack((u, v))
        else:
            # Planar projection (fallback)
            vertices = vertices - vertices.mean(axis=0)
            u = (vertices[:, 0] - vertices[:, 0].min()) / (vertices[:, 0].max() - vertices[:, 0].min())
            v = (vertices[:, 1] - vertices[:, 1].min()) / (vertices[:, 1].max() - vertices[:, 1].min())
            return np.column_stack((u, v))

    def create_texture_image(self, original_image, texture_size=1024):
        """Create a texture image from the generated image"""
        # Resize image to desired texture size
        texture_img = original_image.resize((texture_size, texture_size), Image.Resampling.LANCZOS)
        
        # Enhance texture
        texture_array = np.array(texture_img)
        texture_array = cv2.detailEnhance(texture_array, sigma_s=10, sigma_r=0.15)
        
        # Add some variation to prevent repeating patterns
        noise = np.random.normal(0, 5, texture_array.shape).astype(np.uint8)
        texture_array = cv2.addWeighted(texture_array, 1.0, noise, 0.1, 0)
        
        # Convert back to PIL Image
        enhanced_texture = Image.fromarray(texture_array)
        
        return enhanced_texture

    def create_volume_from_image(self, image, depth_factor=0.3):
        """Create volume with improved depth handling"""
        gray_img = np.array(image.convert('L'), dtype=np.float32) / 255.0
        depth_size = int(gray_img.shape[0] * depth_factor)
        depth_matrix = self._create_depth_matrix(depth_factor, gray_img.shape[0])
        volume = gray_img[np.newaxis, :, :] * depth_matrix
        return volume.transpose(1, 2, 0)

    def create_improved_mesh(self, volume, smooth_factor=1.0):
        """Create an optimized textured 3D mesh"""
        if smooth_factor > 0:
            volume = ndimage.gaussian_filter(
                volume, 
                sigma=smooth_factor,
                mode='reflect',
                truncate=2.0
            )
        
        verts, faces, normals, _ = measure.marching_cubes(
            volume,
            level=0.2,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False,
            method='lewiner'
        )
        
        # Generate texture coordinates
        texture_coords = self.generate_texture_coordinates(verts)
        
        # Create mesh with texture coordinates
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            visual=trimesh.visual.TextureVisuals(
                uv=texture_coords,
                image=None  # Will be set later
            )
        )
        
        # Optimize mesh
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Center and scale
        mesh.vertices -= mesh.center_mass
        scale_factor = 50.0 / mesh.scale
        mesh.vertices *= scale_factor
        
        return mesh

    def display_mesh(self, mesh, texture_image):
        """Display textured mesh"""
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Create colored mesh for display
        colors = np.array(texture_image.convert('RGB'))
        uv = mesh.visual.uv
        color_idx = (uv * np.array([colors.shape[1] - 1, colors.shape[0] - 1])).astype(int)
        vertex_colors = colors[color_idx[:, 1], color_idx[:, 0]] / 255.0
        
        mesh_collection = ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            triangles=faces,
            facecolors=vertex_colors[faces].mean(axis=1),
            shade=True,
            alpha=0.9,
            antialiased=True
        )
        
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)
        ax.set_axis_off()
        plt.tight_layout(pad=0)
        
        plt.savefig(self.output_dir / 'preview.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        with Image.open(self.output_dir / 'preview.png') as img:
            img = img.resize((500, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            for widget in self.display_frame.winfo_children():
                widget.destroy()
                
            label = ttk.Label(self.display_frame, image=photo)
            label.image = photo
            label.pack()

    def generate_3d(self):
        """Enhanced 3D generation process with texturing"""
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            self.progress_var.set("Please enter a prompt")
            return
            
        def generation_task():
            try:
                self.progress_var.set("Generating image...")
                self.root.update()
                
                # Generate base image
                with torch.no_grad():
                    image = self.pipe(
                        prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5
                    ).images[0]
                
                self.progress_var.set("Creating textured 3D model...")
                self.root.update()
                
                # Create texture
                texture_size = int(self.texture_res_var.get())
                texture_image = self.create_texture_image(
                    image,
                    texture_size=texture_size
                )
                texture_image.save(self.output_dir / 'texture.png')
                
                # Create volume and mesh
                volume = self.create_volume_from_image(
                    image, 
                    depth_factor=self.depth_var.get()
                )
                
                mesh = self.create_improved_mesh(
                    volume, 
                    smooth_factor=self.smooth_var.get()
                )
                
                # Apply texture to mesh
                mesh.visual.material = trimesh.visual.material.SimpleMaterial(
                    image=np.array(texture_image)
                )
                
                # Display and save results
                self.display_mesh(mesh, texture_image)
                
                # Export textured model
                mesh.export(self.output_dir / 'textured_model.obj')
                
                # Save MTL file
                with open(self.output_dir / 'textured_model.mtl', 'w') as f:
                    f.write(f"""newmtl material0
Ka 1.000000 1.000000 1.000000
Kd 1.000000 1.000000 1.000000
Ks 0.000000 0.000000 0.000000
Ns 1.000000
map_Kd texture.png""")
                
                self.progress_var.set("Textured model generated and saved in 'output' directory")
                
            except Exception as e:
                self.progress_var.set(f"Error: {str(e)}")
        
        # Run generation in separate thread
        self.executor.submit(generation_task)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImprovedText23DApp(root)
    root.mainloop()
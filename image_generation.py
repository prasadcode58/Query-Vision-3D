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
import gc

class ImprovedText23DApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Text to 3D Generator")
        
        # Configure styles
        self.root.configure(padx=20, pady=20)
        
        # Initialize components
        self._init_ui()
        self.pipe = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize thread pool with optimal worker count
        self.executor = ThreadPoolExecutor(
            max_workers=min(os.cpu_count(), 4)
        )
        
        # Load model after UI initialization
        self.root.after(100, self.load_model)
        
    def _init_ui(self):
        # UI initialization remains the same
        # ... [Previous UI code remains unchanged]
        pass

    @torch.no_grad()
    def load_model(self):
        """Initialize the model with optimal settings"""
        try:
            # Clear any existing GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            model_id = "CompVis/stable-diffusion-v1-4"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                use_safetensors=True,
                variant="fp16" if self.device.type == "cuda" else None,
                safety_checker=None  # Disable if not needed
            ).to(self.device)
            
            # Enable optimizations
            if self.device.type == "cuda":
                self.pipe.enable_attention_slicing(slice_size="auto")
                self.pipe.enable_model_cpu_offload()
                self.pipe.enable_vae_slicing()
            else:
                self.pipe.enable_sequential_cpu_offload()
            
            # Compile model if using PyTorch 2.0+
            if hasattr(torch, 'compile') and self.device.type == "cuda":
                self.pipe.unet = torch.compile(
                    self.pipe.unet, 
                    mode="reduce-overhead",
                    fullgraph=True
                )
            
            self.progress_var.set("Model loaded successfully")
            
        except Exception as e:
            self.progress_var.set(f"Error loading model: {str(e)}")
            self.generate_button.configure(state="disabled")

    @staticmethod
    @lru_cache(maxsize=64)
    def _create_depth_matrix(depth: float, size: int) -> np.ndarray:
        """Create and cache depth matrices"""
        return np.linspace(1, 0, int(size * depth))[:, np.newaxis, np.newaxis].astype(np.float32)

    def create_volume_from_image(self, image: Image, depth_factor: float = 0.3) -> np.ndarray:
        """Create volume with optimized numpy operations"""
        # Convert to grayscale and normalize in one step
        gray_img = np.array(image.convert('L'), dtype=np.float32) / 255.0
        
        # Use cached depth matrix
        depth_matrix = self._create_depth_matrix(depth_factor, gray_img.shape[0])
        
        # Optimize memory usage with direct multiplication
        volume = np.multiply(
            gray_img[np.newaxis, :, :],
            depth_matrix,
            dtype=np.float32
        )
        
        return volume.transpose(1, 2, 0)

    def create_improved_mesh(self, volume: np.ndarray, smooth_factor: float = 1.0) -> trimesh.Trimesh:
        """Create optimized 3D mesh"""
        # Selective smoothing with optimized parameters
        if smooth_factor > 0:
            volume = ndimage.gaussian_filter(
                volume,
                sigma=smooth_factor,
                mode='reflect',
                truncate=2.0,
                output=volume  # In-place operation
            )
        
        # Optimize marching cubes
        verts, faces, normals, _ = measure.marching_cubes(
            volume,
            level=0.2,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False,
            method='lewiner'
        )
        
        # Create mesh with pre-computed normals
        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_normals=normals,
            process=False
        )
        
        # Optimize mesh in batch
        mesh.process(
            validate=False,
            merge_vertices=True,
            remove_unreferenced=True,
            remove_degenerate=True
        )
        
        # Center and scale efficiently
        mesh.vertices = (mesh.vertices - mesh.center_mass) * (50.0 / mesh.scale)
        
        return mesh

    def display_mesh(self, mesh: trimesh.Trimesh) -> None:
        """Display mesh with optimized rendering"""
        # Create figure with optimized settings
        fig = plt.figure(figsize=(10, 10), dpi=100, facecolor='white')
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot with optimized parameters
        ax.plot_trisurf(
            mesh.vertices[:, 0],
            mesh.vertices[:, 1],
            mesh.vertices[:, 2],
            triangles=mesh.faces,
            cmap='viridis',
            shade=True,
            alpha=0.9,
            antialiased=True
        )
        
        # Optimize view
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=45)
        ax.set_axis_off()
        
        # Save with optimized settings
        plt.savefig(
            'temp_plot.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0,
            optimize=True
        )
        plt.close(fig)
        
        # Load and display image efficiently
        img = Image.open('temp_plot.png')
        img = img.resize((500, 500), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        # Update display
        for widget in self.display_frame.winfo_children():
            widget.destroy()
        
        label = ttk.Label(self.display_frame, image=photo)
        label.image = photo
        label.pack()
        
        # Cleanup
        img.close()
        os.remove('temp_plot.png')

    def generate_3d(self) -> None:
        """Optimized 3D generation process"""
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            self.progress_var.set("Please enter a prompt")
            return
        
        def generation_task():
            try:
                self.progress_var.set("Generating image...")
                self.root.update()
                
                # Generate image with optimized settings
                with torch.no_grad(), torch.inference_mode():
                    image = self.pipe(
                        prompt,
                        num_inference_steps=30,
                        guidance_scale=7.5,
                        height=512,
                        width=512
                    ).images[0]
                
                self.progress_var.set("Creating 3D model...")
                self.root.update()
                
                # Process with current settings
                volume = self.create_volume_from_image(
                    image,
                    depth_factor=self.depth_var.get()
                )
                
                mesh = self.create_improved_mesh(
                    volume,
                    smooth_factor=self.smooth_var.get()
                )
                
                # Save and display
                mesh.export("output_model.obj")
                self.display_mesh(mesh)
                
                self.progress_var.set("Model generated and saved as 'output_model.obj'")
                
                # Clean up
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.progress_var.set(f"Error: {str(e)}")
        
        # Run in separate thread
        self.executor.submit(generation_task)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImprovedText23DApp(root)
    root.mainloop()
"""
Synthetic Dataset Pipeline for NVIDIA Isaac
Task: T050 [P] [US3] Create synthetic dataset pipeline examples in src/code/isaac-examples/datasets/

This module implements synthetic dataset generation pipelines using NVIDIA Isaac Sim.
It demonstrates how to create labeled datasets for training AI models in robotics applications.
"""

import os
import json
import random
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import yaml
from datetime import datetime


@dataclass
class SceneConfig:
    """Configuration for a synthetic scene"""
    name: str
    objects: List[Dict]
    lighting: Dict
    camera_positions: List[Dict]
    background: str
    domain_randomization: Dict


@dataclass
class DatasetSample:
    """Represents a single dataset sample"""
    image: np.ndarray
    annotations: Dict
    metadata: Dict
    sample_id: str


class SyntheticSceneGenerator:
    """Generates synthetic scenes with randomized parameters"""

    def __init__(self, output_dir: str = "synthetic_dataset"):
        """
        Initialize the scene generator

        Args:
            output_dir: Directory to save generated datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.scene_id = 0

    def generate_random_scene_config(self) -> SceneConfig:
        """Generate a random scene configuration"""
        # Define possible objects
        possible_objects = [
            {"type": "cube", "size_range": [0.3, 0.8], "color_range": [(0, 0, 255), (255, 0, 0)]},
            {"type": "sphere", "size_range": [0.2, 0.6], "color_range": [(0, 255, 0), (255, 255, 0)]},
            {"type": "cylinder", "size_range": [0.3, 0.7], "color_range": [(255, 0, 255), (0, 255, 255)]},
            {"type": "cone", "size_range": [0.4, 0.9], "color_range": [(128, 0, 128), (0, 128, 128)]}
        ]

        # Randomly select objects for this scene
        num_objects = random.randint(3, 8)
        objects = []
        for _ in range(num_objects):
            obj_template = random.choice(possible_objects)
            obj = obj_template.copy()
            obj["position"] = [random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0.5, 2)]
            obj["rotation"] = [random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360)]
            obj["size"] = random.uniform(*obj["size_range"])
            obj["color"] = [
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            ]
            objects.append(obj)

        # Random lighting configuration
        lighting = {
            "intensity": random.uniform(500, 1500),
            "color_temp": random.uniform(4000, 7000),
            "direction": [
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 0)
            ]
        }

        # Random camera positions
        camera_positions = []
        for _ in range(random.randint(1, 3)):
            camera_positions.append({
                "position": [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(1, 3)],
                "rotation": [random.uniform(-30, 30), random.uniform(-45, 45), random.uniform(-180, 180)],
                "fov": random.uniform(30, 90)
            })

        # Domain randomization parameters
        domain_randomization = {
            "texture_randomization": random.uniform(0.1, 0.8),
            "lighting_randomization": random.uniform(0.2, 0.9),
            "background_randomization": random.uniform(0.1, 0.6),
            "occlusion_probability": random.uniform(0.0, 0.3)
        }

        return SceneConfig(
            name=f"scene_{self.scene_id}",
            objects=objects,
            lighting=lighting,
            camera_positions=camera_positions,
            background=random.choice(["outdoor", "indoor", "warehouse", "office"]),
            domain_randomization=domain_randomization
        )

    def render_scene(self, scene_config: SceneConfig) -> List[DatasetSample]:
        """
        Render a scene configuration and return dataset samples

        Args:
            scene_config: Configuration for the scene to render

        Returns:
            List of dataset samples with images and annotations
        """
        samples = []

        for cam_idx, camera_config in enumerate(scene_config.camera_positions):
            # Create a synthetic image (in a real implementation, this would use Isaac Sim)
            width, height = 640, 480
            image = np.zeros((height, width, 3), dtype=np.uint8)

            # Add background
            if scene_config.background == "outdoor":
                image[:] = [135, 206, 235]  # Sky blue
            elif scene_config.background == "indoor":
                image[:] = [240, 240, 240]  # Light gray
            elif scene_config.background == "warehouse":
                image[:] = [100, 100, 100]  # Dark gray
            else:  # office
                image[:] = [255, 255, 240]  # Light yellow

            # Add objects to the image (simplified rendering)
            annotations = {
                "objects": [],
                "image_width": width,
                "image_height": height
            }

            for obj_idx, obj in enumerate(scene_config.objects):
                # Calculate 2D position based on 3D position and camera
                # This is a simplified projection for demonstration
                obj_x = int(width / 2 + obj["position"][0] * 50)
                obj_y = int(height / 2 - obj["position"][1] * 50)
                obj_size = int(obj["size"] * 30)

                # Ensure object is within image bounds
                obj_x = max(obj_size, min(width - obj_size, obj_x))
                obj_y = max(obj_size, min(height - obj_size, obj_y))

                # Draw object
                if obj["type"] == "cube":
                    cv2.rectangle(
                        image,
                        (obj_x - obj_size, obj_y - obj_size),
                        (obj_x + obj_size, obj_y + obj_size),
                        obj["color"], -1
                    )
                elif obj["type"] == "sphere":
                    cv2.circle(
                        image,
                        (obj_x, obj_y),
                        obj_size,
                        obj["color"], -1
                    )
                elif obj["type"] == "cylinder":
                    cv2.ellipse(
                        image,
                        (obj_x, obj_y),
                        (obj_size, int(obj_size * 0.7)),
                        0, 0, 360,
                        obj["color"], -1
                    )
                elif obj["type"] == "cone":
                    # Draw triangle for cone
                    pts = np.array([
                        [obj_x, obj_y - obj_size],
                        [obj_x - obj_size, obj_y + obj_size],
                        [obj_x + obj_size, obj_y + obj_size]
                    ], np.int32)
                    cv2.fillPoly(image, [pts], obj["color"])

                # Add annotation
                annotations["objects"].append({
                    "id": obj_idx,
                    "type": obj["type"],
                    "bbox": [obj_x - obj_size, obj_y - obj_size, obj_x + obj_size, obj_y + obj_size],
                    "centroid": [obj_x, obj_y],
                    "color": obj["color"],
                    "size": obj["size"]
                })

            # Add noise to simulate real sensor data
            noise = np.random.normal(0, 5, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Create sample ID
            sample_id = f"{scene_config.name}_cam{cam_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            samples.append(DatasetSample(
                image=image,
                annotations=annotations,
                metadata={
                    "scene_config": scene_config.__dict__,
                    "camera_config": camera_config,
                    "generation_timestamp": datetime.now().isoformat(),
                    "sample_id": sample_id
                },
                sample_id=sample_id
            ))

        self.scene_id += 1
        return samples


class DatasetFormatter:
    """Formats dataset samples for different training frameworks"""

    def __init__(self):
        pass

    def to_coco_format(self, samples: List[DatasetSample], output_path: str):
        """
        Convert samples to COCO format

        Args:
            samples: List of dataset samples
            output_path: Path to save the COCO dataset
        """
        coco_dataset = {
            "info": {
                "description": "Synthetic Robotics Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat()
            },
            "licenses": [{"id": 1, "name": "Synthetic Data License", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": []
        }

        # Create category mapping
        category_names = set()
        for sample in samples:
            for obj in sample.annotations["objects"]:
                category_names.add(obj["type"])

        categories = []
        category_id_map = {}
        for idx, name in enumerate(sorted(category_names)):
            category = {
                "id": idx + 1,
                "name": name,
                "supercategory": "object"
            }
            categories.append(category)
            category_id_map[name] = idx + 1

        coco_dataset["categories"] = categories

        # Process each sample
        annotation_id = 1
        for img_id, sample in enumerate(samples):
            # Add image info
            image_info = {
                "id": img_id + 1,
                "width": sample.annotations["image_width"],
                "height": sample.annotations["image_height"],
                "file_name": f"{sample.sample_id}.jpg",
                "license": 1,
                "date_captured": sample.metadata["generation_timestamp"]
            }
            coco_dataset["images"].append(image_info)

            # Add annotations
            for obj in sample.annotations["objects"]:
                x1, y1, x2, y2 = obj["bbox"]
                width = x2 - x1
                height = y2 - y1

                annotation = {
                    "id": annotation_id,
                    "image_id": img_id + 1,
                    "category_id": category_id_map[obj["type"]],
                    "bbox": [x1, y1, width, height],
                    "area": width * height,
                    "iscrowd": 0
                }
                coco_dataset["annotations"].append(annotation)
                annotation_id += 1

        # Save COCO dataset
        with open(output_path, 'w') as f:
            json.dump(coco_dataset, f, indent=2)

    def to_yolo_format(self, samples: List[DatasetSample], output_dir: str):
        """
        Convert samples to YOLO format

        Args:
            samples: List of dataset samples
            output_dir: Directory to save YOLO dataset
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create class names file
        all_classes = set()
        for sample in samples:
            for obj in sample.annotations["objects"]:
                all_classes.add(obj["type"])

        class_names = sorted(list(all_classes))
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        with open(output_path / "classes.txt", 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")

        # Create images and labels directories
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        for sample in samples:
            # Save image
            img_path = images_dir / f"{sample.sample_id}.jpg"
            cv2.imwrite(str(img_path), sample.image)

            # Create YOLO label file
            label_path = labels_dir / f"{sample.sample_id}.txt"
            with open(label_path, 'w') as f:
                img_width = sample.annotations["image_width"]
                img_height = sample.annotations["image_height"]

                for obj in sample.annotations["objects"]:
                    x1, y1, x2, y2 = obj["bbox"]
                    # Convert to YOLO format (normalized center coordinates and width/height)
                    center_x = (x1 + x2) / 2.0 / img_width
                    center_y = (y1 + y2) / 2.0 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    class_id = class_to_id[obj["type"]]
                    f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")


class SyntheticDatasetPipeline:
    """Complete pipeline for generating synthetic datasets"""

    def __init__(self, output_dir: str = "synthetic_dataset"):
        """
        Initialize the synthetic dataset pipeline

        Args:
            output_dir: Directory to save generated datasets
        """
        self.generator = SyntheticSceneGenerator(output_dir)
        self.formatter = DatasetFormatter()
        self.output_dir = Path(output_dir)

    def generate_dataset(self, num_scenes: int, samples_per_scene: int = 2) -> List[DatasetSample]:
        """
        Generate a synthetic dataset

        Args:
            num_scenes: Number of scenes to generate
            samples_per_scene: Number of samples per scene (different camera angles)

        Returns:
            List of generated dataset samples
        """
        all_samples = []

        print(f"Generating {num_scenes} scenes...")
        for scene_idx in range(num_scenes):
            print(f"Generating scene {scene_idx + 1}/{num_scenes}...")

            # Generate scene configuration
            scene_config = self.generator.generate_random_scene_config()

            # Render scene from multiple camera angles
            scene_samples = self.generator.render_scene(scene_config)
            all_samples.extend(scene_samples)

            # Limit samples per scene if needed
            if len(scene_samples) > samples_per_scene:
                all_samples = all_samples[:samples_per_scene]

        print(f"Generated {len(all_samples)} samples in total")
        return all_samples

    def save_dataset(self, samples: List[DatasetSample], format_type: str = "coco"):
        """
        Save the generated dataset in specified format

        Args:
            samples: List of dataset samples to save
            format_type: Format to save the dataset ("coco", "yolo", or "custom")
        """
        if format_type == "coco":
            output_path = self.output_dir / "dataset_coco.json"
            self.formatter.to_coco_format(samples, output_path)
            print(f"Dataset saved in COCO format at {output_path}")
        elif format_type == "yolo":
            output_path = self.output_dir / "yolo_dataset"
            self.formatter.to_yolo_format(samples, output_path)
            print(f"Dataset saved in YOLO format at {output_path}")
        else:
            # Save in custom format with images and annotations separately
            images_dir = self.output_dir / "images"
            annotations_dir = self.output_dir / "annotations"
            images_dir.mkdir(exist_ok=True)
            annotations_dir.mkdir(exist_ok=True)

            for sample in samples:
                # Save image
                img_path = images_dir / f"{sample.sample_id}.jpg"
                cv2.imwrite(str(img_path), sample.image)

                # Save annotation
                annot_path = annotations_dir / f"{sample.sample_id}.json"
                with open(annot_path, 'w') as f:
                    json.dump({
                        "annotations": sample.annotations,
                        "metadata": sample.metadata
                    }, f, indent=2)

            print(f"Dataset saved in custom format at {self.output_dir}")


def main():
    """Example usage of the synthetic dataset pipeline"""
    print("Initializing Isaac Synthetic Dataset Pipeline...")

    # Create dataset pipeline
    pipeline = SyntheticDatasetPipeline("isaac_synthetic_dataset")

    # Generate a small dataset for demonstration
    print("Generating synthetic dataset...")
    samples = pipeline.generate_dataset(num_scenes=5, samples_per_scene=2)

    # Save in COCO format
    print("\nSaving dataset in COCO format...")
    pipeline.save_dataset(samples, format_type="coco")

    # Save in YOLO format
    print("\nSaving dataset in YOLO format...")
    pipeline.save_dataset(samples, format_type="yolo")

    # Save in custom format
    print("\nSaving dataset in custom format...")
    pipeline.save_dataset(samples, format_type="custom")

    print(f"\nDataset generation completed! Generated {len(samples)} samples.")
    print("Dataset formats available: COCO, YOLO, and custom.")


if __name__ == "__main__":
    main()
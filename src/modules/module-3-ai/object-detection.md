---
title: "Object Detection Pipelines in Isaac Sim"
sidebar_position: 4
description: "Implementing object detection using Isaac Sim and Isaac ROS with GPU acceleration"
---

# Object Detection Pipelines in Isaac Sim

## Introduction to Object Detection in Robotics

Object detection is a critical capability for autonomous robots, enabling them to identify, locate, and classify objects in their environment. In the context of NVIDIA Isaac, object detection leverages GPU acceleration to achieve real-time performance for robotic applications. This includes detecting objects for manipulation, navigation, and situational awareness.

## Isaac Sim Object Detection Pipeline

### Synthetic Dataset Generation

Isaac Sim enables the generation of synthetic datasets for training object detection models:

```python
import omni
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.synthetic_utils.sensors import Camera
from pxr import Gf, UsdGeom
import numpy as np
import cv2
import json
from pathlib import Path

class IsaacSyntheticDatasetGenerator:
    def __init__(self, output_dir="synthetic_dataset"):
        self.output_dir = Path(output_dir)
        self.sd_helper = SyntheticDataHelper()
        self.camera = None

        # Create output directories
        self.rgb_dir = self.output_dir / "rgb"
        self.seg_dir = self.output_dir / "segmentation"
        self.labels_dir = self.output_dir / "labels"

        for directory in [self.rgb_dir, self.seg_dir, self.labels_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_camera(self, prim_path="/World/Camera"):
        """Setup camera for data capture"""
        self.camera = Camera(
            prim_path=prim_path,
            frequency=30,
            resolution=(640, 480)
        )

        # Set camera properties
        self.camera.set_focal_length(24.0)
        self.camera.set_clipping_range(0.1, 100.0)

    def setup_objects(self):
        """Setup objects for detection in the scene"""
        # Define object classes and their properties
        self.object_classes = {
            1: "robot_part",
            2: "obstacle",
            3: "target_object",
            4: "tool",
            5: "container"
        }

        # Assign semantic labels to objects in the scene
        self.assign_semantic_labels()

    def assign_semantic_labels(self):
        """Assign semantic labels to objects in the scene"""
        # This would typically iterate through objects in the USD stage
        # and assign semantic labels to them
        pass

    def capture_synthetic_data(self, num_samples=1000):
        """Capture synthetic dataset with annotations"""
        annotations = []

        for i in range(num_samples):
            # Randomize scene
            self.randomize_scene()

            # Capture RGB image
            rgb_image = self.sd_helper.get_rgb_data()

            # Capture semantic segmentation
            seg_image = self.sd_helper.get_semantic_segmentation()

            # Generate bounding boxes from segmentation
            bboxes = self.generate_bounding_boxes(seg_image)

            # Save images
            cv2.imwrite(str(self.rgb_dir / f"image_{i:06d}.png"),
                       cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(self.seg_dir / f"seg_{i:06d}.png"), seg_image)

            # Create annotation
            annotation = {
                "filename": f"image_{i:06d}.png",
                "width": rgb_image.shape[1],
                "height": rgb_image.shape[0],
                "objects": bboxes
            }
            annotations.append(annotation)

            if i % 100 == 0:
                self.get_logger().info(f"Captured {i}/{num_samples} samples")

        # Save annotations
        with open(self.output_dir / "annotations.json", 'w') as f:
            json.dump(annotations, f, indent=2)

    def randomize_scene(self):
        """Randomize scene for domain randomization"""
        # Randomize object positions
        self.randomize_object_positions()

        # Randomize lighting
        self.randomize_lighting()

        # Randomize textures
        self.randomize_textures()

        # Randomize camera pose
        self.randomize_camera_pose()

    def generate_bounding_boxes(self, seg_image):
        """Generate bounding boxes from semantic segmentation"""
        objects = []

        for class_id in np.unique(seg_image):
            if class_id == 0:  # Skip background
                continue

            # Find contours for this class
            mask = (seg_image == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)

                    obj = {
                        "class_id": int(class_id),
                        "class_name": self.object_classes.get(int(class_id), "unknown"),
                        "bbox": [int(x), int(y), int(x + w), int(y + h)],  # xmin, ymin, xmax, ymax
                        "area": int(cv2.contourArea(contour))
                    }
                    objects.append(obj)

        return objects

    def randomize_object_positions(self):
        """Randomize object positions in the scene"""
        # Move objects to random positions within bounds
        pass

    def randomize_lighting(self):
        """Randomize lighting conditions"""
        # Adjust light intensity, color temperature, direction
        pass

    def randomize_textures(self):
        """Randomize object textures and materials"""
        # Change colors, roughness, metallic properties
        pass

    def randomize_camera_pose(self):
        """Randomize camera position and orientation"""
        # Change camera position and look-at target
        pass

# Usage example
generator = IsaacSyntheticDatasetGenerator("my_object_detection_dataset")
generator.setup_camera()
generator.setup_objects()
generator.capture_synthetic_data(num_samples=5000)
```

## Isaac ROS Object Detection Packages

### Isaac ROS DetectNet

GPU-accelerated object detection using DetectNet:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import numpy as np
import torch
import torchvision.transforms as transforms

class IsaacDetectNetNode(Node):
    def __init__(self):
        super().__init__('isaac_detectnet_node')

        self.cv_bridge = CvBridge()

        # Initialize DetectNet model
        self.model = self.load_detectnet_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Input preprocessing parameters
        self.input_width = 960
        self.input_height = 544
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Confidence threshold
        self.confidence_threshold = 0.5

        # Class labels (example for robotics objects)
        self.class_labels = {
            0: 'background',
            1: 'robot_arm',
            2: 'target_object',
            3: 'obstacle',
            4: 'charger',
            5: 'person'
        }

        # Subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.detections_pub = self.create_publisher(
            Detection2DArray,
            '/isaac_ros/detectnet/detections',
            10
        )

        self.get_logger().info(f'Isaac DetectNet Node initialized on {self.device}')

    def load_detectnet_model(self):
        """Load pre-trained DetectNet model"""
        # In practice, this would load a TensorRT optimized model
        # For example, using torch.hub to load a model:
        try:
            # Load a pre-trained model (this is an example)
            # In Isaac ROS, this would typically be a TensorRT model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            # Modify for detection task
            model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(self.class_labels))
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading model: {str(e)}')
            return None

    def preprocess_image(self, image):
        """Preprocess image for DetectNet"""
        # Resize image to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        image_tensor = transforms.ToTensor()(rgb_image).float()
        image_tensor = transforms.Normalize(self.mean, self.std)(image_tensor)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def image_callback(self, msg):
        """Process incoming image for object detection"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Preprocess image
            input_tensor = self.preprocess_image(cv_image)
            input_tensor = input_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)

            # Process detections
            detections = self.process_outputs(outputs, cv_image.shape)

            # Publish detections
            self.publish_detections(detections, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error in object detection: {str(e)}')

    def process_outputs(self, outputs, image_shape):
        """Process model outputs to extract detections"""
        # This is a simplified example
        # In practice, DetectNet outputs would be processed differently
        # depending on the specific model architecture

        detections = []

        # Example: process classification outputs (this would vary by model)
        if isinstance(outputs, torch.Tensor):
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            # Filter by confidence threshold
            high_conf_mask = confidences > self.confidence_threshold

            if torch.any(high_conf_mask):
                for i in range(len(predictions)):
                    if high_conf_mask[i]:
                        detection = Detection2D()

                        # Set bounding box (placeholder - in practice, this would come from model)
                        detection.bbox.center.x = image_shape[1] / 2  # center x
                        detection.bbox.center.y = image_shape[0] / 2  # center y
                        detection.bbox.size_x = 100  # width
                        detection.bbox.size_y = 100  # height

                        # Set object hypothesis
                        hypothesis = ObjectHypothesisWithPose()
                        class_id = int(predictions[i].item())
                        hypothesis.hypothesis.class_id = str(class_id)
                        hypothesis.hypothesis.score = float(confidences[i].item())

                        detection.results.append(hypothesis)
                        detections.append(detection)

        return detections

    def publish_detections(self, detections, header):
        """Publish detection results"""
        detection_array = Detection2DArray()
        detection_array.header = header
        detection_array.detections = detections

        self.detections_pub.publish(detection_array)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacDetectNetNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS Isaac ROS Image Segmentation

### Semantic Segmentation with Isaac ROS

GPU-accelerated semantic segmentation:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import PixelwiseSegmentation
from cv_bridge import CvBridge
import numpy as np
import torch
import torchvision.transforms as transforms

class IsaacSegmentationNode(Node):
    def __init__(self):
        super().__init__('isaac_segmentation_node')

        self.cv_bridge = CvBridge()

        # Initialize segmentation model
        self.model = self.load_segmentation_model()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Input parameters
        self.input_width = 640
        self.input_height = 480

        # Class labels for robotics scene
        self.class_labels = {
            0: 'background',
            1: 'robot',
            2: 'person',
            3: 'obstacle',
            4: 'target_object',
            5: 'floor',
            6: 'wall',
            7: 'ceiling'
        }

        # Color palette for visualization
        self.color_palette = self.create_color_palette()

        # Subscribers and publishers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.segmentation_pub = self.create_publisher(
            Image,  # Segmentation result as colored image
            '/isaac_ros/segmentation/result',
            10
        )

        self.mask_pub = self.create_publisher(
            Image,  # Raw segmentation mask
            '/isaac_ros/segmentation/mask',
            10
        )

        self.get_logger().info(f'Isaac Segmentation Node initialized on {self.device}')

    def load_segmentation_model(self):
        """Load pre-trained segmentation model"""
        try:
            # Load a segmentation model (example with DeepLabV3)
            model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
            # Modify for robotics classes
            model.classifier[4] = torch.nn.Conv2d(256, len(self.class_labels), kernel_size=(1, 1))
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading segmentation model: {str(e)}')
            return None

    def preprocess_image(self, image):
        """Preprocess image for segmentation"""
        # Resize image
        resized = cv2.resize(image, (self.input_width, self.input_height))

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        image_tensor = transforms.ToTensor()(rgb_image).float()
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(image_tensor)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def image_callback(self, msg):
        """Process incoming image for segmentation"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # Preprocess image
            input_tensor = self.preprocess_image(cv_image)
            input_tensor = input_tensor.to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)

            # Get segmentation mask
            masks = outputs['out'].argmax(1)  # Get class predictions
            mask_cpu = masks[0].cpu().numpy()  # Get first image from batch

            # Create color segmentation image
            color_seg = self.colorize_segmentation(mask_cpu)

            # Publish results
            self.publish_segmentation(color_seg, mask_cpu, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error in segmentation: {str(e)}')

    def colorize_segmentation(self, mask):
        """Apply color palette to segmentation mask"""
        h, w = mask.shape
        color_seg = np.zeros((h, w, 3), dtype=np.uint8)

        for label_id, color in self.color_palette.items():
            color_seg[mask == label_id] = color

        return color_seg

    def create_color_palette(self):
        """Create color palette for segmentation classes"""
        palette = {}
        np.random.seed(42)  # For reproducible colors

        for i in range(len(self.class_labels)):
            # Generate random color
            color = np.random.randint(0, 255, 3, dtype=np.uint8)
            palette[i] = color.tolist()

        # Override some colors for better visibility
        palette[0] = [0, 0, 0]      # Black for background
        palette[1] = [0, 255, 0]    # Green for robot
        palette[2] = [255, 0, 0]    # Blue for person
        palette[3] = [255, 255, 0]  # Cyan for obstacle
        palette[4] = [255, 0, 255]  # Magenta for target

        return palette

    def publish_segmentation(self, color_seg, mask, header):
        """Publish segmentation results"""
        # Publish color segmentation
        color_msg = self.cv_bridge.cv2_to_imgmsg(color_seg, encoding='rgb8')
        color_msg.header = header
        self.segmentation_pub.publish(color_msg)

        # Publish raw mask
        mask_msg = self.cv_bridge.cv2_to_imgmsg(mask.astype(np.uint8), encoding='mono8')
        mask_msg.header = header
        self.mask_pub.publish(mask_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacSegmentationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS TensorRT Optimization

### Optimizing Object Detection with TensorRT

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch

class TensorRTOptimizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None

    def optimize_model(self, input_shape, output_shape):
        """Optimize PyTorch model with TensorRT"""
        # Create builder
        builder = trt.Builder(self.trt_logger)

        # Create network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Parse PyTorch model or ONNX model
        parser = trt.OnnxParser(network, self.trt_logger)

        # Load model as ONNX (assumes model has been exported to ONNX)
        # In practice, you would convert your PyTorch model to ONNX first
        with open(self.model_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError("Failed to parse ONNX model")

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision for better performance

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)

        # Create runtime and deserialize engine
        runtime = trt.Runtime(self.trt_logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

        # Create execution context
        self.context = self.engine.create_execution_context()

        return self.engine

    def infer(self, input_data):
        """Perform TensorRT inference"""
        # Allocate I/O buffers
        inputs, outputs, bindings, stream = self.allocate_buffers(input_data.shape)

        # Copy input data to device
        np.copyto(inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)

        # Run inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        # Copy output data to host
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        return outputs[0]['host'].reshape(self.engine.get_binding_shape(1))

    def allocate_buffers(self, input_shape):
        """Allocate input/output buffers for TensorRT inference"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for idx in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(idx)
            binding_shape = self.engine.get_binding_shape(idx)
            size = trt.volume(binding_shape) * self.engine.max_batch_size * np.dtype(np.float32).itemsize

            host_mem = cuda.pagelocked_empty(size // np.dtype(np.float32).itemsize, dtype=np.float32)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))
            if self.engine.binding_is_input(idx):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

class IsaacTRTDetectionNode:
    def __init__(self, engine_path):
        self.tensort_optimizer = TensorRTOptimizer(engine_path)
        self.engine = self.tensort_optimizer.engine
        self.context = self.tensort_optimizer.context

        # Initialize other components
        self.input_shape = (1, 3, 544, 960)  # Example shape
        self.output_shape = (1, 1000)  # Example output shape

    def detect_objects(self, image):
        """Perform object detection using TensorRT optimized model"""
        # Preprocess image
        input_data = self.preprocess_image(image)

        # Run inference
        output = self.tensort_optimizer.infer(input_data)

        # Post-process outputs
        detections = self.postprocess_outputs(output)

        return detections

    def preprocess_image(self, image):
        """Preprocess image for TensorRT inference"""
        # Resize and normalize image
        resized = cv2.resize(image, (960, 544))
        normalized = (resized.astype(np.float32) / 255.0 - 0.45) / 0.225

        # Transpose from HWC to CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched.astype(np.float32)

    def postprocess_outputs(self, output):
        """Post-process TensorRT outputs"""
        # Apply softmax
        exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Get top predictions
        top_indices = np.argsort(probabilities[0])[::-1][:10]  # Top 10 predictions
        top_probs = probabilities[0][top_indices]

        # Filter by confidence threshold
        detections = []
        for idx, prob in zip(top_indices, top_probs):
            if prob > 0.3:  # Confidence threshold
                detections.append({
                    'class_id': int(idx),
                    'confidence': float(prob),
                    'class_name': self.get_class_name(idx)
                })

        return detections

    def get_class_name(self, class_id):
        """Get class name for given class ID"""
        # In practice, this would map to your specific class labels
        class_names = {
            0: 'background',
            1: 'robot_arm',
            2: 'target_object',
            3: 'obstacle',
            4: 'charger',
            5: 'person'
        }
        return class_names.get(class_id, f'unknown_{class_id}')
```

## Isaac Sim Integration for Object Detection

### Training Pipeline Integration

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from pathlib import Path

class IsaacSimDetectionDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = Path(data_dir)
        self.transforms = transforms

        # Load annotations
        self.annotations = self.load_annotations()

    def load_annotations(self):
        """Load annotations from synthetic dataset"""
        annotations_path = self.data_dir / 'annotations.json'

        import json
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get item for training"""
        annotation = self.annotations[idx]

        # Load image
        img_path = self.data_dir / 'rgb' / annotation['filename']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes
        bboxes = torch.tensor(annotation['objects']) if 'objects' in annotation else torch.empty((0, 5))

        # Apply transforms
        if self.transforms:
            image = self.transforms(image)

        return image, bboxes

class IsaacSimTrainer:
    def __init__(self, model, dataset, output_dir="trained_models"):
        self.model = model
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Setup optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()  # Example loss

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(data)

            # Calculate loss
            loss = self.calculate_loss(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def calculate_loss(self, outputs, targets):
        """Calculate loss for object detection"""
        # This would depend on your specific model architecture
        # For example, using a combination of classification and regression losses
        pass

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                loss = self.calculate_loss(outputs, targets)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def train(self, num_epochs=10, val_split=0.2):
        """Full training process"""
        # Split dataset
        dataset_size = len(self.dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # Training loop
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')

            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save model checkpoint
            self.save_checkpoint(epoch, val_loss)

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved: {checkpoint_path}')

# Example usage
# dataset = IsaacSimDetectionDataset("synthetic_dataset")
# trainer = IsaacSimTrainer(model, dataset)
# trainer.train(num_epochs=50)
```

## Multi-Modal Object Detection

### Combining Vision and Depth for Better Detection

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d

class IsaacMultiModalDetector(Node):
    def __init__(self):
        super().__init__('isaac_multi_modal_detector')

        self.cv_bridge = CvBridge()

        # Initialize vision detector
        self.vision_detector = self.initialize_vision_detector()

        # Initialize depth processing
        self.depth_scale = 0.001  # mm to meters

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10
        )
        self.points_sub = self.create_subscription(
            PointCloud2, '/camera/depth/color/points', self.points_callback, 10
        )

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/multimodal_detections', 10
        )

        # Storage for synchronized data
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_points = None

        self.get_logger().info('Isaac Multi-Modal Detector initialized')

    def initialize_vision_detector(self):
        """Initialize vision-based object detector"""
        # In practice, this would load a trained model
        # For now, return a placeholder
        return None

    def rgb_callback(self, msg):
        """Process RGB image"""
        try:
            self.latest_rgb = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.process_multimodal_detection()
        except Exception as e:
            self.get_logger().error(f'Error processing RGB: {str(e)}')

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.process_multimodal_detection()
        except Exception as e:
            self.get_logger().error(f'Error processing depth: {str(e)}')

    def points_callback(self, msg):
        """Process point cloud"""
        try:
            # Convert PointCloud2 to Open3D format
            points = self.pointcloud2_to_array(msg)
            self.latest_points = points
            self.process_multimodal_detection()
        except Exception as e:
            self.get_logger().error(f'Error processing points: {str(e)}')

    def process_multimodal_detection(self):
        """Process multi-modal detection"""
        if self.latest_rgb is None or self.latest_depth is None:
            return

        # Run vision-based detection
        vision_detections = self.run_vision_detection(self.latest_rgb)

        # Enhance with depth information
        enhanced_detections = self.enhance_with_depth(
            vision_detections, self.latest_depth, self.latest_rgb.shape
        )

        # Publish results
        self.publish_detections(enhanced_detections)

    def run_vision_detection(self, rgb_image):
        """Run vision-based object detection"""
        # This would call the actual detection model
        # For now, return empty list
        return []

    def enhance_with_depth(self, detections, depth_image, rgb_shape):
        """Enhance detections with depth information"""
        enhanced_detections = []

        for detection in detections:
            # Get bounding box coordinates
            bbox = detection.bbox
            x, y, w, h = int(bbox.center.x - bbox.size_x/2), int(bbox.center.y - bbox.size_y/2), int(bbox.size_x), int(bbox.size_y)

            # Ensure coordinates are within image bounds
            x = max(0, min(x, depth_image.shape[1] - 1))
            y = max(0, min(y, depth_image.shape[0] - 1))
            w = min(w, depth_image.shape[1] - x)
            h = min(h, depth_image.shape[0] - y)

            # Extract depth region
            depth_region = depth_image[y:y+h, x:x+w]

            # Calculate median depth (robust to outliers)
            if depth_region.size > 0:
                median_depth = np.median(depth_region[depth_region > 0])  # Only consider valid depths
                detection.id = f"{detection.id}_depth_{median_depth:.2f}"

            enhanced_detections.append(detection)

        return enhanced_detections

    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array"""
        # This would convert the PointCloud2 message to an array
        # Implementation depends on the specific format
        pass

    def publish_detections(self, detections):
        """Publish detection results"""
        detection_array = Detection2DArray()
        detection_array.header.stamp = self.get_clock().now().to_msg()
        detection_array.header.frame_id = 'camera_color_optical_frame'
        detection_array.detections = detections

        self.detection_pub.publish(detection_array)

def main(args=None):
    rclpy.init(args=args)
    detector = IsaacMultiModalDetector()

    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization and Benchmarking

### Optimizing Object Detection Performance

```python
import time
import psutil
import GPUtil
from collections import deque

class IsaacDetectionBenchmark:
    def __init__(self, model):
        self.model = model
        self.latencies = deque(maxlen=100)
        self.throughputs = deque(maxlen=100)

        # System monitoring
        self.gpu_monitoring = GPUtil.getGPUs() if GPUtil.getGPUs() else None

    def benchmark_detection(self, test_images, warmup_runs=10):
        """Benchmark detection performance"""
        # Warmup runs
        for _ in range(warmup_runs):
            dummy_input = torch.randn(1, 3, 544, 960).cuda()
            _ = self.model(dummy_input)

        # Actual benchmarking
        for img in test_images:
            start_time = time.time()

            # Run detection
            with torch.no_grad():
                result = self.model(img)

            end_time = time.time()

            latency = (end_time - start_time) * 1000  # ms
            self.latencies.append(latency)

            # Calculate throughput
            throughput = 1000.0 / latency  # FPS
            self.throughputs.append(throughput)

        return self.get_statistics()

    def get_statistics(self):
        """Get performance statistics"""
        if not self.latencies:
            return None

        stats = {
            'avg_latency_ms': sum(self.latencies) / len(self.latencies),
            'min_latency_ms': min(self.latencies),
            'max_latency_ms': max(self.latencies),
            'avg_throughput_fps': sum(self.throughputs) / len(self.throughputs),
            'percentile_95_latency': np.percentile(list(self.latencies), 95),
            'percentile_99_latency': np.percentile(list(self.latencies), 99)
        }

        return stats

    def monitor_system_resources(self):
        """Monitor system resources during detection"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()

        # GPU usage (if available)
        gpu_stats = {}
        if self.gpu_monitoring:
            gpu = GPUtil.getGPUs()[0]  # Assuming single GPU
            gpu_stats = {
                'gpu_load': gpu.load * 100,
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_temperature': gpu.temperature
            }

        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'gpu_stats': gpu_stats
        }
```

## Summary

Object detection pipelines in Isaac Sim leverage synthetic data generation and GPU-accelerated inference to provide real-time detection capabilities for robotics applications. The integration of Isaac Sim for synthetic dataset generation and Isaac ROS for optimized inference enables rapid development and deployment of robust object detection systems. Multi-modal approaches that combine vision and depth information provide enhanced detection accuracy for robotic manipulation and navigation tasks.

## Learning Check

After completing this section, you should be able to:
- Generate synthetic datasets for object detection using Isaac Sim
- Implement GPU-accelerated object detection with Isaac ROS
- Optimize models using TensorRT for deployment
- Combine multiple sensor modalities for improved detection
- Benchmark and evaluate detection performance
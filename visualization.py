import pytorch_lightning as pl

from model import Model
from config import cfg  # Assuming the configuration is defined in config.py
import torch 

def filter_and_extract_bounding_boxes(r_map, p_map, confidence_threshold=0.5):
    filtered_boxes = []
    for box, confidence in zip(r_map, p_map):
        if confidence >= confidence_threshold:
            filtered_boxes.append(box)  # Add box if confidence is high enough
    return filtered_boxes


import cv2

def draw_boxes(image, bounding_boxes):
    for box in bounding_boxes:
        # Assuming box format is [x_min, y_min, x_max, y_max]
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        color = (255, 0, 0)  # Red color in BGR
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image

params = params = {
        "n_epochs":     15,
        "batch_size":   2,  
        "small_addon_for_BCE": 1e-6,
        "max_gradient_norm": 5, 
        "alpha_bce":    1.5,
        "beta_bce":     1.0,
        "learning_rate": 0.001,
        "mode":         "train",
        "dump_vis":     "no",
        "data_root_dir": "../",
        "model_dir":    "model",
        "model_name":   "model1",
        "num_threads":  8
    }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_instance = Model(cfg, params, device)


from pytorch_lightning.callbacks import ModelCheckpoint
from train2 import VoxelNetPL
from skyhehe_utils import get_filtered_lidar, lidar_to_bev

loaded_model = VoxelNetPL.load_from_checkpoint(
    "../LIDAR_VoxelNet/epoch=2-step=300.ckpt",
    model=model_instance
)

from data import PointCloudImageDataset

dataset = PointCloudImageDataset(point_cloud_dir='path/to/point_clouds', image_dir='path/to/images')
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for point_cloud, image in data_loader:
    point_cloud = point_cloud.to(device)
    image = image.to(device)

    # Model Prediction
    r_map, p_map = loaded_model(point_cloud)

    # Bounding Box Processing
    bounding_boxes = filter_and_extract_bounding_boxes(r_map, p_map)

    # Draw Boxes on Image
    image_np = image.cpu().numpy().transpose(1, 2, 0)  # Convert to numpy array and adjust channel order
    image_with_boxes = draw_boxes(image_np, bounding_boxes)

    # Visualization
    plt.imshow(image_with_boxes)
    plt.show()
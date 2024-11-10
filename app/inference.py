from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw
import torch
import torchvision

# Paths
model_path = "/flower_count_model/"
image_path = "../test1.jpg"  # Path to your local image file
output_image_path = "../output_with_boxes1.jpg"  # Path to save the output image

# Initialize model and processor
model = DetrForObjectDetection.from_pretrained(model_path)
processor = DetrImageProcessor(
    do_normalize=True,
    do_rescale=True,
    size={"shortest_edge": 800, "longest_edge": 1333},
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225]
)

# Set model to evaluation mode
model.eval()

# Load and preprocess image
image = Image.open(image_path).convert("RGB")  # Convert to RGB to ensure compatibility
inputs = processor(images=image, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Process outputs (box predictions and class labels)
logits = outputs.logits[0]  # Take the first image in the batch
boxes = outputs.pred_boxes[0]  # Coordinates of detected boxes for the first image

# Apply softmax to logits to get class probabilities and identify the highest probability class per box
probs = logits.softmax(-1)
scores, classes = probs.max(-1)  # Highest probability and corresponding class per box

# Filter out boxes classified as "no object" or with low confidence
confidence_threshold = 0.9
valid_indices = (classes != model.config.num_labels - 1) & (scores > confidence_threshold)
filtered_boxes = boxes[valid_indices]
filtered_scores = scores[valid_indices]
filtered_classes = classes[valid_indices]

# Apply Non-Maximum Suppression (NMS) with an IoU threshold to limit overlapping boxes
nms_threshold = 0.9  # Adjust this threshold to control IoU-based suppression
keep_indices = torchvision.ops.nms(
    filtered_boxes, filtered_scores, nms_threshold
)
final_boxes = filtered_boxes[keep_indices]
final_scores = filtered_scores[keep_indices]
final_classes = filtered_classes[keep_indices]

# Print final classes, scores, and boxes
print("Final Detections:")
for i in range(len(final_boxes)):
    print(f"Class: {final_classes[i].item()}, Score: {final_scores[i].item():.2f}, Box: {final_boxes[i].tolist()}")

# Total count of detections after filtering
total_detections = len(final_boxes)

# Draw bounding boxes on the image
draw = ImageDraw.Draw(image)
width, height = image.size
for box in final_boxes:
    # Extract and scale coordinates
    xmin, ymin, xmax, ymax = box.tolist()
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)

    # Ensure coordinates are ordered correctly
    xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    ymin, ymax = min(ymin, ymax), max(ymin, ymax)

    # Draw the bounding box
    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)

# Save the output image with bounding boxes
image.save(output_image_path)
print(f"Output image saved to {output_image_path}")
print(f"Total detections: {total_detections}")

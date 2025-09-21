import cv2
import os
print("Current working directory:",os.getcwd())
print("Files in samples/ folder:",os.listdir('samples'))
image_files=[f'samples\\Img{i}.jpeg'for i in range(1,24)]
clicked_coords=[]
def mouse_callback(event,x,y,flags,param):
 if event==cv2.EVENT_LBUTTONDOWN:
 clicked_coords.append((x,y))
 print(f"Clicked at (x={x}, y={y})")
start_x=[]
start_y=0
spacing_h=0
spacing_v=0
bubble_size=30
if not os.path.exists('dataset/raw'):os.makedirs('dataset/raw')
img_path=image_files[0]
if not os.path.exists(img_path):print(f"Error: Image not found at {img_path}.");exit()
img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
if img is None:print(f"Error: Failed to load image at {img_path}.");exit()
cv2.namedWindow('image')
cv2.setMouseCallback('image',mouse_callback)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
if len(clicked_coords)>=7:
 start_y=clicked_coords[0][1]
 start_x=[coord[0]for coord in clicked_coords[1:6]]
 spacing_h=clicked_coords[6][0]-start_x[0]
 spacing_v=clicked_coords[7][1]-start_y
 print(f"Detected coordinates - start_y: {start_y}, start_x: {start_x}, spacing_h: {spacing_h}, spacing_v: {spacing_v}")
else:print("Not enough clicks.");exit()
for img_path in image_files:
 if not os.path.exists(img_path):print(f"Image not found: {img_path}. Skipping...");continue
 img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
 if img is None:print(f"Failed to load image: {img_path}. Skipping...");continue
 question=1
 for col in range(5):
 x_base=start_x[col]
 y=start_y
 for q in range(20):
 for opt in range(4):
 x=x_base+opt*spacing_h
 bubble=img[y:y+bubble_size,x:x+bubble_size]
 if bubble.shape==(bubble_size,bubble_size):
 filename=f'dataset/raw/q{question}_opt{opt}_{os.path.basename(img_path).replace(".jpeg","")}.jpg'
 cv2.imwrite(filename,bubble)
 y+=spacing_v
 question+=1
print("Bubbles extracted to dataset/raw/.")

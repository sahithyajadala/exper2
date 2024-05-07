## Bounding_box

It is used to crop the image and draw bounding boxes on image

## instalisation

install PIL by using pip install PIL

Install the required dependencies using pip

## usage

pip install -r requirment.txt

## Explanation

place the csv file containing the images and give the image dir path 
creat a directory by using os.mkdir for store the output

functions draw_boxes for used to draw the bounding boxes
draw.rectangle([left, top, right, bottom], outline="red")

draw_crop is used to crop the images
cropped_img = image.crop((left, top, right, bottom))
          
This method takes 5 mandatory parameters:

    image: A numpy array, channel last (ie. height x width x colors) with channels in BGR order (same as openCV format).
    left: A integer representing the left side of the bounding box.
    top: A integer representing the top side of the bounding box.
    right: A integer representing the right side of the bounding box.
    bottom: A integer representing the bottom side of the bounding box.



output_dir = "/home/sahitya-jadala/Downloads/7622202030987_with_boxes"
this dir is used to save the output

input
![7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/sahithyajadala/exper2/assets/169046012/12d92a80-aa51-4f11-94d5-c00e9003b5e8)

output
![full_7622202030987_f306535d741c9148dc458acbbc887243_L_533](https://github.com/sahithyajadala/exper2/assets/169046012/68a74e9c-b61b-41d9-8073-a406fdc3fc23)












   

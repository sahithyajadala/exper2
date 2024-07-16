## Histogram

  This script is used to convert the image to graph representation using the argparse

## Installation
 install opencv,numpy,matplotlib,argparse by using pip
      pip install opencv-python
numpy
       pip install numpy
       
matplotlib

       pip install matplotlib
       pip install argparse
## Usage

Run the script "histogram.py"
parser.add_argument("--image_path", help = "Enter the path of your image")
parser.add_argument("--output_path", help = "name of output path")
give input path of image passing through the terminal

give the output path to save the image. passing through the terminal.
The script will generate a histogram plot for each color channel and display it.

## Explanation
img = cv.imread(args['image_path']) Reads the input image.
give the output path to save the image.
cv.calcHist() :Calculate histograms for each color channel.
plt.plot() : Plots the histograms using Matplotlib.

## Example
Input
![shizuka](https://github.com/sahithyajadala/expr3/assets/169046012/5de4e999-2253-4078-9ff5-aeb0822333ee)

output


![output](https://github.com/sahithyajadala/expr3/assets/169046012/c91ada1f-5a20-4e4c-bea6-065b87e0cd7f)







     


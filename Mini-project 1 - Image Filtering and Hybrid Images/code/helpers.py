# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

# --- Edited ---- #
def my_imfilter(image: np.ndarray, filtr: np.ndarray):    
    
  """
  Your function should meet the requirements laid out on the project webpage.
  Apply a filter to an image. Return the filtered image.
  Inputs:
  - image -> numpy nd-array of dim (m, n, c) for RGB images or numpy nd-array of dim (m, n) for gray scale images
  - filter -> numpy nd-array of odd dim (k, l)
  Returns
  - filtered_image -> numpy nd-array of dim (m, n, c) or numpy nd-array of dim (m, n)
  Errors if:
  - filter has any even dimension -> raise an Exception with a suitable error message.
  """
  
  assert len(filtr.shape) == 2, "Kernel has to have only 2 dimensions"
  
  # Checking that the kernel has odd dimensions
  if (filtr.shape[0] * filtr.shape[1]) % 2 == 0:
      raise ValueError("Even filter's output is undefined.")

  kernel = np.fliplr(np.flipud(filtr)) # flip
  ksize = kernel.shape
  
  # In case of one channel only is sent
  is1C = (len(image.shape) == 2)
  img = image.reshape(image.shape[0], image.shape[1], 1) if is1C else image
  
  # Padding the image
  padSize = [(i - 1) // 2 for i in ksize]
  padded_image = np.pad(img, [(padSize[0], padSize[0]), 
                                 (padSize[1], padSize[1]), (0,0)], 'reflect')
  
  ''' pixel_calc(m, n, c) - multiplys a window of padded image around
       the pixel at location (m,n,c) with the kernel. Then, sums 
       the calculated values.
  '''
  pixel_calc = lambda m, n, channel : np.sum(padded_image[m:m + ksize[0],
                    n:n + ksize[1], channel] * kernel)
  # Vectorizing the pixel_calc function
  convolve2D = np.vectorize(pixel_calc)
  
  # pixels' locations' vectors
  imgSize = img.shape
  m_s = np.arange(imgSize[0]).reshape(-1, 1, 1)
  n_s = np.arange(imgSize[1]).reshape(1, -1, 1)
  channels = np.arange(imgSize[2]).reshape(1, 1, -1)
  
  # Applying it to our image
  filtered_image = convolve2D(m_s, n_s, channels)
  return (filtered_image if not is1C else filtered_image[:,:,0])

def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float):
  """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.

   Task:
   - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
   - Combine them to create 'hybrid_image'.
  """

  assert image1.shape == image2.shape
  
  def create_gaussian_filter(side_length, sigma):
      # side length represent the length of the kernerl assuming square kernel.
      ax = np.linspace(-(side_length - 1) / 2., (side_length- 1) / 2., side_length)
      xx, yy = np.meshgrid(ax, ax)

      kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

      return kernel / np.sum(kernel)

  # Steps:
  # (1) Remove the high frequencies from image1 by blurring it. The amount of
  #     blur that works best will vary with different image pairs
  # generate a gaussian kernel with mean=0 and sigma = cutoff_frequency,
  # Just a heads up but think how you can generate 2D gaussian kernel from 1D gaussian kernel
  kernel = create_gaussian_filter(23, cutoff_frequency)

  # Your code here:
  low_frequencies = my_imfilter(image1, kernel)

  # (2) Remove the low frequencies from image2. The easiest way to do this is to
  #     subtract a blurred version of image2 from the original version of image2.
  #     This will give you an image centered at zero with negative values.
  # Your code here #
  high_frequencies = image2 - my_imfilter(image2, kernel) # Replace with your implementation

  # (3) Combine the high frequencies and low frequencies
  hybrid_image = low_frequencies + high_frequencies # Replace with your implementation

  # (4) At this point, you need to be aware that values larger than 1.0
  # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
  # and all values larger than 1.0 to 1.0.
  hybrid_image = np.clip(hybrid_image, 0.0, 1.0)

  return low_frequencies, high_frequencies, hybrid_image

def vis_hybrid_image(hybrid_image: np.ndarray):
  """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales + 1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))
    # downsample image
    cur_image = rescale(cur_image, scale_factor, mode='reflect', multichannel=True)
    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))
  return output

def load_image(path):
  return img_as_float32(io.imread(path))

def save_image(path, im):
  return io.imsave(path, img_as_ubyte(im.copy()))

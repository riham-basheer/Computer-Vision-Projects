import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist
from skimage.filters import sobel_h, sobel_v
from skimage.feature import peak_local_max

def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """
    '''
    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    sigma = 1
    k = 0.04
    xs = np.zeros(1)
    ys = np.zeros(1)
    # Step 1 : Computing x and y derivatives of the image
    Ix = sobel_h(image)
    Iy = sobel_v(image)

    # Step 2 : Products of derv at each pixel. And also filtering with gaussian
    Ixx = gaussian_filter(Ix * Ix, sigma=sigma)
    Iyy = gaussian_filter(Iy * Iy, sigma=sigma)
    Ixy = gaussian_filter(Ix * Iy, sigma=sigma)
    
    # AT EACH PIXEL, we will :
    # Step 3 : Compute the sums of the products of derivatives
    m, n = image.shape
    offset = 3 // 2
    
    R = np.zeros((m - 2*offset, n - 2*offset))
    print(R.shape)
    for i in range(offset, n - offset):
        for j in range(offset, m - offset):
            Sxx = np.sum(Ixx[i - offset:i + offset + 1, j - offset:j + offset + 1])
            Syy = np.sum(Iyy[i - offset:i + offset + 1, j - offset:j + offset + 1])
            Sxy = np.sum(Ixy[i - offset:i + offset + 1, j - offset:j + offset + 1])
            """ Step 4 : Defining correlation matrix
            H(i,j) = [ Sxx(i,j)  Sxy(i,j)
                       Sxy(i,j)  Syy(i,j) ]
            """
            # Step 5 : Computing the response of the detector
            #      where R = det(H) - k*[trace(H)]^2
            R[j,i] = Sxx*Syy - Sxy**2 - k * (Sxx + Syy)**2
                
    R = Ixx * Iyy - Ixy ** 2 - k * (Ixx * Iyy) ** 2
    # Step 6 : 
        # Step 6.1 : Thresholding 
        #if R > 0:       # Only corners
    R[R <= 0] = 0
        # Step 6.2 : Non-max suppression
    corners = peak_local_max(R, threshold_rel = 0.2, 
                               exclude_border = True,
                               num_peaks = 2000,
                               min_distance = offset)
    xs = corners[:, 1]
    ys = corners[:, 0]

    '''
    return xs, ys


def get_features(image, x, y, feature_width):
    """
    Returns feature descriptors for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """
    assert feature_width % 4 == 0, "(feature_width) is not a multiple of 4"
        
    features = []
    R, C = image.shape
    offset = feature_width // 2
    CELL = feature_width // 4
    N_BINS = 8
    
    # Computing gradients, magnitudes, and angles for the whole image
    Ix = sobel_h(image)
    Iy = sobel_v(image)
    magnitudes = np.sqrt(Ix**2 + Iy**2)
    angles = np.angle(np.arctan2(Iy, Ix), deg=True)

    x = np.round(x).astype(int).flatten()
    y = np.round(y).astype(int).flatten()

    for i, j in zip(x, y):
        
        mag_wnd = magnitudes[j - offset + 1:j + offset + 1, 
                             i - offset + 1:i + offset + 1]
        angle_wnd = angles[j - offset + 1:j + offset + 1,
                           i - offset + 1:i + offset + 1]
        if mag_wnd.shape != (feature_width, feature_width):
            # skip this; out of boundarues
            continue

        # Reshaping the flatten patchs' magnitudes and angles
        patches_mags = np.array(np.array_split(np.array(np.array_split(mag_wnd, 
                     CELL, axis=1)).reshape(CELL, -1), CELL, axis=1)).reshape(-1, feature_width)
        patches_angles = np.array(np.array_split(np.array(np.array_split(angle_wnd, 
                     CELL, axis=1)).reshape(CELL, -1), CELL, axis=1)).reshape(-1, feature_width)
        #print(patches_mags)
        #print(patches_mags.shape)
        
        feature = []
        
        # Going through each patch to create the histogtram by:
        for patch in range(len(patches_mags)):
            # 1. making patche's angles values disrete [8 bins]
            bin_indx = np.digitize(patches_angles[patch], np.arange(0, 360, 360 // N_BINS))
            #print(bin_indx)
            bin_v = np.zeros(N_BINS)
            
            # 2. how much for each bin? 
            for curr_bin in range(N_BINS):
                bin_v[curr_bin] = \
                    np.sum(patches_mags[patch].flatten()[np.where(bin_indx == curr_bin)])
            feature.append(bin_v)
            # 3. Normalization
        feature = np.array(feature).flatten()
        features.append(feature / feature.sum())
    features = np.array(features)
    print(features.shape)
    return features




def match_features(im1_features, im2_features):
    """
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """
    # Number of features
    N = min(len(im1_features), len(im2_features))
    matches = np.zeros((N, 2))
    confidences = np.ones(N)
    threshold = 0.80
    
    print(N)
    print(im1_features.shape, im2_features.shape)
    
    #print(distances)
    #print(distances.shape)
    dist = np.zeros((len(im1_features), len(im2_features)))
    for i in range(N):
        dist = np.sqrt( ((im1_features[i] - im2_features)**2).sum(axis=1) )
        sorted_indices = np.argsort(dist)
        NNDR = np.nan_to_num(dist[sorted_indices[0]]/dist[sorted_indices[1]])
        
        if (NNDR < threshold):
            matches[i] = np.array([i, sorted_indices[0]])
            confidences[i] = 1 - NNDR 
    print(confidences)
    print(matches.shape)
    
    return matches, confidences

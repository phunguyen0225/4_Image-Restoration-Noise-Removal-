import numpy as np
import math
import cv2


class Filtering:

    def __init__(self, image, filter_name, filter_size, var=None):
        """initializes the variables of spatial filtering on an input image
        takes as input:
        image: the noisy input image
        filter_name: the name of the mask to use
        filter_size: integer value of the size of the mask
        alpha_d: parameter of the alpha trimmed mean filter
        order: parameter of the order for contra harmonic"""

        self.image = image

        if filter_name == 'arithmetic_mean':
            self.filter = self.get_arithmetic_mean
        elif filter_name == 'geometric_mean':
            self.filter = self.get_geometric_mean
        if filter_name == 'local_noise':
            self.filter = self.get_local_noise
        elif filter_name == 'median':
            self.filter = self.get_median
        elif filter_name == 'adaptive_median':
            self.filter = self.get_adaptive_median

        self.filter_size = filter_size
        self.global_var = var

        self.S_max = 15

    def get_arithmetic_mean(self, roi):
        """Computes the arithmetic mean filter
        takes as input:
        kernel: a list/array of intensity values
        returns the arithmetic mean value in the current kernel"""
        total = 0
        for i in range(len(roi)):
            total += roi[i]
        return total / (len(roi))

    def get_geometric_mean(self, roi):
        """Computes the geometric mean filter
                        takes as input:
                        kernel: a list/array of intensity values
                        returns the geometric mean value in the current kernel"""
        total = 1
        for i in range(len(roi)):
            total *= roi[i]
        return math.pow(total, (1/len(roi)))

    def get_local_noise(self, kernel, roi):
        """Computes the result of local noise reduction
                        takes as input:
                        kernel: a list/array of intensity values
                        returns result of local noise reduction value of the current kernel"""
        """
        S_xy = local region
        O_n = variance of the noise corruption f(x,y) to form g(x, y), variance of overall noise
        O_L = local variance of the pixel in S_xy
        m_L = local mean of the pixels in S_xy
        """
        O_n = self.global_var
        O_L = np.var(roi)
        m_L = np.mean(roi)

        ans = 0

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                ans = self.image(
                    i, j) - (O_n / O_L) * (self.image(i, j) - m_L)

        return ans

    def get_median(self, roi):
        """Computes the median filter
        takes as input:
        kernel: a list/array of intensity values
        returns the median value in the current kernel
        """
        ans = np.median(roi)
        return ans

    def get_adaptive_median(self, roi):
        """Computes the adaptive median filtering value
        Note: Adaptive median filter may involve additional steps, you are welcome to create any additional functions as needed, 
        and you can change the signature of get_adaptive_median function as well.
                        takes as input:
        kernel: a list/array of intensity values        
        returns the adaptive median filtering value"""

        output = 0
        if (self.filter_size > self.S_max):
            return np.median(roi)  # z_med

        # stage A
        z_med = np.median(roi)
        z_min = np.min(roi)
        z_max = np.max(roi)

        A1 = z_med - z_min
        A2 = z_med - z_max

        if (A1 > 0) and (A2 < 0):
            # stage B
            z_xy = int(roi([self.filter_size/2][self.filter_size/2]))  # center
            B1 = z_xy - np.min(roi)
            B2 = z_xy - np.max(roi)

            if (B1 > 0) and (B2 < 0):
                output = z_xy
                return output
            else:
                output = z_med
                return output

        else:
            self.filter_size += 2

        self.get_adaptive_median(roi)  # recursion

    def filtering(self):
        """performs filtering on an image containing gaussian or salt & pepper noise
        returns the denoised image
        ----------------------------------------------------------
        Note: Filtering for the purpose of image restoration does not involve convolution.
        For every pixel in the image, we select a neighborhood of values defined by the kernel and apply a mathematical
        operation for all the elements with in the kernel. For example, mean, median and etc.

        Steps:
        1. add the necesssary zero padding to the noisy image that way we have sufficient values to perform the operations
        on the border pixels. The number of rows and columns of zero padding is defined by the kernel size
        2. Iterate through the image and every pixel (i,j) gather the neighbors defined by the kernel into a list (or any data structure)
        3. Pass these values to one of the filters that will compute the necessary mathematical operations (mean, median, etc.)
        4. Save the results at (i,j) in the ouput image.
        5. return the output image

        Please note that the adaptive median filter may involve additional steps, you are welcome to create any additional functions as needed, 
        and you can change the signature of get_adaptive_median function as well.
        """
        """"""
        # 1. Padding
        pad_row = self.filter_size // 2
        pad_col = self.filter_size // 2

        (row, col) = (self.image.shape[0] + pad_row *
                      2, self.image.shape[1] + pad_col * 2)

        img = np.zeros((row, col))
        img[pad_row:img.shape[0] - pad_row,
            pad_col:img.shape[1] - pad_col] = self.image
        outputImage = np.zeros(img.shape)
        # 2. gather the neighbors defined by the kernel into a list
        for i in range(pad_row, img.shape[0] + pad_row + 1):
            for j in range(pad_col, img.shape[1] + pad_col + 1):
                window = img[i-pad_row:i+pad_row+1, j-pad_col:j+pad_col+1]
                roi = window.flatten()
                # 3. Pass the value to one of the filters that will compute the necesssary mathematical opertaion
                # 4. Save the results at (i,j) in the output image
                outputImage[i, j] = self.filter(roi)

        # 5. return the output image
        return outputImage

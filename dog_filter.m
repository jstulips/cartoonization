% Difference of Gaussians
%http://www.mathworks.com/matlabcentral/newsreader/view_thread/283112

function G = DOG_Filter(A, w, s1, s2)

 blueBand = double(A(:,:,3));

 windowSize = w;
 numberOfImagesProcessed = 0;
 for sigma1 = s1 %sigma1 - tweak range as needed.increase this value for thicker edges
 for sigma2 = s2 %sigma2 - tweak range as needed.
 if sigma1 ~= sigma2
 sigma1
 sigma2
 g1 = fspecial('gaussian', windowSize, sigma1);
 g2 = fspecial('gaussian', windowSize, sigma2);
 if sigma1 > sigma2
 hFilter = g1 - g2;
 else
 hFilter = g2 - g1; 
 end


 % ----------------------- Actual filtering is done here !!!!!!!!!!!! --------------------------------------------------------------
 filteredImageArray = imfilter(blueBand, hFilter);

 thresholdValue = 2; 
 positivePixelMaskArray = filteredImageArray > thresholdValue;
 pixelsAtOrAboveThreshold = find(positivePixelMaskArray);
 % Get the mean of only those pixels at or above the threshold.
 positivePixelImage = filteredImageArray .* positivePixelMaskArray;
 numberOfImagesProcessed = numberOfImagesProcessed + 1;
 meanGL = mean(positivePixelImage(pixelsAtOrAboveThreshold));
 stdGL = std(positivePixelImage(pixelsAtOrAboveThreshold));
 results(numberOfImagesProcessed, 1) = numberOfImagesProcessed;
 results(numberOfImagesProcessed, 2) = sigma1;
 results(numberOfImagesProcessed, 3) = sigma2;
 results(numberOfImagesProcessed, 4) = meanGL;
 results(numberOfImagesProcessed, 5) = stdGL;
 results(numberOfImagesProcessed, 6) =  length(pixelsAtOrAboveThreshold);

 end
 end
 end

 G = positivePixelImage;


 %===================================================================
 function [normalizedImageArray gl1Percentile gl99Percentile] = SaveAs8Bit(imageArray, fullImageFileName, colorMapToUse, figureName, intOpenNewImage)
 
 if isa(imageArray, 'uint32') || isa(imageArray, 'double')
 minGL = min(min(imageArray));
 maxGL = max(max(imageArray));
 % Create a double image in the 0-1 range for taking the histogram.
 dblImageArray = (double(imageArray) - double(minGL)) /  double(maxGL);
 [minGL maxGL gl1Percentile gl99Percentile] =  PlotHistogram(dblImageArray);
 % Create an image in the 0-255 range (for saving later as a tiff  file)
 % Rescale it to use gl99Percentile instead of maxGL.
 dblImageArray = (double(dblImageArray) - double(gl1Percentile)) / (double(gl99Percentile) - double(gl1Percentile)) * 255.0;
 else
 [minGL maxGL gl1Percentile gl99Percentile] =  PlotHistogram(imageArray);
 % Create an image in the 0-255 range for saving as a tiff file
 %normalizedImageArray = im2uint8(imageArray);
 % Create an image in the 0-255 range for saving as a tiff file
 dblImageArray = (double(imageArray) - double(gl1Percentile)) / (double(gl99Percentile) - double(gl1Percentile)) * 255.0;
 end

 % Since we rescaled the intensities, the values of gl99Percentile and
 % gl1Percentile are not the same for dblImageArray as they were for
 % ImageArray. We need to reset them for the imshow, or else use
 % imshow(imageArray,[0 255]);
 gl1Percentile = 0;
 gl99Percentile = 255.0;

 % Clip to 255.
 dblImageArray(dblImageArray > 255) = 255.0;
 % Handle outside mask areas, which will now be negative. Make them 0
 % instead. Clip to zero
 dblImageArray(dblImageArray < 0) = 0;
 % Now convert to 8 bit.
 normalizedImageArray = uint8(dblImageArray);

 % Display the intensity-normalized image, if specified.
 if intOpenNewImage > 0
 % Bring up a new window and give it the specified name.
 hfig = figure;
 set(hfig, 'Name', figureName);
 % Put the image array into the figure window.
% imshow(normalizedImageArray,[gl1Percentile gl99Percentile]);
 colormap(colorMapToUse);
 end
 imwrite(normalizedImageArray, fullImageFileName);
 return; % SaveAsTiff
 %=====================================================================
 % Plots the histogram of imgArray in axes axesImage
 % If imgArray is a double, it must be normalized between 0 and 1.
 function [minGL maxGL gl1Percentile gl99Percentile] =  PlotHistogram(imgArray)
 % Get a histogram of the entire image.
 % Use 1024 bins.
 [COUNTS, GLs] = imhist(imgArray, 256); % make sure you label the  axes after imhist because imhist will destroy them.
 % GLs goes from 0 (at element 1) to 256 (at element 1024) but only a  fraction of
 % these bins have data in them. The upper ones are generally 0. Find the last
 % non-zero bin so we can plot just up to there to get better  horizontal resolution.
 maxBinUsed = max(find(COUNTS));
 % Get subportion of array that has non-zero data.
 COUNTS = COUNTS(1:maxBinUsed);
 GLs = GLs(1:maxBinUsed);
 % The first bin is not meaningful image data and just wrecks the  scale of the
 % histogram plot, so zero that one out. This is because it's a huge
 % spike at zero due to masking.
 COUNTS(1) = 0;

 minBinUsed = min(find(COUNTS));
 if isempty(minBinUsed)
 % No pixels were selected - the entire image is zero.
 minGL = GLs(maxBinUsed);
 maxGL = GLs(maxBinUsed);
 minBinUsed = maxBinUsed;
 else
 % There is some spread to the histogram.
 minGL = GLs(minBinUsed);
 maxGL = GLs(maxBinUsed);
 end

 summed = sum(COUNTS);
 cdf = 0;
 gl1Percentile = minGL; % Need in case the first bin is more than 1%  in which case the if below will never get entered.
 for bin = minBinUsed : maxBinUsed
 cdf = cdf + COUNTS(bin);
 if cdf < 0.01 * summed
 gl1Percentile = GLs(bin);
 end
 if cdf > 0.99 * summed
 break;
 end
 end
 gl99Percentile = GLs(bin);

 return; % PlotHistogram